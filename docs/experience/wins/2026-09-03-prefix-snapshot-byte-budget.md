# The prefix store owns the state snapshot and evicts it by bytes — cuda(H20), 2026-09-03

> Status: Shipped — measured on the 27B, H20 GPU 6, both trees on the same card
> in one window, 2026-09-03.

## Context

A prefix-boundary snapshot is a clone of one slot's recurrent state plus its
conv window. On the 27B that is **149.6 MiB**, not the 74.81 MiB the old comment
claimed: 48 GDN layers x 48 heads x 128 x 128 in **f32** is 144 MiB, and the
conv window `[48, 3, 10240]` f32 adds 5.63 MiB. The 74.81 MiB figure is the
bf16 number CPU and metal see; `precision.dtype("recurrent_state", cuda)`
returns f32 because the sm90 fused GDN kernel is f32-IO.

Before this change the snapshots lived in `Engine._prefix_state`, a dict keyed
by token tuple, released only through the store's `on_evict` hook. Two
consequences:

- **`NoPrefixStore` never evicts**, so a training engine never released one.
- **`PrefixStore` bounded entries by count only** (4096), which on the 27B is
  4096 x 149.6 MiB = **598 GiB** of snapshots on a 96 GiB card.

`_publish_prefix` fires at every 16-token boundary — once at the end of a
block-aligned prefill and again from `_commit` every 16 generated tokens — so
the training leak is proportional to tokens generated, not to requests served.

## What Worked

Make the entry the unit. `_Entry` carries `(tokens, blocks, state, nbytes)`,
`insert` takes the snapshot, `lookup` returns it on the hit, and one FIFO
eviction releases blocks and snapshot together. The bound becomes three
conditions instead of one: entry count, `state_bytes` (default 8 GiB), and
block pressure. `NoPrefixStore.insert` returns False and keeps nothing. The
engine-side dict and the `on_evict` hook are deleted.

`insert` returning a bool is what lets `prefix_published` stay honest — a
duplicate is not a publication, and a store that retains nothing publishes
nothing.

## Results

27B NVFP4, H20 GPU 6, `decode_graph=False`, 8 state slots, 2048 KV blocks.
200 distinct 32-token prompts at `max_new_tokens=2` (one publication each),
`scripts/bench_prefix_state.py`. Both trees on the same card minutes apart;
the after arm was run twice and reproduced to the byte.

| tree | store | published | evicted | snapshots held | snapshot bytes | peak GiB |
|---|---|---:|---:|---:|---:|---:|
| main `a702c9a` | `PrefixStore` (4096 entries) | 200 | 0 | 200 | 29.22 | 55.36 |
| main `a702c9a` | `NoPrefixStore` (training) | 200 | 0 | **200** | **29.22** | 87.80 abs |
| this branch | `PrefixStore` (8 GiB) | 200 | 146 | **54** | **7.89** | 34.17 |
| this branch | `NoPrefixStore` (training) | 0 | 0 | **0** | **0.00** | 26.28 abs |

The weights alone are 22.92 GiB on device. So on main a 200-prefix serving run
pays 29.22 GiB of snapshots — 1.3x the weights — and a *training* engine, which
can never serve a prefix hit, pays exactly the same. After: 7.89 GiB served
(54 x 149.6 MiB, the most 8 GiB holds) and 0 in training. The byte budget is
the binding constraint, not the entry count: 146 of 200 entries were evicted
with 3946 entry slots still free.

### The training leak, at rollout shape

A third arm in the same process: 8 requests generating 128 tokens each through
a `NoPrefixStore` engine — the shape a GRPO rollout has.

| tree | outcome |
|---|---|
| main `a702c9a` | `torch.OutOfMemoryError` **inside `_publish_prefix`**, at `self._states.states[req.state_slot].clone()`, "tried to allocate 144.00 MiB", 109 MiB free of 95.22 GiB |
| this branch | completes; 0 snapshots, 0.00 GiB, peak 3.36 GiB |

The 144.00 MiB the allocator names is the f32 state alone, which is the
arithmetic above confirmed from the other side.

Honest reading of that OOM: this arm ran third in one process, so on main it
inherited the two earlier arms' 58.4 GiB of snapshots — those engines were
dropped and did not release (below). It is a cumulative demonstration, not
"1024 generated tokens fill a card". What it does settle is the failure *site*
and that on main nothing above the engine can recover the memory, while the
same script on this branch runs all three arms to completion.

It is also not the traceback P1 died with. `/work/p1.log` OOMed at GRPO step 1
inside `autograd.master_linear` -> `backend.linear_fp4` with 95.15 GiB in use;
the leak had taken headroom that step then needed, but how much is not
measured.

**Dropping the engine did not free its snapshots on main.** The second arm
started with 55.36 GiB still allocated — the first arm's engine had been
dereferenced and `torch.cuda.empty_cache()` called. On this branch the second
arm started at 22.92 GiB, the weights alone. The difference is
`prefix_store.on_evict = lambda tokens: self._prefix_state.pop(tokens, None)`,
which closes over the engine and makes store -> lambda -> engine -> store a
cycle refcounting cannot break. Inferred from the allocator baseline, not from
a gc trace, but it is the only cycle the two trees differ by.

## Gates

`test_prefix_state_budget_evicts` (byte budget evicts, the survivor's snapshot
still comes back on lookup), `test_noprefix_store_retains_no_snapshot` (fails
on main), `test_prefix_hit_survives_evicting_its_own_entry` (a hit that outlives
its own FIFO entry). `TILERL_TARGET=cpu uv run pytest`: 172 passed, 4 skipped.

## Rule

A cache entry owns everything that dies with it. A side table keyed by the same
tuple and released through a callback is not a cache — it is a leak with a
hook, and the hook makes the owner uncollectable besides.

Size a per-entry cost in the dtype the target actually uses: the same snapshot
is 74.81 MiB on CPU and 149.6 MiB on sm90, and the entry-count cap was written
against the smaller one.

Raw artifacts: `scripts/bench_prefix_state.py`, pod logs `/work/prefixafter.log`,
`/work/prefixbefore.log`, `/work/prefixafter2.log`, `/work/prefixbefore2.log`.
