# KV tiering on the V100: design, before any code

Read of `PagedKvPool`, `LinearStatePool`, `PrefixStore` and the two engine call
sites, plus four measurements on the pod. One conclusion up front, because it
changes what the first PR should be.

## The headline: the disk tier does not pay on THIS pod; on an NVMe host it is untested

Both numbers are measured, not assumed.

| | figure | how |
|---|---:|---|
| `/data00` sequential read | **189 MB/s** | `dd` 1 GiB `iflag=direct` |
| `/data00` sequential write | **229 MB/s** | `dd` 1 GiB `oflag=direct` |
| pinned DRAM→HBM | **11.52 GiB/s** | `copy_` of a 149.6 MiB snapshot, 20 iters |
| pinned HBM→DRAM | **12.26 GiB/s** | same, reversed |
| unpinned DRAM→HBM | **6.12 GiB/s** | same buffer without `pin_memory` |
| PCIe link | **Gen3 x16** | `nvidia-smi --query-gpu=pcie.link.gen.current` |
| host RAM | **31 GiB total, 25 available** | `free -g` |
| both block devices | **`ROTA 1`** — no NVMe | `lsblk -d -o NAME,SIZE,ROTA` |

I had been about to design against a 2 GB/s NVMe assumption. The real device is
**10.6x slower** than that, and it is not flash.

**The pinned figure is now measured, not estimated.** 11.52 GiB/s against the 12
GB/s I estimated — 1.03x, so every projection below that used the estimate stands.
And **pinning is worth 1.88x on H2D, 2.76x on D2H**, which was also an untested
claim in the first draft of this page.

The other number that matters is one this session measured on the card: an
11019-token prompt re-prefills in **163 s** (14.68 ms/token). That is what any
tier has to beat.

## What a hit costs to reload, per tier

An entry is `ceil(tokens/16)` KV blocks at **2.125 MiB** plus **one GDN snapshot at
149.6 MiB** (f32 on sm70: 48 linear layers × 48 heads × 128² state, plus one
conv-window plane).

| tokens | entry size | DRAM→HBM (**measured** 11.52 GiB/s) | disk→HBM (**measured** 189 MB/s) | re-prefill |
|---:|---:|---:|---:|---:|
| 2048 | 421.6 MiB | 36 ms | **2231 ms** | 30065 ms |
| 11019 | 1613.7 MiB | **137 ms** | **8538 ms** | 161759 ms |
| 32768 | 4501.6 MiB | 382 ms | **23818 ms** | 481034 ms |

A **snapshot alone** — which is what the first tier moves — is **12.7 ms**, measured
directly rather than divided out of the table.

Even at 189 MB/s the disk beats re-prefilling by **13-20x**, and the breakeven is
**57 tokens**. So the disk tier "works" arithmetically. The reason not to build it
is the next table.

## Why DRAM is the whole design and disk is a rounding error

Host RAM is **31 GiB against a 32 GiB card**. The DRAM tier is not a large backing
store behind a small cache; it is **slightly smaller than HBM**.

| entry size | entries in 25 GiB DRAM |
|---:|---:|
| 2048-token (421.6 MiB) | 60.7 |
| 11019-token (1613.7 MiB) | 15.9 |
| 32768-token (4501.6 MiB) | **5.7** |

At the context we now serve, the entire DRAM tier holds **under 6 entries**. A
disk tier behind that adds capacity at 45x the latency, on a spinning device that
is also where the checkpoint, the logs and `~/pytmp` live — and `/` is currently
**100% full, 0 bytes**, so the write path has one volume available and it is the
one under memory pressure already.

**Recommendation: build the DRAM tier first; the disk tier's value is on the H20,
not here.** Not "defer for simplicity" — the measurement says its
capacity-per-latency is poor on **this specific hardware**, a spinning device on a
host whose `/` is already 100% full. The same code on a Gen4/NVMe host is a
different verdict, and the H20's `/work` has not been measured yet. So the disk
tier ships, and every number it reports from this pod is a **lower bound** on what
the target host does: 189 MB/s is what a `ROTA 1` device gives, not what the
feature is worth.

## The snapshot, not the KV, is what to tier first

This is the finding I did not expect, and it reorders the work.

| tokens in entry | snapshot share of entry bytes |
|---:|---:|
| 512 | **68.8%** |
| 2048 | 35.5% |
| 11019 | 9.3% |
| 32768 | 3.3% |

At short prefixes the GDN snapshot *is* the entry. And a snapshot is published
every `BLOCK_TOKENS`, so short entries are the common case, not the rare one.

**It is already the binding constraint on the live server.** `build_engine` sets
`state_bytes = mem_get_info()[0] // 4` (`engine.py:1280`). Measured after the 32K
restart, free-after-pools is 6788 MiB, so `state_bytes` is **1697 MiB = 11.3
snapshots**, and a decode publishes one every 16 tokens — the store starts
evicting after roughly **181 generated tokens**, on state bytes, with KV blocks
still available.

So a DRAM tier that holds *only snapshots* raises the store's reach from 11 to
**171 entries in 25 GiB** — **15.1x** — for **12.7 ms of measured transfer per
hit**, touching no KV path at all. Against the 163 s that re-prefilling the 11019-
token prompt costs, that is **12,860x**; against the 19.4 s one-time JIT it also
avoids, **1,530x**. It is a much smaller change than block tiering and it relieves
the limit that actually binds.

## Measured after the fact: every eviction was state bytes, and that changed the shape

The section above predicted this; the live server confirmed it and the confirmation
reordered the code. With `/health` reporting `prefix_evictions` and
`prefix_state_bytes`, one 4-turn chat at `--grow 40 --max-tokens 200`:

| | value |
|---|---:|
| published | 54 |
| **evictions** | **43** |
| state bytes | 1646 MiB |
| block pool | 728 of 2048 used — **64% free** |

**Every one of the 43 evictions ran with two thirds of the block pool idle.** So none
of them needed the blocks back; each threw away a reusable prefix to reclaim a
snapshot byte the host could have held.

That rules out the obvious design. Three shapes were considered:

| shape | outcome |
|---|---|
| demote the snapshot **when the entry is evicted** | dead — `_evict_one` removes the entry from `_entries`, and `lookup` only reads `_entries`, so the demoted snapshot is unreachable |
| split eviction into two kinds | `PrefixStore` would carry blocks-without-state, state-without-blocks and whole entries — three states where there was one |
| **demote INSTEAD of evicting, under state pressure** | the entry keeps its tokens, its blocks and its index slot; only `_Entry.state` moves. One new field, one new branch |

The third ships. `_demote_one` moves the LRU resident snapshot to the host and marks
the entry; `lookup` promotes it back on the hit; entries are still evicted outright
when the **block pool** is the pressure, which is the one case a snapshot tier cannot
help with. `clear()` needs no new path — it drives `_drop`, which calls the tier's
`forget`, so an optimizer step invalidates DRAM with HBM and a stale snapshot cannot
outlive its weights.

A promotion that comes back empty (the tier's own byte LRU dropped it) **drops the
entry rather than serving it**. Blocks without a snapshot would run the GDN layers
from a zero state over non-zero KV: wrong, and silent in exactly the way the
truncated-entry bug was — byte-identical output with the snapshot norm off 1.86x.

Isolated arms, six prefixes against room for two snapshots:

| arm | entries | evictions | hits |
|---|---:|---:|---:|
| no tier | 2 | 4 | 2/6 |
| DRAM | 6 | 0 | **6/6** |

## Verdict: it wins when concurrent sessions exceed the HBM snapshot budget, and only then

I first read this as a structural dead end — "demotion picks the LRU end, a lookup
always wants the MRU one, so they never meet" — and that reading is **wrong**. It holds
*within* one conversation. Across conversations the LRU end **is** some other
conversation's newest entry, so a lookup reaches it as soon as there are more live
conversations than the budget holds.

Rotating N conversations against a fixed 9-snapshot budget:

| conversations | promotions | demotions |
|---:|---:|---:|
| 2 | 0 | 0 |
| 4 | 0 | 3 |
| **9** | **17** | 35 |
| 12 | 24 | 51 |
| 20 | 40 | 91 |

And at 12 conversations the tier is the difference between no reuse and full reuse:

| arm | hits | evictions |
|---|---:|---:|
| no tier | **0** | 63 |
| DRAM | **24** | 0 |

So the value proposition is **concurrent sessions > HBM snapshot budget** — 9 users on
this V100. A single-user endpoint never reaches it, which is exactly what the earlier
measurements showed: one conversation gave 43 demotions and 0 promotions, and two
interleaved gave 16 and 0. Those are not the tier failing; they are two sessions
against a budget of nine.

The earlier `state_bytes` sweep says the same thing read correctly — "only a budget of
2 produces promotions" *at two conversations* is the same rule with the other operand
fixed.

Cost when it runs without winning: **161.9 ms per demotion in situ**, 14.0x the
11.55 ms an idle-card probe reports, and **1.51x total wall clock** on the single 4-turn
chat. Only 7.0 s of that 68.7 s is the copies; the rest is contention in the prefill
they interrupt.

**Ships `dram_bytes=0`, opt-in for multi-user serving.** Off is right for the
single-session endpoint this pod runs and wrong above 9 concurrent sessions, and the
flag is the only thing that distinguishes them. What is still unmeasured is the V100
wall clock at 12 sessions — the CPU numbers above are hit counts, not time.

**RL rollouts are not a second case for it:** within a group the shared prompt is the
MRU entry, and the store is cleared between steps, so nothing ages out and returns.

**Consequence for the disk tier:** a runtime LRU-demotion disk tier inherits the same
condition. The cold-start path does not — after a restart HBM is empty, so *every*
lookup reaches back. Build the disk tier as write-through plus restart recovery
(rescan the directory, rebuild the index, serve the first hit), not as a third rung
under the DRAM LRU.

See [errors/2026-09-05-a-two-variable-condition-read-as-a-dead-end.md](experience/errors/2026-09-05-a-two-variable-condition-read-as-a-dead-end.md).

## Layout per tier

**HBM (today, unchanged).** `PagedKvPool` planes `[num_planes, num_blocks,
num_kv_heads, 16, head_dim]`; block id names the same page in every plane;
`refcount` makes a block shared between a live slot and the store. A published
block is always whole, so there is no copy-on-write — `PrefixStore.insert`
enforces that (`kv_cache.py:349`) and any tier must preserve it.

**DRAM.** One pinned buffer per tiered entry, not a pool: `torch.empty(...,
pin_memory=True)`. Snapshot first (contiguous already — `states[slot].clone()`),
KV blocks second as a gathered `[n_blocks, planes, heads, 16, head_dim]` copy so
the reload is one `copy_(non_blocking=True)` per entry rather than one per block.
Pinned is required for the async copy to overlap; unpinned H2D on this link
measured elsewhere at roughly half rate. **Cost to be honest about:** pinned
memory is not swappable, and at 31 GiB total, pinning 25 would destabilise the
host. Cap the tier at a configured byte budget well under available, default
conservative.

**Disk.** Designed, not built (above). If it lands later: one file per entry under
a `--kv-tier <dir>`, `snapshot || blocks` in the same order as the DRAM layout, so
the two tiers share one serialiser and the only difference is the destination.

## When the tier is touched

Prefill only, exactly as briefed, and there are only two points:

**Demote** — inside `PrefixStore._evict_one` (`kv_cache.py:398`). Today it frees
blocks and drops the snapshot. Tiered, it copies to DRAM first, then frees. This
is the only place bytes leave HBM, and it already runs under the engine lock.

**Promote** — in `Engine.submit` between `_match_prefix` and the retain loop
(`engine.py:476-496`). A DRAM hit has no blocks yet, so it must allocate
`total_blocks`, H2D into them, restore the snapshot, and only then proceed as a
normal hit. Note this *increases* block pressure at exactly the moment
`evict_until_free` runs, so promotion has to be attempted **after** the eviction
call has room, or it will thrash: promote, and if allocation fails, fall through
to a miss rather than evicting to make room for a promotion.

Nowhere else. Decode never touches it; `clear()` (`kv_cache.py:410`) must drop
DRAM entries too, or a training step would serve KV from stale weights — that is
the on-policy gate and it is the one correctness risk in this design.

## LRU per tier

Today's policy is **FIFO**, not LRU (`_fifo: deque`, `popleft`), and `lookup`
does not touch recency. For tiering, HBM→DRAM demotion order should be LRU or
FIFO demotes the entry a caller is about to ask for again. That is a small change
— stamp a counter on hit, evict the minimum — but it is a behaviour change to a
shipped path, so it belongs in its own PR with its own before/after hit rate,
**not** folded into the tier PR where it would be invisible.

## Baseline discipline

Every number in the before-arm must be taken at **`--max-ctx 32768`**, not the
4096 that shipped until today. 8x of the headroom came from a flag; folding it
into offload's credit would overstate offload by that factor.

Also worth recording as a baseline: two 11019-token requests left the store
holding **1384 of 2048 blocks (67.6%)**, `prefix_published: 6`. A third such
request needs 689 blocks against 664 free, so it evicts. That is the pressure a
tier is meant to relieve, and it is measured rather than projected.

## What I have not established

- Whether promotion can overlap the prefill it saves. If yes the 137 ms is hidden
  entirely; if the copy must complete before the first attention call, it is on
  the critical path. This is a code-reading question about the engine's stream
  usage, not a measurement.
- Hit rate. Every figure here is the cost of a hit, and none of it is worth
  anything if the workload does not re-read prefixes. The two shapes to measure
  are the chat client's growing prefix and a rollout group's shared prompt; both
  are real, neither is measured yet. `prefix_hits: 0 / misses: 2` is all the
  evidence that exists so far.
- The **bandwidth figures were taken with the 27B server resident and idle**, so
  they are lower bounds on an idle card. Co-tenancy can only make them worse,
  which is the safe direction for a design that rests on them.
