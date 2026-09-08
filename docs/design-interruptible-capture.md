# Interruptible capture: what SGLang's BCG is, and what it buys here

**Status:** approach note, no code. tileRL read from `origin/main` (`61c733c`);
SGLang from `sgl-project/sglang` `5bebe7a03`. Card 6 is busy — every tileRL
number below is quoted from an entry, a docstring or a code comment, not
re-measured. What needs a card is listed in §5.

---

## 1. BCG is segmented capture, not abortable capture

ckl's "bcg 可中断的捕获" is **Breakable CUDA Graph**. "Breakable" is not "you may
abort a capture in progress" — it is "the captured region may be **broken into
segments**". Primitive docstring
(`srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`):

> capture a region as a sequence of torch.cuda.CUDAGraph segments separated by
> eager break points.

Three parts:

- **`eager_on_graph(enable, capture_stub=None)`** wraps a callable. During
  capture it ends the current segment, optionally runs a rank barrier, runs the
  body **once eagerly** so its outputs get fixed addresses, appends a `replay_fn`
  that re-runs the real body and copies into that bridge buffer, and opens the
  next segment. At replay the wrapper is bypassed; `_break_fns[i]` runs instead.
- **`BreakableCUDAGraph.replay()`** walks `segments[i].replay()` then
  `break_fns[i]()` in order — real graph replays with real eager code between.
- **Shared mempool.** All segments capture into one pool; each segment's
  `CUDAGraph` destructor calls `releasePool`, so the pool stays pinned while any
  segment lives. That is what keeps `weak_ref_tensor` views of intermediates
  valid without Python-managed bridge buffers.

**PRs:** #27436 introduced it (diffusion DiTs), #29458 made it default, #28173
ROCm, #30273 XPU, #30586 moved it into `model_executor/runner_backend_utils`,
#31987 extended it to dsa and the DeepEP a2a backend.

**It is not diffusion-only.** `model_executor/cuda_graph_config.py:112`:

```python
def default_prefill_backend() -> str:
    """BCG (breakable) is the prefill default on CUDA only; ..."""
    return Backend.BREAKABLE if is_cuda() else Backend.TC_PIECEWISE
```

So on CUDA, **BCG is how SGLang captures prefill**, and `decode` stays
`Backend.FULL` — one whole graph, no segments. That split is the whole finding
of this note.

**And BCG does not move capture off the request path.** SGLang does that
separately: capture is "an explicit, idempotent `capture()` call (driven at
warmup) so that serving never triggers a fresh capture" — which is exactly what
`engine.py:1005 precapture()` already does here. Both readings of ckl's phrase
are answered before we write any code: abort-mid-capture is not what BCG does,
and off-the-request-path we already have (with one hole, §3).

---

## 2. What BCG actually breaks on, and the one eager region we share

The break points in SGLang's model path are **attention**, not exotica:

| break point | file |
|---|---|
| unified attention (+ lse variant) | `layers/radix_attention.py:584,587` |
| linear attention (mamba/gdn shape) | `layers/radix_linear_attention.py:232` |
| MLA bmm + attention | `models/deepseek_common/.../forward_mla.py:1029` |
| DeepEP a2a (`capture_stub` zeroes the output) | `layers/moe/ep_moe/layer.py:215` |
| fp8 rope + store_kv | `layers/attention/hpc_ops_backend.py:700` |

Two distinct reasons, and only the first is ours:

1. **Prefill attention's shape is per-batch.** The prefill runner captures "the
   transformer body with one request slot, then replays it with live batch
   metadata; multi-request prefill is supported by running attention metadata and
   the LM-head/logits tail outside the captured body". BCG is what makes a
   *prefill* graph possible at all — everything shape-static gets captured,
   attention runs eager between segments.
2. **Rank-coupled collectives cannot be captured.** DeepEP NORMAL, hard 100 s
   timeout; the `capture_stub` records the buffer address and skips the a2a.

Our tick, from `origin/main`:

| eager region | where | cost |
|---|---|---|
| **mixed / prefill ticks** | `engine.py:8-10` — "mixed ticks and every other target run eager" | **~10x slower per tick** than a replay (`engine.py:1257` comment) |
| capture failed for any `(B,W)` | `engine.py:947` sets `_decode_graph_on = False` permanently | every tick eager from then on |
| un-fused KV write | `kv_cache.py:139` — "2 syncs/layer (the two tolist)" | fallback targets only, 4 syncs/tick |
| draft step | after the graph | not captured |

(The sampler's own host syncs were the fourth region until 2026-08-30: 7→4 greedy
and 15→4 at B=8 temp>0, and the **3 of 7 that ran on every target went to 0**.
`sample_batch` now reads temperature/top_p/seed on the host. Listed because it is
the shape of fix that removes a break point instead of capturing around it.)

**So we share reason 1 and not reason 2.** We have no collective inside the
forward today — TP/CP is #185's line of work, and when a rank-coupled all-reduce
lands in the decode forward, BCG stops being optional. What we have now is the
prefill case: **the single largest eager cost in a tileRL tick is exactly the one
SGLang built BCG to capture.**

The honest caveat: our prefill is chunked and mixed with decodes in the same
forward, so "capture the body, break at attention" is a bigger change here than
in SGLang, where prefill is its own runner. I have not costed it.

---

## 3. `precapture` has a hole — capture can still land on a request

`precapture` walks `graph_keys()` =
`{(_graph_bucket(rows), w) for rows in 1..max_batch}`; at `max_batch=8` the
buckets are `{1,2,4,8}`, times the widths spec adds. Measured on the V100
(`wins/2026-09-02-precapture-the-decode-graphs.md`): **8 graphs in 19 s** with a
warm tilelang cache, **208 s cold**; a B=8 spec config captured **16 graphs in
155 s** (`errors/2026-09-03-batching-is-non-monotone-padding-rows-cost-3x.md`).
The docstring's "~14 s each" matches the late captures that warmup missed there
(14.0 s and 11.7 s).

But `_run_decode_graph` can ask for a bucket `graph_keys` never enumerates:

```python
B = self._graph_bucket(n)
if n < B and self._pad_slot is None:
    try:
        self._pad_slot = self._states.alloc_slot()
        self._pad_block = self._kv.alloc_block()
    except RuntimeError:
        B = n  # no spare capacity to park padding rows on: exact size
g = self._graph_for(B, W, keep=bool(chains))   # captures NOW if absent
```

`n=3` gives `B=3`, and `3 ∉ {1,2,4,8}` — `graph_keys` only ever enumerates
buckets, so `_graph_for` captures on the spot, **inside a live request**.

How reachable: the pad row is normally reserved in `__init__` (`engine.py:369`),
so this needs the constructor's `alloc_slot`/`alloc_block` to have failed —
"pools sized without the spare", i.e. `num_slots == max_batch` — and the runtime
retry to fail too. Not the default configuration, and I have not seen it fire.
But it is the one place "capture never happens on the request path" is still
false, and it does not need BCG to fix: either enumerate exact sizes in
`graph_keys` as well, or make that branch `return False` and run the tick eager.
The second is one line, and trades ~10x on one tick for a full capture on one
tick.

---

## 4. The RL recapture is probably avoidable outright

27 asked whether the optimizer writes weights in place at fixed addresses. **It
does.** `autograd.py`, `AdamW.step_one` ends:

```python
p.copy_(p32.to(p.dtype))      # in place; p.data_ptr() unchanged
```

A captured graph holds `p`'s address, `copy_` writes through it — on that count
alone a replay would see the new weights with no recapture.

**What breaks is the cached cast.** `backend.py:1374 _const_f32`:

```python
key = (t.data_ptr(), pad_to, dtype)
...
elif ver == t._version:
    return c
c = self._dev(t, dtype)                 # NEW allocation, NEW address
if pad_to is not None and pad_to != c.shape[0]:
    c = torch.nn.functional.pad(c, (0, pad_to - c.shape[0]))
self._const_f32_cache[key] = (weakref.ref(t), t._version, c)
```

`copy_` bumps `t._version`, the cache misses, and the converted copy is
re-materialised **at a different address**. The graph baked the old one. That is
the `# ponytail: no recapture after training — the graph bakes the f32 embed
cast` marker at `engine.py:234`, and it is not just the embedding — `_const_f32`
has **27 call sites** in `backend.py`.

So the fix is not a capture scheme:

> **Make the cached cast reuse its buffer's storage.** `c.copy_(self._dev(t,
> dtype))` on a version bump instead of rebinding `c`. The address a graph baked
> stays valid, and an in-place optimizer step needs **no recapture at all**.

`invalidate_weights()` currently drops every graph per step, re-captured lazily,
so the real cost is (buckets the run touches) × ~14 s per step — not the whole
ladder, and not zero. The change touches one function in the kernels package and
nothing in `engine.py`. Untested; §5.1.

---

## 5. Which option removes which cost

| option | startup (19 s warm / 208 s cold, 8 graphs) | in-request capture (§3) | RL per-step recapture | prefill eager ~10x |
|---|---|---|---|---|
| BCG (segmented capture) | no | no | no | **yes — this is its purpose** |
| stable-address `_const_f32` | no | no | **yes, entirely** | no |
| exact-size bucket → eager (§3) | no | **yes**, 1 line | no | no |
| background/async capture | **yes** (moves it off startup) | partly | partly | no |
| revert `precapture` to lazy | yes, but | no | no | no — costs 1088 ms/token cold, which is what `precapture` fixed |

**Recommended order, cheapest first:** §3 (one line, closes the only remaining
in-request capture, and it is a correctness hole rather than a perf item) → §4
(≈10 lines, removes the RL recapture entirely if it holds) → BCG, and only
against a measured prefill/mixed-tick share of wall clock, because it is the only
item here that is a real project.

### What needs a card

1. **`_const_f32` stable-address experiment.** Capture a graph, run an optimizer
   step, replay, compare against an eager forward on the new weights. Must fail
   today; must pass with the storage reused.
2. **Recapture seconds inside one GRPO step**, against the measured 34.09 s/step
   (`wins/2026-09-05-recapture-after-update.md`, n=10 pooled over both arm
   orders). The startup side is already measured (§3); this half is not.
3. **Prefill share of wall clock**, before anyone scopes BCG. The ~10x is a code
   comment; what decides BCG is what fraction of a serving second is spent in
   mixed ticks.
