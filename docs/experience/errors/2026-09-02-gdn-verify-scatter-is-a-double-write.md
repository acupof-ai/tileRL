# The GDN verify scatter is a 604 MB double write — V100 (sm70), 2026-09-02

> Status: measured and priced, **not fixed**. The fix is a kernel-ABI change worth
> 1.038× on the spec path; `linear_fp4_gemv_sm70_m` is 62% of the same tick.
> Recorded so the next person does not re-derive it.

## Context

Region attribution (errors/2026-09-02-differencing-attributed-the-trunk-to-the-draft.md)
put 6.39 ms of a depth-3 tick's torch time in `trunk.gdn`, with `_index_put_impl_`
and `aten::index` the two largest ops. "240 index launches at 10.5× their byte
cost" reads like launch overhead. Adding `record_shapes=True` — one flag — says it
is not.

## What the shapes say

| region / aten op / input shapes | n | ms |
|---|---:|---:|
| `_index_put_impl_` `[4,4,48,128,128]` ← `[4,48,128,128]` | 48 | **2.154** |
| `aten::index [4,48,128,128]` | 48 | 0.663 |
| `_index_put_impl_ [4,4,3,10240]` ← `[4,3,10240]` | 48 | 0.289 |
| `aten::index [4,2,3,10240]` | 48 | 0.267 |
| `aten::index [4]` | 48 | 0.193 |

The big one is `state_scatter`'s `step_states[slots, layer, :ks] = new_state`
(reference.py:875). `[4, 4, 48, 128, 128]` is the pool plane — 4 slots × 4 spec
steps × 48 value heads × 128 × 128 — and the value is one layer's 4 step states:

    4 steps × 48 heads × 128 × 128 × 4 B = 12.58 MB per layer
    × 48 GDN layers                      = 604 MB written per verify tick

Read + write = 1.21 GB, which is **1.34 ms at 900 GB/s** against 2.154 measured.
**1.6× off its own bytes, not 10.5×** — this is bandwidth, and no amount of
launch-count work touches it. The per-launch reading was an average over five ops
whose sizes span 16 B to 12.58 MB; averaging those is how a 12 MB memcpy came to
look like index arithmetic.

`aten::index [4]` at the other end *is* pure launch cost: 48 launches for 16 bytes
(`parity[slots]`), 4 µs each, and `state_gather` and `state_scatter` each compute
it, so 96 launches per tick for one 4-element read.

## The redundancy

`_gdn_chunk_fused` passes `step_states` into the kernel as an operand
(backend.py:920-947) and the kernel writes it (kernels_gdn.py:380,
`StepStates[bb, t, vh, j, tv] = state_local[j]`). Then `state_scatter` copies that
scratch into the pool. **The state is written twice**, and the second write is the
2.154 ms.

The fix already has a precedent one function away: `gdn_decode_fused` takes
`pool.states`, `slots_i` and `int(layer)` and writes the pool in place. Mirroring
that convention in `gdn_chunk_fused` — add `Slots` and `LayerIdx`, write
`StepStates[Slots[bb], LayerIdx, t, ...]` — deletes the scatter and its 1.21 GB,
and the same operands let the kernel read `states[Slots[bb], LayerIdx]` directly,
deleting the 0.663 ms gather too.

## Why it is not fixed here

| lever | prize on the 66.46 ms depth-3 tick |
|---|---|
| delete the scatter's double write (2.154 + 0.289 ms) | **1.038×** |
| delete every torch op in `trunk.gdn` (6.39 ms) | 1.106× (ceiling) |
| `linear_fp4_gemv_sm70_m` — 332 launches, 41.49 ms | **62% of the tick** |

1.038× costs a kernel signature change across `kernels_gdn.py`, `backend.py` and
`model.py`, plus a parity gate, plus the CPU twin (which reaches the same
`state_scatter` through `reference.gdn_forward` and would need the scatter kept —
a conditional on which path wrote the pool). That is a real ABI change for 3.7%
while 62% of the tick sits in one kernel that is 4.1× off its own FLOP floor
(task #26).

Task #32 carries the lever with these numbers attached.

## Rule

**A per-launch cost is meaningless across ops of different sizes.** Five ops
averaged to "10.5× off bytes"; resolved by shape, one is 1.6× off bytes (a real
memcpy, bandwidth-bound) and another is 4 µs for 16 bytes (real launch overhead).
Those two have opposite fixes, and the average points at neither. `record_shapes`
is one flag.

Second: **when a kernel already takes an output buffer as an operand, check who
writes it before optimizing the copy that follows.** The 2.154 ms is not a slow
scatter, it is a scatter that should not exist.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-02 | 13f61c1 | V100 | cuda sm70 | qwen38-27b d3 | `state_scatter` step_states | 604 MB, 2.154 ms |
| 2026-09-02 | 13f61c1 | V100 | cuda sm70 | qwen38-27b d3 | its byte floor at 900 GB/s | 1.34 ms (1.6×) |
| 2026-09-02 | 13f61c1 | V100 | cuda sm70 | qwen38-27b d3 | `parity[slots]`, 16 B | 96 launches, 0.19 ms |
| 2026-09-02 | 13f61c1 | V100 | cuda sm70 | qwen38-27b d3 | full-fix ceiling | 1.038× |

No runtime change in this entry; nothing to bench.
