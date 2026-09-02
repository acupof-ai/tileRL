# Capturing the draft is rejected: the tick is 88% GPU-bound — V100 (sm70), 2026-09-02

> Status: REJECTED, and it corrects a number this branch published twice. "Capture
> `_draft_step` for 1.18× (50.3 → 59.5 tok/s)" was an analogy, not a measurement.
> Measured GPU-busy is **58.5 ms of a 66.46 ms tick = 88%**, so removing *every*
> host launch caps at **1.14×**, and 71% of the GPU time is the fp4 GEMV, which
> capture does not make faster.

## Context

`_draft_step` is called after `_run_decode_graph` returns (engine.py:836), so all
D draft forwards run outside the captured graph. From that plus a byte roofline —
a draft forward streams 954 MB, 1.06 ms at 900 GB/s, against 5.53 ms measured —
I wrote "5.2×, where fully-captured dense decode sits at 1.7× of its own floor.
Capturing the trunk was worth 2.66×; the same factor gives 5.53 → 2.08 ms and
50.3 → 59.5 tok/s."

Every step of that is a plausible inference and none of it was measured.

## Root Cause of the wrong estimate

**The byte floor was the wrong floor.** Per-kernel attribution of one depth-3
tick:

| kernel | n | µs ea | ms |
|---|---:|---:|---:|
| `linear_fp4_gemv_sm70_m` | 332 | 125.0 | **41.49** |
| `index_elementwise` (torch) | 252 | 15.8 | 3.97 |
| `gdn_chunk_fused` | 48 | 52.4 | 2.51 |
| everything else | ~1900 | — | 10.53 |
| **GPU-busy** | | | **58.5** |

A draft forward issues ~9 GEMV launches (332 total = ~305 trunk verify + 3 × 9
draft). At the measured 125 µs each that is **1.12 ms of GPU time per draft
forward** — against the 1.06 ms byte floor the whole argument rested on. The GEMV
at M=1 is *launch-shaped*, not byte-shaped: its cost is set per launch, and the
floor I compared to assumed the 954 MB was the constraint.

So the 5.53 ms is not "1.06 ms of work plus 4.5 ms of host overhead". Most of it
is GPU time that a graph replays at exactly the same speed.

**The ceiling, computed from what capture can actually remove:** host-exposed time
is at most `66.46 − 58.5 = 8.0 ms` (12%). Perfect capture of every launch gives
66.46 → 58.5 ms = 50.3 → 57.1 tok/s = **1.14×**, and that is the unreachable
bound, not a target.

## Verdict

**Reject.** 1.14× is the ceiling, and reaching any of it means restructuring
`_draft_step` around what a CUDA graph forbids:

| blocker | why capture can't take it as-is |
|---|---|
| `plan` built from `r.hidden is None` / phase | host control flow, per tick |
| `w = max(hi-lo+1)`, `nb = max(len(r.blocks))` | shapes vary per tick |
| `np.zeros((n,w))`, `torch.zeros(n,nb)`, `torch.cat(hs)` | host allocation per tick |
| `live = [i for … if hi+j < blocks*16]` | data-dependent, per depth step |
| `tk.tolist()` / `cf.tolist()` | device→host sync |
| `verify_lens(survival(confs))` | host policy on synced values |

That is a bucketed-shape rewrite of the whole draft path for at most 12%, most of
which the remaining host work would keep. Against it, the same tick has 41.49 ms
(71%) sitting in one kernel at 125 µs per launch — that is where the tick is, and
it is task #26's territory.

## Rule

**A roofline is only a bound if it is the binding constraint.** 954 MB / 900 GB/s
is arithmetic, but nothing established that bandwidth was what limited a GEMV at
M=1. Measured, the same work is 9 launches × 125 µs, and the byte floor sat a
factor of 5 below the real one — which is exactly the size of the "overhead" I
then attributed to the host.

Second: **"the same factor applies" is not a measurement.** Capturing the trunk
was worth 2.66×; transferring that ratio to the draft assumed the two have the
same host/GPU split, which is the very thing in question. A borrowed ratio should
never be written as a number with a tok/s attached — it propagated into two docs
and a task before anyone profiled it.

Third: **a profiler that disagrees with the end-to-end number by 1.8× has one
usable output and it is not the wall.** This run reads 121.5 ms/tick against a
known 66.46 (`torch.profiler` serializes launches), so its wall and its "52%
host" line are instrument artifacts and are not quoted here. GPU-busy is a sum of
kernel durations and survives: 58.5 ms against the real 66.46 is the ratio that
decided this.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---:|
| 2026-09-02 | c307971 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | GPU-busy, depth-3 tick | **58.5 ms** |
| 2026-09-02 | c307971 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | GPU-bound share | **88%** |
| 2026-09-02 | c307971 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | fp4 GEMV share of GPU | 71% |
| 2026-09-02 | c307971 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | capture ceiling | **1.14×** |
| 2026-09-02 | (withdrawn) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | claimed capture prize | 1.18× |

The withdrawn 1.18× and its "5.2× of the byte floor" premise are corrected in
`engine.py` and `wins/2026-09-02-draft-is-two-thirds-of-a-spec-tick.md`.

Also worth keeping: **252 launches of torch's `index_elementwise` at 3.97 ms** are
the third-largest GPU item and are not model math — they are the indexing in
`_draft_step`'s chain bookkeeping and `_verify`. Cheaper than the GEMV work but
larger than `gdn_chunk_fused`, and unlike capture they are removable without a
rewrite.

Raw artifact: `scripts/prof_draft_kernels.py` (its `--source` was silently unused
and it fetched the placeholder hub id; fixed here, and `_build_model` now takes
`source=` so the next script cannot repeat it).
