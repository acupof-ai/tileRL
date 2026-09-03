# Prefill's chunk loop was quadratic in its own output — V100 (sm70), 2026-09-02

> Status: Shipped. Prefill 8.35 → 7.89 ms/prompt token at 512, 5.3-5.8% at every
> context. Decode untouched by construction.

## Context

After the rung-32 flag fix, prefill is 8.35-9.37 ms per prompt token (from 31).
Task #24 still carried its original framing — "~50× off roofline" — and that
framing was wrong in the denominator, so the first job was to fix the question.

**A byte roofline does not apply to prefill.** At M=512 a chunk re-reads the
weights once and does 512 rows of work against them: prefill is compute-bound
where decode is bandwidth-bound. Dividing prefill time by the 14.43 GB weight
stream overstates the gap by a factor of M.

The right floor, derived rather than looked up:

- 25.62 G params touched per forward → 51.2 GFLOP/row → **26.2 TFLOP** per
  512-row chunk.
- V100 SXM2: 80 SM × 64 fp32 cores × 2 flop × 1.53 GHz = 15.7 TFLOPS fp32,
  **31.3 TFLOPS** for packed f16.
- The extern issues `fma.rn.f16x2` (kernels_linear.py:881), so packed f16 is the
  applicable peak. `mma.sync`'s 125 TFLOPS is unreachable from this path.

So the floor is **838 ms/chunk against a measured 4277 — 5.1×**, not 50×.

## What the profile says

`prof_prefill_budget.py`, ctx=512 (one chunk), 4269 ms GPU inside a 4286 ms wall
tick — 0.4% apart, and 8.34 ms/token against the 8.35 that `bench_prefill.py`
measures end to end. The window is right, which is worth stating because the
first run of this script reported **0 ms** (below).

| class | ms/tick | % |
|---|---:|---:|
| fp4 GEMV | 3675.2 | 86.1% |
| — of which `torch.cat` | 269.4 | 6.3% |
| — of which X f32→f16 cast | 29.8 | 0.7% |
| GDN | 116.0 | 2.7% |
| attention | 108.5 | 2.5% |
| rmsnorm | 26.3 | 0.6% |

98.3% accounted for.

**The stated suspect is refuted.** The script named `gdn_chunk_fused` in advance
— 48 of 64 layers, ~6% SM utilization by its own entry, "a serial scan that does
not scale with T yields exactly this signature". It is **116 ms, 2.7%**, 48
calls. Writing the suspect down before the run is what made this a clean refutation
instead of a hunt.

## The finding

```
linear_fp4_gemv_sm70_m_kernel                  3675.2 ms    4865 calls
nsorSizeStride<unsigned int, 4u>, int, unsigned)  269.4 ms    4560 calls
```

4865 = 16 chunks × 305 launches. And 4560 = 4865 − 305: one per launch *except*
the first chunk of each pass. That is `y2 = torch.cat([y2, y], 0)` in the chunk
loop — it rebuilds every row accumulated so far on each iteration:

    chunk i copies i*32 already-accumulated rows + 32 new
    16 chunks -> 4352 row-copies where 512 would do (8.5x)

**269.4 ms of a 4269 ms tick, 6.3%, and quadratic in the chunk count** — so it
gets worse with a larger prefill budget, which is the direction that budget wants
to move.

## Fix

Preallocate `[M, N]` and write each chunk into its slice. Decode is a single
chunk and keeps the kernel's own buffer, so it allocates nothing new.

One trap: the obvious buffer is `self._zeros2(M, N)`, and that returns a
**shared cached block** other calls read as their residual. Writing into it
would corrupt them. It has to be a real `torch.empty`.

Verified bit-identical to the old `cat` on every shipped chunk shape including
the ragged ones (M = 1, 4, 32, 33, 64, 500, 512), which is the case a slice-write
gets wrong when the last chunk is short. 150 CPU tests pass.

**Decode cannot regress, by construction rather than by measurement.** B=1 decode
is M=1 and a speculative verify is M=B×W≤32, so `len(chunks) == 1`, the branch is
not taken, and `y2 = y` is what the old code did on its first iteration. The
prealloc only exists for the multi-chunk case, which is prefill.

## End to end

| ctx | before | after | gain |
|---:|---:|---:|---:|
| 512 | 8.35 | **7.89** | 5.5% |
| 1024 | 8.52 | **8.03** | 5.8% |
| 2048 | 8.80 | **8.33** | 5.3% |
| 4096 | 9.37 | **8.92** | 4.8% |

Time-to-first-token at 4096: 38383 → **36532 ms**.

Predicted from the profile: 8.34 → 7.81 (removing 269.4 ms of 4269). Measured
8.35 → 7.89. The attribution was 0.08 ms/token optimistic, which is the residual
cost of the slice-write itself — the copy does not vanish, it stops being
repeated.

## The instrument failed first, and silently

The first run printed a clean table of **0 ms / 0 calls**. `one_prefill_tick`
submitted *and* ran the tick, then the profiler wrapped a *second* `e.step()` —
and at ctx == the chunk budget the prompt is one chunk, so step #1 had already
done the prefill and emitted the token. The profiled step found an empty queue.

An empty window rendered as a well-formed table of zeros: no error, no warning,
column headers and all. The script now profiles the first step after submit, and
raises rather than print an empty budget.

## Rule

**A roofline is a property of the inner loop, and M is part of the inner loop.**
Prefill and decode run the same kernels and have different bounds — bytes for
M=1, FLOP for M=512. Reusing decode's denominator for prefill produced a 50×
gap that is really 5×, and a 10× error in the framing is enough to pick the
wrong optimization.

Second: **`cat` in a loop is quadratic, and profilers name it something
unrecognizable.** The cost showed up as
`nsorSizeStride<unsigned int, 4u>` with no hint of `cat` in it. The call count is
what identified it: 4560 against 4865 launches is not a coincidence, it is
`n − n/chunks`. Read call counts as arithmetic, not as labels.

Third, on empty windows: **a measurement of nothing must fail, not format.**
Both profiler bugs found in this subsystem this week (the captured-graph
`synchronize()` and this one) produced confidently formatted output from a wrong
window, and in both cases a reconciliation check against a known end-to-end
number was the tell.

## Results

| date | commit | machine | target | model | ctx | prefill ms/tok |
|---|---|---|---|---|---:|---:|
| 2026-09-02 | (this) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 512 | **7.89** |
| 2026-09-02 | (this) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 4096 | **8.92** |
| 2026-09-02 | fb31be5 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 512 | 8.35 |
| 2026-09-02 | fb31be5 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 4096 | 9.37 |

Decode is unchanged and not re-measured: the multi-chunk branch it would need to
be affected by is one it never takes (dense 37.6 tok/s, spec 50.8 at 1024 stand).

Raw artifacts: `scripts/prof_prefill_budget.py` (the budget table),
`scripts/bench_prefill.py` (ms/token).

## What is left in prefill

The GEMV is still **3406 ms of a 4000 ms tick, 4.1× its 838 ms FLOP floor**, and
that is now the whole story — everything else together is under 7%. The next
question is why the extern gets ~24% of packed-f16 peak at M=32, and that is a
kernel-schedule question (32 rows share one weight stream; the per-row FMA chain
is what has to overlap the loads), not a launch-count one.
