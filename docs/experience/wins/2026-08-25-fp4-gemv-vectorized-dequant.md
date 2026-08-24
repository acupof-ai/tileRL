# fp4 GEMV: vectorized dequant — warp-shuffle LUT + partial-scale, slice decode 1.92 -> 1.29 ms/tick — H20, 2026-08-25

> Status: Shipped

## Context

The fp4 decode GEMV (`make_linear_fp4_gemv`) ran at 24-33% of HBM roof on the
big projections. The bf16 GEMV (same split-K + warp-allreduce schedule, no
dequant) hit 42-116% — the schedule is sound; the dequant is the bottleneck.
The old kernel decoded each nibble serially in the K loop with a 9-op integer
bitcast, and applied the per-32 scale on the 2-mul chain `acc += X * w * s`
(loop-carried through the accumulator). Driver: `scripts/bench_fp4_gemv.py`
(H20, idle GPU, JIT-free after same-shape warmup), same process before/after.

## What Worked

- **Partial-scale: `acc += s * sum(X*w)`.** The per-32 scale is applied once
  per micro-tile to the partial sum, not per-element on the FMA chain. This
  makes the FMA loop 1 FP op/elem (like the bf16 GEMV) instead of 2
  (`X*w*s`). The nodecode floor (w=1.0) jumped from 30% to 57% roof — the
  2-mul chain was the primary cap, not the decode.

- **Warp-shuffle LUT decode (1 op/elem).** The e2m1fn grid is a 16-entry
  power-of-two grid. Lane kr holds LUT[kr&15] (built once per thread via the
  integer bitcast — no exp2); each nibble is 1 `tvm_warp_shuffle`. Beats the
  bitcast (9 int ops/elem, 30% roof) and the shared-memory LUT (load latency,
  42%). The SOTA's lop3 intrin only covers affine int4 grids, not e2m1fn's
  float grid.

- **micro_size_k=8, 1 accumulator.** The sweep showed 2 accumulators don't
  help (the compiler already extracts ILP; the kernel is issue-throughput-bound,
  not FMA-latency-bound). Wider micro-tiles (16/32) are worse (longer FMA
  chain, more shuffles per micro-tile).

- **Sweep (N=17408 K=5120, BW 3312, direct kernel, no backend overhead):**

  | variant | ms | %roof |
  |---|---:|---:|
  | nodecode floor | 0.0356 | 56.7 |
  | bitcast (9 ops/elem) | 0.0674 | 29.9 |
  | lutshfl (1 op/elem, shipped) | 0.0453 | 44.6 |

  The lutshfl reaches 78% of the nodecode floor. Tested and rejected:
  2 accumulators (no help), micro=16/32 (worse), shared-X (sync overhead),
  shared-LUT (load latency), 256-entry byte-LUT (load latency), f32-X
  (2x load traffic).

- **Parity: rtol=1e-2 holds.** 31/31 CUDA parity tests green
  (`tests/test_ops_parity.py`, tiny shapes, vs torch-eager). The partial-scale
  changes the accumulation order (scale on the partial sum vs per-element) but
  stays within 1e-2 (rel-err 2.7e-3).

- **Per-linear GEMV, before (old bitcast kernel) vs after (this), BW 3308:**

  | shape (N,K) | before %roof | after %roof | speedup |
  |---|---:|---:|---:|
  | 5120,17408 | 23.9 | 30.3 | 1.27x |
  | 17408,5120 | 26.5 | 30.4 | 1.15x |
  | 248320,5120 (lm_head) | 32.7 | 54.4 | 1.66x |

  The big projections are capped at ~30% by the backend dispatch overhead
  (~0.022 ms/call, 33% of the small-shape time — the tilelang kernel launch +
  Python). The kernel itself (direct, no overhead) is at 44% roof. The lm_head
  (large N, overhead amortized) hits 54%.

- **Slice decode (graph-captured wall, 10-tick avg):**
  - slice2 (2 GDN): 1.922 -> 1.285 ms/tick (1.50x, 520 -> 778 tok/s)
  - slice4 (3 GDN + 1 FA): 6.941 -> 1.893 ms/tick (3.67x vs WGMMA, 144 -> 528 tok/s)

  The slice2 baseline (1.922 ms) is the old bitcast GEMV's wall time
  (2026-08-25-bf16-gemv-decode.md); the slice4 baseline is the WGMMA path
  (the old GEMV's slice4 wall was not benchmarked).

## Rule

For a memory-bound GEMV with a quant grid: (1) apply the block scale to the
partial sum, not the per-element FMA chain — the 2-mul chain `X*w*s` is the
primary cap, not the decode; (2) decode with a warp-shuffle LUT (1 op/elem) —
the e2m1fn grid is power-of-two, so the LUT is built once per thread via
integer bit-pattern synthesis; (3) the kernel is issue-throughput-bound at
~29 instructions/micro-tile — 2 accumulators and wider micro-tiles don't help.
The remaining gap to the bf16 GEMV's roofline on small shapes is the backend
dispatch overhead, not the kernel.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | eb4a463 | H20 | cuda/sm90 | 27B slice (2 GDN) | — | 1.922 (wall) | 520 decode |
| 2026-08-25 | 1190885 | H20 | cuda/sm90 | 27B slice (2 GDN) | — | 1.285 (wall) | 778 decode |
| 2026-08-25 | 1190885 | H20 | cuda/sm90 | 27B slice (3 GDN + 1 FA) | — | 1.893 (wall) | 528 decode |

Decode ms/tick is the graph-captured per-tick wall (10-tick avg,
`scripts/profile_slice.py` path via `bench_fp4_gemv.py`); the profiler's GPU
tracer no longer captures the GEMV kernel under graph capture, so the wall is
the contention-independent metric.

Raw artifacts: pod `/work/bench_gemv_clean.log` (slice2),
`/work/bench_slice4.log` (slice4), `/work/sweep_clean.log`
(H20, GPU 1, JIT-free).
