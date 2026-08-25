# fp4 GEMV: grouped dequant — hoist shuffles off the FMA critical path — H20, 2026-08-25

> Status: Shipped

## Context

The fp4 decode GEMV (`make_linear_fp4_gemv`) sat at ~42% of HBM roof on the
big projections (direct-call, no backend overhead); the nodecode floor
(w=1.0, same schedule) was ~57%. The bf16 GEMV (same schedule, no dequant)
hits 42-116% — the schedule is sound; the dequant is the gap. Round 1
(2026-08-25-fp4-gemv-vectorized-dequant.md) shipped the warp-shuffle LUT
(1 op/elem) + partial-scale, reaching 44.6% roof, and rejected 2
accumulators / micro=16,32 / shared-X / shared-LUT / 256-LUT / f32-X. The
SGLang source analysis (2026-08-25-sglang-fp4-kernel-comparison.md) pointed at
the remaining lever: dequant OFF the FMA critical path — software-pipeline
dequant(k+1) with FMA(k).

## What Worked

- **Grouped dequant (GROUP=4): load 4 micro-tiles, decode all 4 (32 shuffles),
  then FMA all 4.** The flat kernel issued each shuffle right before its FMA,
  so every FMA stalled on its shuffle's latency. Hoisting all 32 shuffles
  before the 32 FMAs lets the shuffle latency hide behind the FMA dependency
  chain. The grouped buffers are `T.unroll(GROUP)`-indexed (compile-time
  constant -> registers — the PTX shows `float ws[32]` etc., zero STL/LDL
  spills). A runtime `%2` ping-pong double-buffer spills to local memory
  (22% roof, rejected).

- **Sweep (N=17408 K=5120, BW 3312, direct kernel, no backend overhead):**

  | variant | ms | %roof |
  |---|---:|---:|
  | flat (round-1 lutshfl) | 0.0480 | 42.0 |
  | group4 (shipped) | 0.0439 | 46.0 |
  | noxbuf (group4, X reloaded in FMA) | 0.1048 | 19.3 |

  On the transposed shape (N=5120 K=17408): flat 38.5% -> group4 42.7%.
  group4 is +9.5-10.9% over flat on both orientations.

- **The X buffer is essential.** Dropping it (reload X from global during the
  FMA) drops to 19% roof — the extra load instructions and global-memory
  latency dwarf the register-pressure savings. X is 2 KB (fits L1) but the
  reload still costs.

- **2 accumulators don't help (45.0% vs 45.0%).** The compiler already
  reassociates the 8-deep FMA chain; the chain is not the bottleneck once the
  shuffle latency is hidden.

- **Tested and rejected this round:**
  - Register double-buffer (software-pipelined K loop, `ko%2` ping-pong):
    spills to local memory, 22% roof.
  - 6-op bitcast decode (`((n&8)<<28)|((252+(n&7))<<22)`, bit-exact vs the
    9-op): 32% roof — 6 int ops/elem is still more issue than 1 shuffle/elem.
  - 256-entry byte-LUT (int64, 1 load per 2 elems): 26% roof — 64-bit gather
    loads are slower than shuffles; round 1's rejection holds.
  - noxbuf (above).

- **Parity: rtol=1e-2 holds.** 31/31 CUDA parity tests green
  (`tests/test_ops_parity.py`, tiny shapes, vs torch-eager). The grouped
  decode changes only WHEN the dequant issues, not the math — same partial
  sums, same accumulation order, bit-exact vs the flat kernel (rel-err
  2.74e-3, unchanged).

- **Slice decode (graph-captured wall, 10-tick avg, slice4 = 3 GDN + 1 FA):**
  flat GEMV 1.887 ms/tick (530 tok/s) -> group4 1.837 ms/tick (545 tok/s),
  +2.7%. The per-linear roofline through the backend is ~30% on big shapes
  (backend dispatch overhead ~0.022 ms/call dominates the small-shape time);
  the direct-call roofline (46%) is the clean kernel-level signal.

## Rule

For a memory-bound GEMV with a warp-shuffle LUT decode: group the K loop
(GROUP=4 micro-tiles) and hoist ALL shuffles before ANY FMA — the shuffle
latency hides behind the FMA dependency chain, buying ~10% on the kernel.
Use compile-time-constant buffer indices (T.unroll) so the grouped buffers
stay in registers; a runtime ping-pong index spills. Keep the X buffer
(reloading X in the FMA is 2x worse). The shuffle LUT at 1 op/elem remains
the lowest-instruction decode — bitcast (6 ops/elem) and byte-LUT (gather
loads) are both worse.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | 1190885 | H20 | cuda/sm90 | 27B slice (3 GDN + 1 FA) | — | 1.887 (wall) | 530 decode |
| 2026-08-25 | ba1818e | H20 | cuda/sm90 | 27B slice (3 GDN + 1 FA) | — | 1.837 (wall) | 545 decode |

Decode ms/tick is the graph-captured per-tick wall (10-tick avg,
`scripts/bench_fp4_gemv.py` path); the profiler's GPU tracer no longer
captures the GEMV kernel under graph capture, so the wall is the
contention-independent metric.

Raw artifacts: pod `/work/bench_before.log` (flat GEMV),
`/work/bench_after.log` (group4 GEMV), `/work/sweep6.log`
(H20, GPU 1, JIT-free, both shape orientations).
