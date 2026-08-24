# Final bench after bf16 GEMV + native FP8: 80/3800 not met — decode 35.5 / prefill 992 tok/s, gap is fp4 dequant efficiency, not H20 roofline — H20, 2026-08-25

> Status: Shipped

## Context

Final measurement of the 27B NVFP4 kernel round after the last two levers
landed: the bf16 GEMV decode kernel (eb4a463 — the 2026-08-25
bf16-gemv-decode entry showed it does NOT touch the fp4 27B decode path,
which is all-fp4) and native FP8 weight retention for the GDN projections
(1a3e5c3..c59da8b — GDN prefill 1.48x in isolation). Driver:
`scripts/profile_slice.py` on both slices, graph-captured decode (the
production path) plus prefill-512, in a fully-idle window (all 8 GPUs at 0%
— the usual 99%-util co-tenant was away), 30-tick decode averages. The full
NVFP4 checkpoint is not on the pod (confirmed: only
`/host/tc27-nvfp4-slice{2,4}`; `/host/Qwen3.6-27B-FP8` is a different VLM
checkpoint), so full-model numbers are extrapolations from slice4, whose
3 GDN + 1 full-attn mix is exactly the 27B's 48:16 pattern.

## What Worked

**Slice measurements (HEAD c97f79c, idle GPU, graph-captured decode, 30-tick avgs):**

| slice | decode ms/tick | decode tok/s | prefill-512 wall | prefill tok/s |
|---|---:|---:|---:|---:|
| 2 GDN | 1.837 | 544.4 | 0.0433 ms/tok | 23091 |
| 3 GDN + 1 FA | 2.662 | 375.7 | 0.0654 ms/tok | 15293 |

Prefill per-op (slice4): linear_fp4 62.3% (the NVFP4 MLP path, fp4->e4m3
K-loop dequant), linear_attn_chunk 24.1%, linear_fp8 9.0% (the native GDN
projections — the fp8 path is live), paged_attention 0.7%.

**Extrapolation method (corrected).** The slice wall includes lm_head once;
scaling a per-layer average by 16 counts it 16 times. lm_head is the fp4
GEMV at 0.890 ms (measured idle, 32.4% of the 3310 GB/s measured roof —
`scripts/bench_fp4_gemv.py`): it is 33% of the slice4 wall but only 3% of
the full-model tick. Splitting the two slice walls with lm_head = 0.890 ms
and fixed cost (pinned H2D copies + sampling) = 0.07 ms: GDN layer 0.438 ms,
FA layer 0.387 ms. Full model = 48x0.438 + 16x0.387 + 0.890 + 0.07 =
28.2 ms.

**Full-model extrapolation vs targets:**

| metric | extrapolated | target | gap |
|---|---:|---:|---:|
| decode | 28.2 ms/tok (35.5 tok/s) | 12.5 ms (80 tok/s) | 2.25x |
| prefill-512 | 1.008 ms/tok (992 tok/s) | 0.263 ms (3800 tok/s) | 3.8x |

The all-levers entry (2026-08-24) published 59.3 ms / 16.9 tok/s decode by
scaling slice2's GDN-only per-layer cost x 64 — that overcounts lm_head
32x and treats the 16 FA layers as GDN. Applied to its own slice4 number
(2.798 ms), the corrected method gives 30.4 ms / 32.9 tok/s. The kernel
delta since then is ~8% (the fp8 GEMV decode: 2.799 -> 2.662 ms on slice4);
the rest of the headline improvement is the method correction, not a kernel
win.

**Eager per-op overcounts the M=1 tick.** The eager profile (slice4,
40-tick avg) sums to 9.499 ms GPU vs the 2.662 ms graph wall — 3.6x. Each
per-op CUDA-event span absorbs the CPU launch latency of the op's kernels
(63 ops/tick, ~100 us/op of GPU-idle-in-span), and M=1 kernels are
launch-latency-bound. The graph wall is the only honest decode metric; the
eager table is useful only for the op mix.

**Roofline analysis (contention-independent):**

- **Decode** reads ~20.4 GB/tick (48 GDN x 320 MB + 16 FA x 256 MB +
  lm_head 954 MB; fp4 at 0.75 B/elem, fp8 GDN projections at 1.03, bf16
  a/b at 2). Measured achievable HBM BW is 3310 GB/s at idle (copy
  benchmark; the H20 spec is 4.0 TB/s). Roof: 6.2 ms = 162 tok/s (measured
  BW) / 5.1 ms = 196 tok/s (spec). The 80 tok/s target is 41-49% of roof —
  physics allows it. We are at 35.5 tok/s = 18-22% of roof. The fp4 GEMV
  runs at 24-32% of roof in isolation (the bf16 GEMV on the same schedule
  hits 42-116% — the fp4 nibble-decode + per-tile scale is the cap, not the
  GEMV schedule), and lm_head alone is 0.890 ms at 32%.
- **Prefill** is 25.7 TFLOP (2 x 25.1B params x 512 tokens). Every linear
  runs on fp8-class hardware (296 TFLOPS on H20; the NVFP4 MLP dequantizes
  to e4m3 in the K-loop), so the roof is 86.8 ms = 5898 tok/s. The 3800
  target is 64% of roof — physics allows it. We are at 992 tok/s = 17%.
  The fp4 MLP path is at 21% of peak (62.6 TFLOPS, 62% of the tick), the
  GDN chunk op is 24% of the tick, the native fp8 path at 31%
  (92.8 TFLOPS).

**Verdict: 80/3800 not met, and the remaining gap is code, not physics.**
The H20 roofline allows ~162-196 decode / ~5900 prefill tok/s. Decode needs
the fp4 GEMV dequant stage fixed (the schedule itself is BW-saturating —
the bf16 GEMV proves it): every projection at 50%+ of roof puts the tick at
~12-15 ms (65-80 tok/s). Prefill needs the fp4 dequant off the K-loop (the
native-fp8 pattern gave 1.48x on the GDN; the MLP ships NVFP4, so its lever
is a better dequant schedule, not weight retention) plus a look at the GDN
chunk op. Neither target is blocked by hardware.

## Rule

Two measurement rules. (1) A slice wall includes lm_head once — scaling a
per-layer average by the layer count overcounts it N times, and at 0.890 ms
(32% of roof) it is the single biggest decode op; split lm_head out before
extrapolating. (2) Eager per-op CUDA-event spans overcount an M=1 tick
~3.6x (each span absorbs launch latency); the graph-captured wall is the
decode metric, and bytes/FLOPs roofline is the contention-independent
verdict.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | 3b1327f | H20 | cuda/sm90 | 27B slice (2 GDN) | — | 48.85 | 20.5 decode |
| 2026-08-24 | 53c1398 | H20 | cuda/sm90 | 27B slice (2 GDN) | 11.26 (512-tok) | 31.09 | 32.2 decode / 89 prefill |
| 2026-08-24 | 76826b0 | H20 idle | cuda/sm90 | 27B slice (2 GDN) | 0.2226 (512-tok) | 5.46 | 183 decode / 4491 prefill |
| 2026-08-24 | 6dfa7d8 | H20 low-ct | cuda/sm90 | 27B slice (2 GDN) | 0.0443 (512-tok) | 1.922 | 520 decode / 22589 prefill |
| 2026-08-24 | 6dfa7d8 | H20 low-ct | cuda/sm90 | 27B slice (3 GDN + 1 FA) | 0.0661 (512-tok) | 2.798 | 357 decode / 15132 prefill |
| 2026-08-25 | c59da8b | H20 | cuda/sm90 | 27B slice (3 GDN + 1 FA) | — | 2.672 | 374 decode |
| 2026-08-25 | c97f79c | H20 idle | cuda/sm90 | 27B slice (2 GDN) | 0.0433 (512-tok) | 1.837 | 544 decode / 23091 prefill |
| 2026-08-25 | c97f79c | H20 idle | cuda/sm90 | 27B slice (3 GDN + 1 FA) | 0.0654 (512-tok) | 2.662 | 376 decode / 15293 prefill |
| 2026-08-25 | c97f79c | H20 idle | cuda/sm90 | 27B extrapolated (48 GDN + 16 FA) | 1.008 (512-tok) | 28.2 | 35.5 decode / 992 prefill |

Decode ms/tok is the graph-captured per-tick wall (30-tick avg); prefill is
the 512-token wall. Extrapolation: slice4's 3:1 mix x 16, with lm_head
(0.890 ms, measured) and fixed cost (0.07 ms) counted once — GDN layer
0.438 ms, FA layer 0.387 ms. Rooflines: decode 162 tok/s at the measured
3310 GB/s (196 at the 4.0 TB/s spec; 20.4 GB/tick), prefill 5898 tok/s
(25.7 TFLOP at 296 TFLOPS fp8-class). The 2026-08-24 all-levers entry's
59.3 ms / 16.9 tok/s used GDN-only per-layer x 64 (lm_head overcounted
32x); the corrected method on its own data gives 30.4 ms / 32.9 tok/s.

Raw artifacts: pod `/work/final_slice2_graph.log`,
`/work/final_slice4_graph.log`, `/work/final_slice4_eager.log`,
`/work/final_bench_fp4_gemv.log` (H20, GPUs 2-3, idle, JIT-free).
