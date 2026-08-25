# Final bench after GEMV grouped-dequant + GDN-WY: decode 49 / prefill 1172 tok/s, 80/3800 not met — gap is dequant issue throughput + GDN chunk, not physics — H20, 2026-08-25

> Status: Shipped

## Context

Final 80/3800 measurement after the fp4 GEMV rounds: vectorized dequant
(1190885 — warp-shuffle LUT + partial-scale), grouped dequant (6b39e50 —
hoist shuffles off the FMA path), and the GDN-WY prefill attempt (306a8bf —
rejected, 2.6x slower than serial scan). A parallel e4m3 block-scale round
(a5ccc82, bit-trick fix 6548450) also landed: it is a 5-11% decode regression
(see `2026-08-25-fp4-e4m3-block-scales.md`), reverted in ea8ba7f after a
revert/reapply/re-revert cycle. The numbers below are the f32-scale state
(HEAD ea8ba7f, code-identical to 306a8bf). Driver: `scripts/profile_slice.py`
on slice2/slice4 (graph decode + prefill-512) + `scripts/bench_fp4_gemv.py`,
H20 fully-idle
window (all 8 GPUs at 0%, BW 3312 GB/s measured), 30-tick decode avgs. The full
checkpoint is not on the pod; full-model numbers are extrapolations from
slice4, whose 3 GDN + 1 full-attn mix is exactly the 27B's 48:16 pattern.

## What Worked

**Slice measurements (HEAD f853d0d, idle, graph-captured decode, 30-tick avgs):**

| slice | decode ms/tick | decode tok/s | prefill-512 ms/tok | prefill tok/s |
|---|---:|---:|---:|---:|
| 2 GDN | 1.586 | 630.5 | 0.0725 | 13789 |
| 3 GDN + 1 FA | 1.828 | 547.1 | 0.0557 | 17960 |

Prefill per-op (slice4): linear_fp4 56.6% (NVFP4 MLP, dequant-in-loop),
linear_attn_chunk 27.4% (GDN serial scan), linear_fp8 10.6%, rmsnorm 2.8%,
paged_attention 0.8%.

**Extrapolation (lm_head and fixed cost counted once).** slice4's 3:1 mix is
exactly the 27B's 48:16, so the full model is 16x the slice4 per-layer cost,
with lm_head and the fixed cost (H2D copies + sampling, 0.07 ms) counted once:

    full = 16 x (slice4 - lm_head - fixed) + lm_head + fixed
         = 16 x (1.828 - 0.5195 - 0.07) + 0.5195 + 0.07
         = 20.41 ms/tok = 49.0 tok/s

lm_head is the fp4 GEMV at 0.5195 ms (55.4% of the 3312 GB/s roof, measured
idle via `bench_fp4_gemv.py`). The 2026-08-24 all-levers entry's 59.3 ms /
16.9 tok/s overcounted lm_head 32x (GDN-only per-layer x 64); the 2026-08-25
final bench's 28.2 ms / 35.5 tok/s was the pre-vectorized state. The kernel
delta since the final bench: vectorized + grouped dequant took slice4 from
2.662 to 1.828 ms (1.46x) and the extrapolated decode from 35.5 to 49.0 tok/s.

**e4m3 block scales: tried, reverted (5-11% decode regression).** The e4m3
round (a5ccc82) changed the per-32 block scale from f32 to e4m3 uint8 (4x less
scale traffic, ~29% less fp4 weight traffic), decoded in-register. Measured in
the same idle window: lm_head 0.5195 -> 0.5765 ms (55.4% -> 35.4% roof),
slice4 1.828 -> 1.937 ms (+6.1%), slice2 flat (1.586 -> 1.596). The bit-trick
fix (6548450, integer bit synthesis replacing exp2) recovered the big-GEMV
efficiency to 23-24% but left lm_head at 35.4% — the decode instruction overhead
on the issue-throughput-bound GEMV (~29 inst/micro-tile) outweighs the 29%
traffic reduction. The bf16 GEMV (no dequant) hits 42-116% on the same
schedule, so the dequant is the cap, not the schedule. Reverted in ea8ba7f
(after a revert/reapply/re-revert cycle — the reapply's "neutral" claim rested
on an anomalous 1.841 ms slice4 measurement, inconsistent with the per-linear
regression); the e4m3 work stays in git history if a memory-bound config
appears.

**Roofline analysis (contention-independent):**

- **Decode** reads ~20.4 GB/tick (48 GDN x 320 MB + 16 FA x 279 MB +
  lm_head 954 MB; fp4 at 0.75 B/elem, fp8 GDN projections at 1.03). Measured
  achievable HBM BW is 3312 GB/s at idle (copy benchmark; H20 spec 4.0 TB/s).
  Roof: 6.16 ms = 162 tok/s (measured BW) / 5.10 ms = 196 tok/s (spec). The 80
  tok/s target is 41-49% of roof — physics allows it. We are at 49 tok/s =
  25-30% of roof. The fp4 GEMV runs at 31% roof on the per-layer shapes
  (through backend) and 55% on lm_head (large N, overhead amortized); the
  direct kernel is at 46% (grouped dequant).
- **Prefill** is 25.7 TFLOP (2 x 25.1B params x 512 tokens). Every linear runs
  on fp8-class hardware (296 TFLOPS on H20), so the roof is 86.8 ms =
  5898 tok/s. The 3800 target is 64% of roof — physics allows it. We are at
  1172 tok/s = 20% of roof. The fp4 MLP path is at 39-61% of peak (the
  vectorized-dequant entry), the GDN chunk op is 27.4% of the tick.

**Verdict: 80/3800 not met, and the remaining gap is code, not physics.** The
H20 roofline allows ~162-196 decode / ~5900 prefill tok/s. Decode needs the
fp4 GEMV dequant off the FMA critical path: the grouped dequant hoisted the
shuffles (42% -> 46% direct), but the kernel is still issue-throughput-bound at
~29 inst/micro-tile; a spill-free software-pipelined K-loop (double-buffered
dequant) is the lever — the register double-buffer was rejected (22% roof,
spills to local memory). Prefill needs the GDN chunk op (27.4% of the tick)
faster: the WY attempt was rejected (2.6x slower — the serial kernel is
compute-bound, state fits L2), so the lever is the serial scan's peak
efficiency, not a different algorithm.

## Rule

For a memory-bound GEMV with a quant grid on an issue-throughput-bound kernel:
the dequant instruction count is the cap, not the weight traffic — e4m3 scales
cut traffic 29% but lose 5-11% because the in-register decode adds more
instructions than the traffic saves. Keep the dequant at 1 op/elem (warp-shuffle
LUT) and the scale as a direct f32 load; spend the instruction budget on
hoisting dequant off the FMA path (grouped dequant), not on a narrower scale
dtype. The 80/3800 targets are 41-64% of the H20 roofline — physics allows
both; the gap is dequant issue throughput (decode) and the GDN serial scan's
peak efficiency (prefill).

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | c97f79c | H20 idle | cuda/sm90 | 27B extrapolated (48 GDN + 16 FA) | 1.008 (512-tok) | 28.2 | 35.5 decode / 992 prefill |
| 2026-08-25 | 1190885 | H20 | cuda/sm90 | 27B slice (3 GDN + 1 FA) | — | 1.893 (wall) | 528 decode |
| 2026-08-25 | 6b39e50 | H20 | cuda/sm90 | 27B slice (3 GDN + 1 FA) | — | 1.837 (wall) | 545 decode |
| 2026-08-25 | a5ccc82 | H20 idle | cuda/sm90 | 27B slice (3 GDN + 1 FA, e4m3) | 0.0563 (512-tok) | 1.931 (wall) | 518 decode |
| 2026-08-25 | 6548450 | H20 idle | cuda/sm90 | 27B slice (3 GDN + 1 FA, e4m3 bit-trick) | 0.0565 (512-tok) | 1.945 (wall) | 514 decode |
| 2026-08-25 | ea8ba7f | H20 idle | cuda/sm90 | 27B slice (2 GDN) | 0.0725 (512-tok) | 1.586 (wall) | 630 decode / 13789 prefill |
| 2026-08-25 | ea8ba7f | H20 idle | cuda/sm90 | 27B slice (3 GDN + 1 FA) | 0.0557 (512-tok) | 1.828 (wall) | 547 decode / 17960 prefill |
| 2026-08-25 | ea8ba7f | H20 idle | cuda/sm90 | 27B extrapolated (48 GDN + 16 FA) | 0.853 (512-tok) | 20.41 | 49.0 decode / 1172 prefill |

Decode ms/tick is the graph-captured per-tick wall (30-tick avg); prefill is
the 512-token wall. Extrapolation: slice4's 3:1 mix x 16, with lm_head
(0.5195 ms, measured) and fixed cost (0.07 ms) counted once. The e4m3 rows
(a5ccc82, 6548450) are the reverted regression — the f32 state (ea8ba7f) is the
shipped one. Rooflines: decode 162 tok/s at the measured 3312 GB/s (196 at the
4.0 TB/s spec; 20.4 GB/tick), prefill 5898 tok/s (25.7 TFLOP at 296 TFLOPS
fp8-class). 31/31 CUDA parity, 30/30 CPU (rtol=1e-2 vs torch-eager).

Raw artifacts: pod `/work/idle_f32_slice2.log`, `/work/idle_f32_slice4.log`,
`/work/idle_f32_gemv.log` (f32 final state, H20 idle, BW 3312);
`/work/idle_fix_slice2.log`, `/work/idle_fix_slice4.log`, `/work/idle_fix_gemv.log`
(e4m3 bit-trick, reverted); `/work/idle_e4m3_*.log` (e4m3 exp2, reverted).
