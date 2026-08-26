# Decode tick profile: where the ms go (B=1 / B=8, slice4) — cuda/sm90, 2026-08-26

> Status: Profile

## Context

The prefill path is settled (W=4, A=e4m3, tiles bM=128/bN=64/ks=2, all three
layers verified 2026-08-26). Decode is the remaining bottleneck: the last full
bench measured 49 tok/s vs the 80 tok/s target (12.5 ms/tick), and the 27B fp4
weight read on H20 (~4 TB/s) floors at ~3.4 ms/tick — decode is ~6x its
bandwidth floor, i.e. launch/schedule-bound, not bandwidth-bound. Kernel A/Bs
without knowing where the tick goes would be blind; this entry is the recon:
a per-op breakdown of the decode tick at B=1 and B=8 through the shipped
serving path (decode graph on, fuse_projections on), on the slice4 checkpoint
(3 GDN + 1 full-attn layers — exactly the 27B's 48:16 mix).

Method: `scripts/profile_decode_tick.py` (dev tool, this entry is its bench
record). The shipped decode tick is a captured CUDA graph, and a captured graph
cannot be per-op timed from Python (the wrappers do not run during replay), so
the per-op breakdown is measured on the **eager** path — the exact kernel
sequence the graph replays — with CUDA events around every backend call.
Linears are timed at the `Model._linear` seam, so every GEMV/MMA carries its
projection name and dispatch path (fp4-gemv / fp8-mma / ...); RMSNorms are
named by weight. The graph path is measured separately: total wall, replay,
sampling, host. Steady-state only: 12 warmup ticks flush the B=8 one-per-tick
prefill admissions and the first-tick JIT/capture; numbers are 30-tick avgs.
H20 GPU 6 quiet-gated (0% util outside our run), JIT cache warm.

## Results

### B=1 decode tick (slice4, 30-tick avg)

Eager per-op (CUDA-event spans; include launch gaps, exclude host dispatch).
GPU-event sum 5.852 ms, wall 7.46 ms. The shipped graph tick is 1.791 ms — the
delta (~4.1 ms, 70% of the eager GPU sum) is the Python dispatch tax the graph
removes.

| op | path | calls | ms/tick | % tick |
|---|---|---:|---:|---:|
| rmsnorm (11 norms, 22 kernels) | f32 split-K | 11 | 1.4234 | 24.3% |
| lm_head | fp4-gemv | 1 | 0.5895 | 10.1% |
| gate_up (x4 layers) | fp4-gemv | 4 | 0.5892 | 10.1% |
| down_proj (x4) | fp4-gemv | 4 | 0.4463 | 7.6% |
| qkvz (x3 GDN) | fp8-gemv | 3 | 0.3565 | 6.1% |
| out_proj (x3 GDN) | fp8-gemv | 3 | 0.3095 | 5.3% |
| state_scatter | torch index | 3 | 0.2725 | 4.7% |
| ab (x3 GDN) | fp4-gemv | 3 | 0.2640 | 4.5% |
| rope (q+k) | f32 | 2 | 0.2452 | 4.2% |
| paged_attention | sm90 MMA | 1 | 0.2360 | 4.0% |
| silu_mul (x4) | f32 | 4 | 0.2306 | 3.9% |
| state_gather | torch index | 3 | 0.2190 | 3.7% |
| write_tokens | sm90 | 1 | 0.1702 | 2.9% |
| add (x8 residuals) | torch | 8 | 0.1254 | 2.1% |
| embedding | f32 table | 1 | 0.1198 | 2.0% |
| o_proj (FA layer) | fp4-gemv | 1 | 0.1033 | 1.8% |
| qkv (FA layer) | fp4-gemv | 1 | 0.1029 | 1.8% |
| sample (greedy) | torch argmax | 1 | 0.0487 | 0.8% |

Categories: linears 2.761 ms (47%), rmsnorm 1.423 ms (24%), attn 0.652 ms
(11%), gdn-core 0.492 ms (8%), other 0.476 ms (8%), sampling 0.049 ms (1%).

Two regimes in one table. The **linears are real GPU work**: lm_head reads
954 MB in 0.59 ms (1.6 TB/s, ~49% of the measured 3.3 TB/s roof — matches the
final bench's direct 0.5195 ms), gate_up reads 134 MB in 0.147 ms (~45%
roof). Everything else is **launch-gap-dominated**: rmsnorm's 22 f32 kernels
move 20 KB each yet span 0.12-0.15 ms apiece; state_gather/scatter are torch
advanced indexing (3 MB + an H2D slot copy per call); rope/silu/add/embedding
are single tiny kernels. Their GPU work is microseconds; their span is the
~50-75 us/op Python dispatch gap (4.1 ms / 55 calls). In the shipped graph these are baked (replay
launches ~1-2 us), so the graph tick is 1.79 ms, not 5.85 ms.

Shipped graph tick: **wall 1.7912 ms (558.3 tok/s)** = replay 1.7071 (95.3%) +
sampling 0.0455 (2.5%) + host/copies 0.0386 (2.2%).

### B=8 decode tick (slice4, 30-tick avg)

Eager per-op, GPU-event sum 9.847 ms, wall 11.88 ms. Shipped graph tick 4.822 ms.

| op | path | calls | ms/tick | % tick |
|---|---|---:|---:|---:|
| down_proj (x4) | fp4-mma | 4 | 1.3869 | 14.1% |
| gate_up (x4) | fp4-mma | 4 | 1.3798 | 14.0% |
| rmsnorm (11 norms) | f32 split-K | 11 | 1.2308 | 12.5% |
| lm_head | fp4-mma | 1 | 1.2293 | 12.5% |
| qkvz (x3 GDN) | fp8-mma | 3 | 0.8238 | 8.4% |
| out_proj (x3 GDN) | fp8-mma | 3 | 0.7579 | 7.7% |
| ab (x3 GDN) | fp4-mma | 3 | 0.4931 | 5.0% |
| sample (greedy, x8) | torch argmax | 8 | 0.3938 | 4.0% |
| state_scatter | torch index | 3 | 0.3604 | 3.7% |
| o_proj (FA layer) | fp4-mma | 1 | 0.2693 | 2.7% |
| rope (q+k) | f32 | 2 | 0.2677 | 2.7% |
| state_gather | torch index | 3 | 0.2599 | 2.6% |
| paged_attention | sm90 MMA | 1 | 0.2445 | 2.5% |
| qkv (FA layer) | fp4-mma | 1 | 0.2405 | 2.4% |
| write_tokens | sm90 | 1 | 0.2030 | 2.1% |
| embedding | f32 table | 1 | 0.1418 | 1.4% |
| silu_mul (x4) | f32 | 4 | 0.0873 | 0.9% |
| add (x8 residuals) | torch | 8 | 0.0768 | 0.8% |

Categories: linears 6.581 ms (67%), rmsnorm 1.231 ms (12%), attn 0.715 ms
(7%), gdn-core 0.620 ms (6%), sampling 0.394 ms (4%), other 0.306 ms (3%).

**At B=8 every linear runs the prefill WGMMA kernel, not the GEMV**: M=8 pads
to bM=16 (50% wasted rows), the fp4 path takes `linear_fp4_fp8` with k_split=2
atomics, and every linear pays an e4m3 activation-quant launch. Same weights as
B=1, 1.9-3.1x the cost:

| linear | B=1 GEMV ms | B=8 MMA ms | ratio |
|---|---:|---:|---:|
| lm_head | 0.5895 | 1.2293 | 2.09x |
| gate_up (per call) | 0.1473 | 0.3450 | 2.34x |
| down_proj (per call) | 0.1116 | 0.3467 | 3.11x |
| qkvz (per call) | 0.1189 | 0.2746 | 2.31x |
| out_proj (per call) | 0.1032 | 0.2526 | 2.45x |
| o_proj / qkv (FA) | 0.103 | 0.269 / 0.241 | 2.6x / 2.3x |
| ab (per call, W=0.37 MB) | 0.0880 | 0.1644 | 1.87x |

Shipped graph tick: **wall 4.8223 ms (207.4 tok/s per-request, 1659.0
aggregate)** = replay 4.1811 (86.7%) + sampling 0.3936 (8.2%) + host/copies
0.2476 (5.1%). Sampling is 8 separate argmax-over-248320 calls + 8 D2H syncs
— linear in B, 8.2% of the tick already at B=8.

### Full-27B extrapolation (final-bench method: per-layer x16, lm_head + fixed once)

| B | slice4 graph ms/tick | extrapolated 27B ms/tick | tok/s | target |
|---|---:|---:|---:|---:|
| 1 | 1.7912 | 18.556 | 53.9 per-request | 12.5 (80) |
| 8 | 4.8223 | 49.100 | 20.4 per-request / 162.9 aggregate | — |

B=1 improved from the final bench's 49.0 to 53.9 tok/s (slice4 1.828 -> 1.791
ms): the 2026-08-26 qkvz fusion removed 1 GEMV launch per GDN layer (qkv+z ->
one fused GEMV, 3 fewer launches/tick), plus measurement variance. The HBM floor is 6.16 ms (162 tok/s, 20.4 GB/tick at
the measured 3.3 TB/s) — B=1 sits at 33% of roof, B=8 at 13%.

## A-precision verdict (decode ladder layer 2): settled by physics

All three decode GEMV kernels consume **bf16 A** — no path quantizes A to
e4m3 at M=1:

| kernel | A dtype | site |
|---|---|---|
| `linear_bf16_gemv` | bf16 | `backend.py:238-253` (io=bf16), `kernels_linear.py:424` X: bf16 |
| `linear_fp4_gemv` | bf16 | `backend.py:313-326` (io=bf16), `kernels_linear.py:325` X: bf16 |
| `linear_fp8_gemv` | bf16 | `backend.py:402-403` (explicit bf16 cast), `kernels_linear.py:484` X: bf16 |

The e4m3 activation quant (`quant_fp8`) exists only in the M>1 paths
(`linear_fp4_fp8`, `linear_fp8` MMA) — i.e. B>=2 decode and prefill. At decode
M=1, A is one row of K=5120 bf16 = 10 KB, 4-6 orders of magnitude smaller than
the weight matrix it multiplies, so A's dtype is free: GEMV is W-bandwidth-bound
and a narrower A would add dequant instructions for zero bandwidth gain (the
e4m3-scale round already proved this on W: 29% less traffic, 5-11% slower,
reverted — `2026-08-25-fp4-e4m3-block-scales.md`). Layer 2 needs no work.
(Caveat: at B=8 the MMA paths DO quantize A to e4m3 — 20 quant launches/tick —
but that is intrinsic to the fp8 WGMMA path and folds into hypothesis 2 below,
not a layer-2 issue.)

## Top-3 Phase-2 hypotheses

**1. fp4 GEMV dequant issue throughput — the B=1 lever toward 80 tok/s.**
The fp4 GEMVs are the only ops doing large real GPU work in the B=1 tick:
~1.4 ms of the 1.71 ms graph replay (83%), running at 28-49% of the HBM roof
(gate_up 134 MB/0.147 ms; lm_head 954 MB/0.59 ms; fp4 traffic ~1.83 GB/tick at
~39% roof). The final bench traced this to the dequant being on the FMA
critical path — issue-throughput-bound at ~29 inst/micro-tile; the register
double-buffer was rejected (22% roof, spills). Lever: a spill-free
software-pipelined K-loop (double-buffered dequant in shared memory, or fewer
inst/micro-tile). Target: `kernels_linear.py:278` (`make_linear_fp4_gemv`).
Expected gain: lifting 39% -> 50-60% roof saves ~0.25-0.5 ms of the 1.79 ms
graph tick, i.e. ~4-8 ms on the 27B -> 60-95 tok/s. This is the only hypothesis
that moves the 80 tok/s B=1 target.

**2. A small-M GEMV for B>=2 decode — the B=8 lever.** The GEMV gate is
`M == 1` (`backend.py:244,319,392`), so B=8 decode (M=8) runs the prefill
WGMMA kernels padded to 16 rows at 1.9-3.1x the GEMV's per-byte cost
(lm_head 1.229 vs 0.590 ms, same weights, same process) plus 20 e4m3
A-quant launches. Lever: generalize the three GEMV kernels from M=1 to a small
fixed M (8): stream W once, M-way FMA, no padding, no k_split atomics, no A
quant. Targets: `kernels_linear.py:278` (`make_linear_fp4_gemv`),
`:464` (`make_linear_fp8_gemv`), `:401` (`make_linear_bf16_gemv`), and the
three `M == 1` gates in `backend.py`. Expected gain: B=8 linears 6.58 -> ~3 ms
eager (roughly halved); graph replay 4.18 -> ~2.5-3 ms; extrapolated 27B B=8
aggregate 163 -> ~200-260 tok/s.

**3. Batched greedy sampling — the B-scaled leak.** Sampling is B separate
argmax calls over 248320 logits + B D2H result syncs: 0.049 ms at B=1, 0.394 ms
at B=8 (8.2% of the tick, perfectly linear in B). Greedy (temperature=0) is the
common serving case. Lever: one `argmax` over [B, V] + one sync. Target:
`backend.py:723` (`Backend.sample`) / `reference.py:986` (`sample`). Expected
gain: ~0.3 ms at B=8 (6%), negligible at B=1.

Honorable mentions (launch-bound in eager, already baked by the graph — low
priority for the shipped path): rmsnorm's 2-kernel split (22 launches/tick,
`backend.py:178-191`), state_gather/scatter as torch advanced indexing
(6 calls/tick, `backend.py:652-656` — a fused kernel or folding into
`gdn_decode_fused` would remove them).

## Rule

The decode tick is two regimes: W-read linears (real GPU work, roofline-bound)
and everything else (launch-gap-bound, baked by the graph). Per-op recon must
measure eager with events AND the graph total — eager alone overstates the
small ops 10-100x, the graph alone hides which kernels to fix. At B>=2 the
GEMV path's `M == 1` gate silently routes decode through the prefill WGMMA
kernels at 2-3x the per-byte cost — the batch-size axis is part of the kernel
choice, not just a shape.

## Results table

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-26 | 2f5c50a | H20 pod | cuda/sm90 | Qwen3.6-27B NVFP4 slice4, B=1 graph | — | 1.7912 | 558.3 |
| 2026-08-26 | 2f5c50a | H20 pod | cuda/sm90 | Qwen3.6-27B NVFP4 slice4, B=8 graph | — | 4.8223 | 207.4 / 1659.0 agg |
| 2026-08-26 | 2f5c50a | H20 pod | cuda/sm90 | 27B extrapolated, B=1 | — | 18.556 | 53.9 |
| 2026-08-26 | 2f5c50a | H20 pod | cuda/sm90 | 27B extrapolated, B=8 | — | 49.100 | 20.4 / 162.9 agg |

Raw artifacts: `scripts/profile_decode_tick.py` stdout (BENCH_COMMIT=2f5c50a,
GPU 6 quiet-gated, 30-tick avgs), dev tooling exempt from the bench-entry rule.
