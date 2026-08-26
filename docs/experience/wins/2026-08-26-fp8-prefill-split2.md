# fp8 prefill GEMM: 2-way K-split, +7.4% geo-mean incl. zeroing — H20, 2026-08-26

> Status: Shipped

## Context

`make_linear_fp4_fp8_mma` (fp4 weight -> e4m3 dequant + fp8 WGMMA, the M>1
prefill path of `linear_fp4`) ships with block_N=64 (the v_n64 schedule,
+33% geo-mean, 2026-08-25). A sweep (`scripts/_sweep_fp8_prefill.py`)
measured a further +8% geo-mean from `v_n64_split2` — the same schedule with
a 2-way K-split grid and f32 atomic add into a zeroed output — but the
sweep's timing excluded the cost of zeroing the [M,N] f32 output buffer the
split requires. This A/B includes it: arm A is the shipped k_split=1 launch;
arm B allocates `torch.zeros` and launches k_split=2, zeroing inside the
timed region.

## What Worked

- **K-split=2 as the sm90 default (`k_split` param on
  `make_linear_fp4_fp8_mma`).** Each (bx, by, bk) block sums K/2 K-tiles and
  f32 atomic-adds its partial into the caller-zeroed Y; the AScale divide
  distributes over the split sum. k_split=1 reproduces the shipped kernel
  unchanged. The backend pads K to a multiple of 2*BK so each split sums an
  exact tile count.

  A/B (H20 pod, GPU 6 idle, JIT-warm, mean of 20 iters, same process —
  `scripts/bench_fp8_split2.py`, commit 4f862f9):

  | shape (M,K,N) | A: shipped ms | B: split2+zero ms | B/A | rel-err vs A |
  |---|---:|---:|---:|---:|
  | 512,5120,17408 (gate/up) | 0.4257 | 0.4371 | 0.974x | 4.44e-03 |
  | 512,17408,5120 (down) | 0.5613 | 0.4753 | 1.181x | 8.59e-03 |
  | 512,5120,10240 (qkv) | 0.2837 | 0.2722 | 1.042x | 4.02e-03 |
  | 512,5120,6144 (z) | 0.1748 | 0.1666 | 1.049x | 4.36e-03 |
  | 512,6144,5120 (out) | 0.2043 | 0.1798 | 1.136x | 4.29e-03 |

  Geo-mean B/A = 1.074x with zeroing included (the sweep's +8% excluded it).
  gate/up (N=17408, 272 N-tiles, 4+ waves already) regresses 0.974x — the
  split's extra blocks add atomic traffic without buying waves; the four
  smaller-N shapes win 1.04-1.18x. Parity: rel-err 4.0e-3..8.6e-3 vs the
  shipped kernel (same fp8 math, split reduction order) — under the 1e-2
  gate, not the 4% fp8 floor vs torch. `uv run pytest -q`: 75 passed,
  4 skipped.

## Rule

For the fp4->fp8 dequant GEMM on sm90, a 2-way K-split with f32 atomic add
into a zeroed output is worth +7.4% geo-mean even when the zeroing is timed:
the split doubles the grid on the small-N shapes that sit at 1-2 waves, and
the zeroing (a memset of the [M,N] f32 buffer) is cheap relative to the
kernel. It is NOT free on already-saturated grids (gate/up, 4+ waves:
0.974x) — the wave count, not the K-loop, is still the first-order lever.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-26 | 4f862f9 | H20 pod | cuda/sm90 | GEMM micro-bench (5 prefill shapes) | — | — | 214.4/162.6/189.2/184.3/157.7 -> 208.8/192.0/197.2/193.3/179.2 TFLOP/s |

TFLOP/s columns are gate/up, down, qkv, z, out (fp8 path, kernel only — the
0.024ms per-token quant is outside this A/B). Raw artifacts:
`scripts/bench_fp8_split2.py` output (pod, GPU 6, this run).

## Iteration

Hypothesis -> verdict in 12.6 min agent wall time (2 pod round-trips, 8
edits) — one of two parallel A/Bs on GPUs 6/7 (workflow wall 14.1 min for
both, 236k subagent tokens). The bench script + kernel port compiled and
passed parity on the first pod run; the second sync was the A/B itself.
