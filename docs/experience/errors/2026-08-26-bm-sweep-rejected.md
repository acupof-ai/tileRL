# block_M sweep under k_split for fp4 prefill GEMM — REJECTED (shipped bM=128 wins geo-mean), 2026-08-26

## Context

The shipped prefill GEMM (`make_linear_fp4_fp8_mma`, sm90) launches with
block_M=128 (backend `_snap_mma_tile(M, 128)`) and k_split=2 (registry). The
bM=128 default was set before k_split=2 landed; the grid is
(N/64) x (M/bM) x k_split, so bM trades M-tiles against per-tile K
throughput, and the best bM could differ between k_split=1 and =2. Swept
bM in {64,128,256} x k_split in {1,2} at the 5 prefill shapes (M=512),
ref = the shipped output. Gate: a non-shipped config wins the geo-mean ->
ship it; per-shape dispatch only if the winner differs by shape and the gain
justifies a backend branch.

## Root Cause

Measured on the H20 pod (GPU 7 idle, JIT-warm per shape, mean of 20 iters
per arm, same process, zeroing inside the timed region for k_split=2 arms,
commit eb6600f). ms per arm, speedup vs shipped in parens:

| config | gate/up 5120x17408 | down 17408x5120 | qkv 5120x10240 | z 5120x6144 | out 6144x5120 | geo-mean |
|---|---:|---:|---:|---:|---:|---:|
| bM=64 ks=1 | 0.5828 (0.745x) | 0.6627 (0.715x) | 0.3623 (0.747x) | 0.2221 (0.748x) | 0.2392 (0.747x) | 0.740x |
| bM=64 ks=2 | 0.5959 (0.729x) | 0.6109 (0.776x) | 0.3584 (0.755x) | 0.2231 (0.745x) | 0.2264 (0.790x) | 0.758x |
| bM=128 ks=1 | 0.4244 (1.023x) | 0.5602 (0.846x) | 0.2825 (0.958x) | 0.1738 (0.956x) | 0.2032 (0.880x) | 0.930x |
| **bM=128 ks=2 (shipped)** | **0.4342** | **0.4739** | **0.2706** | **0.1661** | **0.1788** | **1.000x** |
| bM=256 ks=1 | 0.4661 (0.932x) | 0.7416 (0.639x) | 0.3317 (0.816x) | 0.2347 (0.708x) | 0.2574 (0.695x) | 0.751x |
| bM=256 ks=2 | 0.4503 (0.964x) | 0.5792 (0.818x) | 0.3095 (0.874x) | 0.1914 (0.868x) | 0.2162 (0.827x) | 0.869x |

1. **bM=128 is the M-tile at every shape, under both k_split values.**
   bM=64 is 0.74-0.76x: halving the M-tile doubles the M-grid, and each
   block's WGMMA does half the M-work per K-tile while the dequant +
   shared-memory traffic per K-tile is fixed — the fixed per-K-tile overhead
   is amortized over half the MMA work. bM=256 is 0.75-0.87x: doubling the
   M-tile halves the M-grid, starving occupancy on the 1-2-wave shapes
   (down/qkv/z/out), and the larger WGMMA tile does not amortize the dequant
   enough to pay it back. The bM optimum is robust to the k_split choice —
   the two axes are orthogonal, as the grid factorization suggests.
2. **k_split=2 stays global.** ks=1 wins only gate/up (1.023x — the
   saturated 14-wave shape where the split's atomics are pure cost, the
   known 0.974x regression from the split2 ship). Per-shape ks dispatch
   (ks=1 for gate/up, ks=2 otherwise) buys ~0.5% geo-mean
   (1.023^(1/5) = 1.0046x) for a shape-dependent branch in the backend —
   below the noise floor of a default flip, not justified.
3. **Accuracy: free.** Relerr vs shipped is 4.0e-3..8.6e-3 for the ks=1
   arms (K-reduction order only) and 0.0 for the ks=2 arms (bit-identical:
   the M-tiling does not touch the per-element math, and the two splits of
   a row atomic-add in order). All arms pass the 1e-2 gate.

## Fix

None — the shipped bM=128 / k_split=2 stands. The sweep bench script was
dev-only and is deleted; this entry's table is the tile-space record.

## Rule

For the fp4 prefill GEMM at M=512, block_M=128 is the M-tile under both
k_split values: smaller tiles starve the WGMMA of per-block K work (the
dequant/shared-memory overhead per K-tile is fixed), larger ones starve
occupancy. The bM optimum is orthogonal to the K-split factor — don't
re-sweep bM when changing k_split.

## Iteration

Hypothesis -> verdict in one pod run (green first try): 6 configs x 5
shapes = 30 JIT compiles (M/K/N are T.const, so every config recompiles
per shape), ~4 min pod wall including sync.

## Results

| date | commit | machine | target | arm | geo-mean vs shipped |
|---|---|---|---|---|---:|
| 2026-08-26 | eb6600f | H20 pod GPU 7 | cuda/sm90 | best non-shipped (bM=128 ks=1) | 0.930x (reject) |

## Iteration

Hypothesis -> verdict in 14.8 min agent wall time (2 pod round-trips; the
6-config x 5-shape sweep is 30 JIT compiles, ~4 min pod wall) — one of two
parallel arms of the Phase 2 tile sweep (workflow wall 22.3 min for both,
187k subagent tokens).
