# e5m2 activation precision for fp4 prefill GEMM — REJECTED (tie speed, 2x error), 2026-08-26

## Context

The precision ladder for the sm90 prefill GEMM: W is fixed at 4-bit (NVFP4
e2m1 packed); the open question is A (activation) precision. Shipped: A
per-token quantized to e4m3, W dequantized e2m1→e4m3 in-kernel, fp8 WGMMA
(`make_linear_fp4_fp8_mma`, k_split=2). This arm: A=e5m2. Hopper wgmma
requires both MMA operands the same fp8 dtype, so the W dequant targets e5m2
too. Gate: ship only if faster than shipped AND relerr vs the bf16 torch
reference (`reference.linear_fp4`) ≤ 1e-2; a speed tie rejects (keep shipped)
— e4m3 already passes at its ~4% floor.

## Root Cause

Two independent failures, measured on the H20 pod (GPU 6 idle, JIT-warm, mean
of 20 iters per arm, same process — `scripts/bench_fp8_e5m2.py`, commit
2e5921e):

| shape (M,K,N) | A: e4m3 ms | B: e5m2 ms | B/A | A relerr vs bf16 | B relerr vs bf16 |
|---|---:|---:|---:|---:|---:|
| 512,5120,17408 (gate/up) | 0.4369 | 0.4367 | 1.001x | 4.03e-02 | 7.81e-02 |
| 512,17408,5120 (down) | 0.4748 | 0.4753 | 0.999x | 3.78e-02 | 8.03e-02 |
| 512,5120,10240 (qkv) | 0.2708 | 0.2706 | 1.001x | 3.77e-02 | 7.57e-02 |
| 512,5120,6144 (z) | 0.1666 | 0.1668 | 0.999x | 3.95e-02 | 7.54e-02 |
| 512,6144,5120 (out) | 0.1785 | 0.1793 | 0.996x | 3.87e-02 | 7.44e-02 |

1. **Speed: geo-mean B/A 0.999x — a tie.** e5m2 and e4m3 are both 1
   byte/elem and lower to the same WGMMA instructions; only the dtype
   differs, so identical timing is the structural result, not noise.
2. **Accuracy: e5m2 relerr 7.44e-2..8.03e-2 vs the bf16 oracle — 7.6x over
   the 1e-2 gate.** Error decomposition: the shipped e4m3 arm (W4
   dequant-to-e4m3 requant + e4m3 A-quant, the accepted floor) is
   3.77e-2..4.03e-2; the e5m2 A-quant contribution (the gap) is
   ~3.6e-2..4.2e-2 — e5m2's 2-mantissa-bit grid (up to 12.5% per-element
   rounding error) roughly doubles the shipped error.

## Fix

None — reverted. The e5m2 variants (`make_quant_fp8_e5m2` + an `fp8_dtype`
param on `make_linear_fp4_fp8_mma`) stay out of the tree; `registry.py` was
never wired. e4m3's 3 mantissa bits are the A-precision floor for this GEMM:
e5m2 buys nothing (same bytes, same instructions) and costs ~2x the error.

## Rule

For fp8 WGMMA operands on sm90, e4m3 vs e5m2 is a precision choice, not a
speed choice — same byte count, same instructions, so the timing ties
structurally and e5m2's 2-mantissa-bit grid ~doubles the GEMM error. Don't
A/B fp8 dtypes for speed; only for accuracy headroom, and e4m3 already sits
at the ~4% floor.
