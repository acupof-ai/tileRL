# Dual-format fp8 weights for prefill — REJECTED (fp8 MMA already wired), 2026-08-25

## Context

Prefill decomposition on slice4: `linear_fp4` 56.3% of the tick. The proposed
road to 3800 tok/s was dual-format weights — pack an e4m3 copy of every fp4
projection at load, route prefill through native fp8 tensor cores (H20 has
them; fp4 tensor cores are Blackwell-only), keep fp4 for decode (fp8 GEMV
reads 2x the bytes and decode is bandwidth-bound).

## Root Cause

The plan was already implemented. `backend.py:431` routes `linear_fp4`
through `make_linear_fp4_fp8_mma` on CUDA whenever M > 1 — prefill already
computes on fp8 tensor cores, quantizing x on the fly (0.024ms, negligible).
The "fp4 dequant GEMM" the plan assumed is the M=1 decode path only.

Measured (bench_fp8_prefill.py, M=512, H20):

| shape (M,K,N) | bf16 fallback TFLOP/s | fp8 MMA TFLOP/s | speedup |
|---|---:|---:|---:|
| 512,5120,17408 (gate/up) | 120.7 | 175.8 | 1.46x |
| 512,17408,5120 (down) | 110.7 | 113.8 | 1.03x |
| 512,5120,10240 (qkv) | 117.7 | 143.4 | 1.22x |
| 512,5120,6144 (z) | 117.9 | 127.3 | 1.08x |
| 512,6144,5120 (out) | 102.3 | 103.8 | 1.01x |

## Fix

None — no code changed. The prefill gap to 3800 tok/s is kernel efficiency,
not weight format:

- the fp8 MMA sits at **59% of fp8 peak** (176/296 TFLOP/s) on the best
  shape and goes neutral at large K (down/out: ~1.03x) — the kernel leaves
  1.5-2x on the table there;
- the GDN chunk is 27.6% of the tick at ~50% of peak (separate sweep).

## Rule

Before proposing a format/casting change, grep the backend dispatch for the
kernel that already handles that shape regime — `linear_fp4` on CUDA M>1 is
already fp8 MMA; the remaining prefill lever is fp8 GEMM kernel efficiency
(59% -> 90% peak, fix the K-large stall), not dual weights.
