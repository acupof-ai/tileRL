# Three quantization parity gates modelled a kernel the backend never ran — cuda(H20), 2026-08-29

> Status: fixed. CUDA `test_ops_parity` 36 passed / 4 failed → 39 / 1.

## Context

The pod's suite had been red for a while: 13 CUDA failures, never enumerated
(the log was piped through `tail -5`, so only four names were ever visible).
Three of them were the project's own correct-inference gate —
`test_linear_fp4_parity`, `test_linear_fp4_fp8_parity`, `test_linear_fp8_parity`.
A parity gate that is red is a gate that is off, and TP and TF32 both landed
while it was.

## Root Cause

All three picked their reference with **`M > 1`**. The real dispatch boundary
is **`M > _MX`, and `_MX` is 8** (`backend.py:98`): M=1 is the scalar GEMV,
2–3 the M-row GEMV, 4–8 mma8 — and only above 8 does the quantization itself
change. So each test compared a kernel against a reference for a DIFFERENT
numeric path, and the mismatch showed up as a 2–4% "kernel error".

Swept M across the boundaries (`scripts/probe_linear_err.py`,
`scripts/probe_fp8_quant.py`):

`linear_fp4` — which quantization the result matches:

| M | vs f32-fp4 ref | vs fp8 ref |
|---|---:|---:|
| 1–8 | **3.1–4.5e-03** | 2.8–4.0e-02 |
| 16, 17 | 3.5–3.7e-02 | **8.6e-08** |

`linear_fp8` — which ACTIVATION precision the result matches:

| M | vs w8a8 | vs w8a16 |
|---|---:|---:|
| 1 | 2.53e-02 | **3.66e-03** |
| 4 | 2.24e-02 | **2.96e-04** |
| 8 | 2.46e-02 | **3.19e-04** |
| 16 | **4.04e-03** | 2.57e-02 |
| 32 | **2.65e-03** | 2.63e-02 |

Against the reference it actually computes, every kernel is at its floor —
the bf16 accumulation floor for the same product is 2.13e-03. Nothing was
wrong with any kernel.

A useful side finding: **`linear_fp8` keeps the activation in bf16 through
M=8**, exact to 3e-04. Only M>8 pays the ~2.6% per-token e4m3 activation
quant. The decode range B=1..8 is more accurate than the tests assumed.

## Fix

Pick the reference by the dispatch rule (`M > _MX`), and cover BOTH sides of
the boundary — each test previously exercised one M, and it was the one its
own comment described wrongly.

## Rule

**Before reading any error number, establish which code path the input takes.**
Sweeping M across the dispatch boundaries cost one run and would have been the
right first step; instead the 2–4% was explained three different ways first
(metric artifact → real kernel bug → wrong reference), and the middle one was
stated confidently while resting on a comparison against the wrong reference.

Corollary: an error that lands in the "plausible quantization noise" band is
the most dangerous kind, because it recruits your priors instead of your
attention. Anchor it against a floor you can compute — here, the same product
in bf16 vs f32 — and a 2% that should have been 0.03% stops looking normal.
