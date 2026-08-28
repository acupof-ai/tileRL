# Tensor-core decode GEMM for M ≤ 8 (Marlin-style, twiddle-fed) — cuda(H20), 2026-08-28

> Status: Shipped for 2 ≤ M ≤ 8 (fp4 and fp8); M=1 keeps the scalar GEMV

## Context

The B=8 tick was 87% two batched scalar GEMVs at 204 regs/lane (12%
occupancy) — `errors/2026-08-28-batched-scalar-gemv-register-bound.md`.
Register file, not bandwidth, was the wall; the tensor cores take the
activation rows as a fragment and the twiddle decode already emits bf16x2
pairs, which is the B-fragment format.

## What Worked

- **No re-packing.** Under `mma.sync.m16n8k16`, lane (g=l/4, q=l%4)'s 4
  bytes = 8 consecutive k of row 8·grp+g decode to d0..d3; treat them as
  b0/b1 of two k16 tiles with virtual k {2q,2q+1,2q+8,2q+9} standing for
  actual k 8q+{0..3} (+4). The A fragment uses the same map: one LDG.128 of X
  row g at k0+8q gives a0/a2 of both tiles (rows 8–15 zero). A lane's 8
  elements share one 16-block → one scale per lane per k32 chunk, multiplied
  onto the B fragment in bf16 (exact: block scales are e4m3 values).
- Block = 4 warps × 32 output rows; warps split K in k32 chunks, G=4 chunks
  loaded per call (one chunk per call was latency-bound: 2× slower than the
  scalar GEMV at M=1), shared-memory reduction, `Res + y·OScale` epilogue.
- fp8 twin: LDG.64 per lane, e4m3→bf16x2 bit placement, 128-block scale.

Parity at 27B dims: fp4 M=8 **1.7e-3**, fp8 M=8 1.6e-3 (f32 accumulate on the
tensor cores; the w4a8 path it replaces was 3.5e-2 with activation quant).
B=8 aggregate (quiet host): d512 219 → **286.7** (+31%), d2k 216 → 254.7
(+18%), d8k 171 → 202.2 (+18%). M=1 through this kernel: 39.9 tok/s vs 87
on the scalar GEMV (15 of 16 tensor rows idle, same decode cost) — M=1 keeps
the GEMV.

## Rule

For 2 ≤ M ≤ 8 the decode is the cost and the tensor cores are free: feed
`mma.sync` straight from the packed decode, permute k consistently on both
operands instead of re-packing weights, and keep G chunks of loads in
flight per warp. Scalar batched GEMVs cannot fit MX rows of X in registers.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-28 | f0dc914 | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | 0.55 | 27.9 (B=8, d512) | B=8 agg **286.7** d512 / 254.7 d2k / 202.2 d8k; B=1 87.3 unchanged |
