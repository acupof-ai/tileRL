# f16 tile accumulate overflows on this model's activations — cuda(H20), 2026-08-28

## Context

Moved the fp8 kernels to the f16 tensor path (`cvt.rn.f16x2.e4m3x2`, 1 op per
pair instead of 7). The mma8 kernel passed; the M=1 fp8 GEMV tile made
verify check 2 (greedy text) FAIL.

## Root Cause

The M=1 tile accumulated `x·w` for 16 elements in f16x2 and applied the
128-block scale afterwards. Raw e4m3 weights reach 448 (the block scale is
what brings them to O(1)); with post-norm activations of O(10) an unscaled
tile sum passes 65504 → inf. The mma8 kernel was safe only because it scales
the B fragment before the mma.

Scaling the weights first was not enough: check 2 still collapsed to token
0. Qwen3.5's zero-centered norm weights make post-norm activations large,
and 16 scaled products summed in f16 still pass 65504 on some rows. The
mma8 kernel is safe because `mma…f16.f16.f32` accumulates in f32; only the
scalar f16x2 accumulate is exposed.

## Fix

The M=1 fp8 GEMV tile is back on the bf16x2 path (f32-sized exponent, 61%
DRAM); the f16 hardware-cvt path stays in the mma8 kernel where the
accumulator is f32.

## Rule

f16 is a product format, not an accumulator, for this model: use it only
where the hardware accumulates in f32 (mma). Scalar packed FMAs stay bf16.
Parity on sampled tensors did not catch it — only the greedy-text check did;
never skip check 2.
