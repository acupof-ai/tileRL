# Halving the mma8 block-scale bytes bought nothing — reverted, 2026-08-28

## Context

The decode GEMM's per-lane scale loads were the last named lever in the decode
record: each lane reads a 4-byte f32 block scale per (chunk, group), and four
lanes of a group read the SAME address, so L1 sees roughly 2.5x the weight
bytes. Block scales are e4m3 values, so bf16 holds them exactly (f16 for the
fp8 arm, whose mma already rounds the scale through `cvt.rn.f16x2.f32`): half
the bytes, and the scalar-to-pair widening collapses from a `cvt` to a bit
replication `(s << 16) | s`.

## Root Cause

Nothing was wrong with the change — it is correct (mma8 parity fp4 2.8e-3 /
fp8 1.7e-3 at M=2/4/8). It simply does not move the metric. Matched A/B on the
27B, H20 GPU 7, B=8 (B=1 never enters mma8, so those rows isolate it cleanly):

| row | baseline | with bf16/f16 scales |
|---|---:|---:|
| decode-kv/d512-b8 | 308.6 | 308.0 (0.998x) |
| decode-kv/d8192-b8 | 280.2 | 277.7 (0.991x) |

Batched decode is not bound on the L1 traffic this saves.

## Fix

Reverted the scale dtype. Kept: the 100 lines of dead `tl_fp4_mma_k32` /
`tl_fp8_mma_k32` externs the change happened to expose, and
`scripts/parity_mma8.py`.

Cost of the detour: the scale's storage dtype has to agree with the asm
constraint of the `cvt` that widens it, and the two arms disagree — the fp4 mma
multiplies in bf16x2, the fp8 mma in f16x2. A bf16 tensor on the fp8 arm gave
`asm operand type size(2) does not match type/size implied by constraint 'f'`,
and tilelang's bf16 pointer needed `const void*` before it would convert. Two
debug cycles for zero throughput.

## Rule

The 2.5x-L1-traffic argument was an arithmetic estimate, not a measurement of a
bound. Before trading a dtype invariant for bandwidth, check that the kernel is
actually bound on that bandwidth — B=8 decode is not. And when a correct change
measures flat, revert it: the invariant it adds outlives the gain it did not
deliver.
