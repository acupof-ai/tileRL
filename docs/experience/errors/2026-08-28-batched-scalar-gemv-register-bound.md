# Batched scalar GEMV (M≤8, X rows in registers) is register-bound — cuda(H20), 2026-08-28

> Status: verdict. Shipped only because it is correct and +2–6% on B=8 vs
> the padded WGMMA path; it is NOT the B=8 lever.

## Context

B=8 in-graph profile after the M=1 work: WGMMA w4a8 / fp8 decode kernels at
3–4× the M=1 GEMV's per-byte cost. Hypothesis: a GEMV that keeps the 8
activation rows of a lane's K-slice in registers and walks R=4 weight rows
reads W once and cuts X traffic /R (the 2026-08-26 small-M GEMV reloaded X
per warp and was L1-bound).

## Root Cause

ncu on `linear_fp4_gemv_mx`: **204 registers/thread → 8 warps/SM, 11.5%
occupancy, DRAM 13–16%**, gate_up 212 µs (ncu clocks) vs 182 for the WGMMA
path. 8 rows × 16 bf16 of X (64 regs) + 32 f32 accumulators + decode
temporaries; GROUP=1 (203 tok/s) and removing the aggregate X copies did not
move the register count. The B=8 tick is 87% these two kernels.

## Fix

None in this form. The instruction and register budgets both say tensor
cores: `mma.sync.m16n8k16.bf16` with the twiddle decode producing the B
fragment pairs directly — and the natural twiddled layout already IS a valid
B fragment if the k order is permuted identically on the A side (lane q's 8
consecutive k become virtual k {2q,2q+1,2q+8,2q+9} of two k16 tiles), so no
re-packing; each lane's 8 elems sit in one 16-block → one scale per lane per
k32 chunk applied on the B fragment. Per lane per k32: 1 LDG.128 (X row) +
per n8 group (LDG.32 + 18 decode + 4 mul + 2 mma) ≈ 3.2 instr/elem for ALL
M ≤ 8 — the M=1 GEMV's cost, at B=8.

## Rule

For M>1 the register file, not L1, is the wall for scalar GEMVs: MX rows ×
K-slice × 2 B per lane is inherent. Above M≈2 use the tensor cores; the
decode already produces bf16x2 pairs, which is the fragment format.
