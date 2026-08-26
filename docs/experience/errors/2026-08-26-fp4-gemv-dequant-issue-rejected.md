# fp4 GEMV dequant issue throughput — REJECTED (PRMT 0.85x, MMA 0.50x; prmt.b32 compiler bug), 2026-08-26

> Status: Killed — no kernel change shipped. The shipped shuffle-LUT GEMV
> (`make_linear_fp4_gemv`) is unchanged. Dev tools: `scripts/_matrix_gemv.py`
> (matrix A/B), `scripts/_test_prmt_standalone.cu` (prmt.b32 bug repro).

## Context

Hypothesis 1 of the decode Phase-2 recon
(`docs/experience/wins/2026-08-26-decode-tick-profile.md`): the shipped fp4
GEMV's warp-shuffle LUT dequant does 32 shuffles per group (GROUP=4, 8
nibbles/thread), allegedly "doubling the 32-cyc nodecode floor" — the
dequant issue throughput is the B=1 lever toward 80 tok/s. The proposed
treatment: replace the shuffle LUT with `__byte_perm` (prmt.b32) hardware
byte-permute (agent-infer's winning scalar technique, 9.3→31.5 tok/s), or
move to MMA (tensor-core) dequant (agent-infer's Marlin, 31.5→84.5 tok/s).

Gate: direct-kernel A/B on lm_head shape (N=248320, K=5120), same process,
same inputs, relerr vs torch-eager reference. Win = candidate faster than
shipped AND relerr <= 1e-2. H20 GPU 6, quiet-gated, JIT cache warm.

## Root Cause

**No dequant reorganization beats the shipped shuffle-LUT GEMV.** Matrix
A/B (scripts/_matrix_gemv.py, 5 cells, lm_head N=248320 K=5120):

| cell | ms | x-vs-shipped | relerr | %roof | verdict |
|---|---:|---:|---:|---:|---|
| gemv_shuffle (shipped) | 0.5215 | 1.000x | 2.38e-03 | 55.4 | OK |
| gemv_prmt (extern scalar) | 0.6128 | **0.851x** | 1.49e+00 | 47.2 | slower + wrong |
| mma_bitcast_m16 (shipped MMA) | 1.0353 | 0.504x | 5.02e-06 | 27.9 | 2x slower |
| mma_prmt_ext_m16 (PRMT+MMA) | — | — | — | FAIL (tilelang layout bug) |
| mma_prmt_ext_m64 (PRMT+MMA m64) | 1.2554 | 0.415x | 1.49e+00 | 23.0 | 2.4x slower |

Three independent reasons the hypothesis fails:

1. **PRMT is 15% slower than the shuffle LUT.** The e2m1fn grid needs 12
   `__byte_perm` per 8 elements (4 LUT selects + 4 bf16 assembles + 4
   interleaves) vs the shuffle LUT's 8 shuffles per 8 elements. The shuffle
   LUT is already at 1 op/elem — the theoretical minimum for a lookup-based
   dequant. The SHFL pipe is separate from the FMA/load pipes, so the
   shuffles overlap with the FMA dependency chain; the kernel sits at 55.4%
   roof, within 3% of the 57% nodecode floor. The shuffles add ~3% overhead,
   not the 100% the hypothesis assumed.

2. **MMA is 2x slower at M=1.** The shipped prefill MMA kernel
   (`make_linear_fp4_mma`) at block_M=16 pads M=1→16 (15/16 rows wasted),
   putting it at 27.9% roof vs the GEMV's 55.4%. Agent-infer's Marlin beat
   their scalar because their scalar was at 21% roof; tileRL's shipped GEMV
   is already at 55%, so the 16x M-waste dominates. block_M=64 (the only
   PRMT+MMA variant that compiles) is 64x M-waste, 2.4x slower.

3. **prmt.b32 is broken on this CUDA version.** Standalone CUDA test
   (scripts/_test_prmt_standalone.cu, no tilelang): `__byte_perm(0x04030201,
   0x08070605, 0x03020100)` returns 0x01020101 instead of 0x04030201. SASS
   disassembly shows the compiler (CUDA 12.9, sm_90) truncates the 32-bit
   PTX selector to 16 bits when emitting the `PRMT` immediate, losing the
   upper 2 selector bytes. Selectors with zero upper 16 bits (0x00000000,
   0x01010101, 0x03030303) work correctly; selectors with non-zero upper
   bits (0x03020100, 0x07060504) silently produce wrong results. This
   affects both `__byte_perm` and inline `prmt.b32` PTX, in both tilelang
   externs and standalone nvcc compilation. H20, driver 535.161.08, CUDA
   12.9.86.

A tilelang codegen bug also blocks the PRMT+MMA direction at block_M≤32:
`T.call_extern` + `T.gemm` triggers `Layout((64,1)->(4,100)) "Could not
normalize iterators"` — a tilelang lowering bug, not a kernel structure
issue. block_M=64 compiles but is 2.4x slower.

## Fix

None for the dequant direction — the shipped shuffle-LUT GEMV is the local
optimum. The 55%→57% gap to the nodecode floor needs fewer dequant
instructions per element (a narrower grid or a hardware decode path), not a
different dequant mechanism or buffer. The prmt.b32 compiler bug is
orthogonal to tileRL and should be reported to NVIDIA / worked around by
using register-based selectors (upper 16 bits = 0) if prmt.b32 is ever
needed on this CUDA version.

## Rule

For a memory-bound GEMV with a warp-shuffle LUT decode: the shuffle LUT at
1 op/elem is the floor. PRMT (1.5 ops/elem on the ALU pipe) is slower, and
MMA at M=1 has 16x M-waste. The SHFL pipe overlaps with FMA/load — the
shuffles add ~3% overhead (55.4% vs 57% nodecode floor), not 100%. And:
prmt.b32 on CUDA 12.9 / sm_90 silently truncates the 32-bit selector to 16
bits — never use a selector with non-zero upper 16 bits on this toolchain.

## Results

| date | machine | target | variant | ms (lm_head) | %roof |
|---|---|---|---|---:|---:|
| 2026-08-26 | H20 | cuda/sm90 | gemv_shuffle (shipped) | 0.5215 | 55.4 |
| 2026-08-26 | H20 | cuda/sm90 | gemv_prmt (extern scalar) | 0.6128 | 47.2 |
| 2026-08-26 | H20 | cuda/sm90 | mma_bitcast_m16 | 1.0353 | 27.9 |
| 2026-08-26 | H20 | cuda/sm90 | mma_prmt_ext_m64 | 1.2554 | 23.0 |

Raw artifacts: pod `/work/results.jsonl` (matrix A/B, 5 cells, JIT cache
warm), `scripts/_test_prmt_standalone.cu` (prmt.b32 bug repro, standalone
nvcc). Dev tooling exempt from the bench-entry rule.

## Iteration

Wall time ~3 h, 6 pod round-trips: (1) first matrix run — 5 cells, PRMT
scalar 0.85x with relerr=1.49 (wrong output), MMA 0.50x; (2) extern
correctness debug — constant-write extern works, input deref works,
`__byte_perm` returns wrong value; (3) inline PTX `prmt.b32` — same wrong
value, ruled out missing header; (4) standalone CUDA test (no tilelang) —
same wrong value, ruled out tilelang; (5) SASS disassembly — compiler
truncates 32-bit selector to 16 bits; (6) this entry. The PRMT scalar
timing (0.851x) is correctness-independent (same instruction count
regardless of values) and is the decisive signal: the shuffle LUT is
already at the 1-op/elem floor.
