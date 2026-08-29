# The fp8 dX kernel is correct and 0.55x — an 80-block grid, and a parity test that lied

## Context

After chunking cut `linear_fp8_frozen`'s backward peak from 14.2 to 2.054 GiB,
the obvious next step was the kernel the fp4 path already has and fp8 lacks:
dequantize inside the K-loop so no dequantized weight is materialized at all.
Written as `make_linear_fp8_bwd_mma`, and simpler than the fp4 one — an fp8
scale covers a 128-square, and both `_RED_TILE` (32) and `block_N` (64) divide
128, so a whole tile shares one scalar.

It works: the op's peak fell **2.054 -> 0.060 GiB**.

## Root Cause

**Perf: the grid is 80 blocks on a 132-SM card.** The kernel tiles the OUTPUT,
`[M, K]`, so the grid is `ceildiv(K, 64) x ceildiv(M, block_M)`. For lm_head
K=5120, that is 80 blocks in the K direction, and at small M the M direction
contributes 1-2. Meanwhile the contraction is over N = 248320 — **7760 serial
K-loop iterations** inside each of those 80 blocks. The fp4 kernel has the same
shape and does not suffer because no fp4 weight has an N anywhere near that.

| B x T | grid blocks | ratio vs the chunked eager path |
|---|---:|---:|
| 1x64 | 80 | **0.618x** |
| 1x128 | 160 | **0.549x** |
| 1x256 | 320 | 1.030x |
| 2x256 | 640 | 0.956x |

It wins only once the grid fills, and then only by noise.

**Two of my own mistakes, both in the measurement rather than the kernel:**

1. **The parity test failed on a correct kernel.** `allclose(rtol=1e-2,
   atol=1e-2)` elementwise on a dX output is the wrong gate: dX has heavy
   cancellation, so elements land near zero and any absolute error there fails
   a relative test. Measured properly — `||a-b|| / ||b||` — the kernel is
   **0.0034** at every size tested (M=8/64/256, N up to 248320), and the two
   bf16 roundings it inherits (weight cast, gradient cast) account for
   0.0021-0.0024 of that on their own. The kernel has no indexing bug.
2. **The first error probe used `max(|a-b| / (|b| + 1e-6))`** and reported
   errors of 249 and 11001, which I nearly wrote down as a precision verdict.
   Same cancellation, same wrong denominator. A metric that can return 11001
   for a result accurate to 0.3% is not measuring what it claims.

## Fix

Reverted: kernel, registry entry, backend branch, and the parity test. The
memory it saved is no longer the binding constraint — the chunked eager path
already brought this op to 2.054 GiB, and the step's peak at 2x256 is 67.6 GB
either way.

## Rule

Two rules, one per mistake.

**Size the grid against the SM count before writing a tiling kernel.** The
output shape decides the grid; if the contraction is the large dimension, the
grid can be tiny no matter how much work there is. The reopen path here is a
split-N variant — the same trick `linear_fp4_fp8_decode` already uses with
`k_split=8` — which would turn 80 blocks into 640.

**Never gate a cancelling output with an elementwise relative test.** dX, a
gradient, a residual — anything that sums signed terms — needs a norm-relative
bound. Both of my error numbers here were artifacts of the denominator, and one
of them nearly condemned a correct kernel.
