# Occupancy is not the sm70 GEMV's cap either — V100, 2026-09-03

> Status: **rejected at 1.00×**, threshold was ≥1.15×. `min_blocks=4` quadruples
> resident warps and M=32 does not move. The SASS numbers that motivated it (255
> registers, 1 block/SM, 2.29 inst/HFMA2) are facts and are recorded below; the
> occupancy *diagnosis* built on them is refuted by its own A/B. I had written that
> diagnosis up as a win entry while the measurement was still running — that file is
> deleted rather than left in the tree contradicting this one.

## Context

SASS gave a real number: `Used 255 registers`, so 255 × 128 threads = one block per
SM = 4 of Volta's 64 warps, 6.25% occupancy. Four ceilings were already excluded
(FMA peak, L1 capacity, L1 bandwidth, instruction issue at 2.29 inst/HFMA2), and a
kernel at 40% of two independent ceilings looks latency-bound. Occupancy was the
obvious remaining suspect and `T.annotate_min_blocks_per_sm` the one-line handle.

`ncu` is denied on this pod (ERR_NVGPUCTRPERM), but **`nvcc -Xptxas=-v` is not** —
it reports registers, spills and the occupancy limit with no performance counters at
all, in a 5-second static rebuild of tilelang's cached `device_kernel.cu`. That part
of the method is worth keeping: four indirect A/Bs went by before anything in the
loop reported a register count.

Threshold committed before measuring: **≥1.15× at M=32, no M=1 regression.**

## The SASS numbers (these stand)

    ptxas info : Used 255 registers, 16 bytes cumulative stack size
                 16 bytes stack frame, 24 bytes spill stores, 20 bytes spill loads

255 is the per-thread maximum; 255 × 128 = 32640 of the SM's 65536 registers, so one
resident block, 4 warps. Instruction mix of the shipped `_xh` M=32 kernel:

| opcode | n | share |
|---|---:|---:|
| **HFMA2** | 1280 | 43.6% |
| LDG | 363 | 12.4% |
| HADD2 | 325 | 11.1% |
| FADD | 320 | 10.9% |
| FFMA | 192 | 6.5% |
| SHFL | 161 | 5.5% |
| total | 2936 | |

2936 / 1280 = 2.29 instructions per HFMA2 → issue ceiling 31.3 / 2.29 = **13.6
TFLOPS**. The `HADD2.F32 R, X.H0_H0, -RZ` + `FADD`/`FFMA` block is the per-tile
scale application (kernels_linear.py:652-653): **837 instructions, 28.5%**, against
1280 doing the arithmetic.

Probed register budgets, same kernel:

| minBlocksPerSM | registers | blocks/SM | warps | spill stores |
|---:|---:|---:|---:|---:|
| 1 (shipped) | 255 | 1 | 4 | 24 B |
| 2 | 255 | 1 | 4 | 24 B |
| **4** | **128** | **4** | **16** | **180 B** |

`-maxrregcount` is ignored here — `__launch_bounds__` overrides it, which is worth
knowing before reaching for the compiler flag.

## Results

`scripts/ab_gemv_variant.py`, `VARIANT = {"min_blocks": 4}`. `relerr = 0.00e+00`
everywhere — bit-identical, as a register-budget change must be.

| M | shipped | min_blocks=4 | gain |
|---:|---:|---:|---:|
| 1 | 20.7 ms | 20.6 | 1.00× |
| 8 | 73.8 | 87.8 | **0.84×** |
| 32 | 271.9 | 272.6 | **1.00×** |

Per shape at M=8, sorted by N, with the grid each one launches:

| shape | N | grid | blocks/SM | gain |
|---|---:|---:|---:|---:|
| down | 5120 | 1280 | 16.0 | **0.63×** |
| gdn out | 5120 | 1280 | 16.0 | **0.55×** |
| attn o | 5120 | 1280 | 16.0 | **0.55×** |
| qkv | 14336 | 3584 | 44.8 | 1.06× |
| qkvz | 16384 | 4096 | 51.2 | 1.04× |
| gate_up | 34816 | 8704 | 108.8 | 1.05× |

## What it decides

**1. Occupancy is not the M=32 cap.** 4 warps → 16 warps is 4× the latency-hiding
capacity and M=32 moved 0.3%. If four resident warps were failing to cover the
LDG latency, this could not be flat. Rejected.

**2. My "N small starves the SMs" reading of the M=8 split was also wrong.** Every
shape queues ≥16 blocks per SM — 1280 blocks over 80 SMs is not starvation. The
split is not about filling the machine: the three shapes that lose are the three
*fastest* ones (99.8-294.5 µs), where 180 bytes of spill traffic per thread is a
larger fraction of a shorter kernel; the three that gain are the slow ones, where
the extra warps pay for it. That is a spill/parallelism trade whose sign depends on
kernel duration, not on grid size.

**3. What survives.** The register number and the instruction mix are facts:
255 regs, 1 block/SM, 2.29 inst/HFMA2, issue ceiling 13.6 TFLOPS. Measured M=32 is
17.6% of the 31.3 FMA peak = 40% of that issue ceiling and 35% of the L1-bandwidth
ceiling. **Five candidates are now excluded by measurement** — FMA chain,
`n_partition`, L1 capacity, L1 bandwidth, occupancy — and the gap is unexplained.

That is where task #26 stands. It is not a mystery worth another five A/Bs at this
prize: prefill is 8.92 ms/token and halving the gap saves ~27 s of a 36.5 s TTFT,
but every mechanism I can name has now been measured and rejected, which means the
next attempt needs a *different instrument*, not another hypothesis. `ncu` is the
instrument, and it is denied on this pod (ERR_NVGPUCTRPERM) — getting counters
enabled, or reproducing on a card where they work, is the actual next step.

## Rule

**Do not write the win entry before the A/B returns.** I published
`gemv-is-occupancy-capped` on the strength of a register count and a plausible
mechanism, with the measurement still running. The count was right and the mechanism
was wrong, which is exactly the failure mode the loop's own rule ("a roofline is only
a bound if it is the binding constraint") was written for — and I reproduced it one
entry later, in the entry that quoted the rule.

Second: **a fix that is bit-identical and moves nothing has told you something.**
1.00× at 4× the warps is not a null result; it is a positive exclusion, and it is
worth more than the four indirect A/Bs that preceded it, because it kills a
mechanism rather than a parameter.

Third: **when the per-shape signs disagree, read the axis that actually separates
them.** I reached for N (grid size) because the losers shared N=5120; the axis that
separates them is kernel duration against a fixed spill cost. Both correlate with N
in this shape set, which is why a wrong axis was available.

## Gate

`scripts/ab_gemv_variant.py` with `VARIANT = {}` is the noise floor; the
`min_blocks` flag stays in `make_linear_fp4_gemv_sm70_m` (default 0 = tilelang's
own launch bounds, i.e. shipped behaviour unchanged) so the A/B is reproducible.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 3a94952 | V100 | cuda sm70 | GEMV M=32 | `min_blocks=4` | **1.00× — reject** |
| 2026-09-03 | 3a94952 | V100 | cuda sm70 | GEMV M=8 | `min_blocks=4` | 0.84× |
| 2026-09-03 | 3a94952 | V100 | cuda sm70 | GEMV M=1 | `min_blocks=4` | 1.00× |
