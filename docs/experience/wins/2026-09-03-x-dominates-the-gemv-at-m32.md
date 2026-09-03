# X dominates the sm70 GEMV at M=32 — V100, 2026-09-03

> Status: located by ablation, after five A/Bs excluded everything else and two of
> my own rejections turn out to have been wrong. `X_REUSE` is **8.75× at M=32**.
>
> **Correction, same day, before acting on it:** that 8.75× is an **upper bound on
> everything X-related**, not a measurement of X's loads alone. Making the address
> loop-invariant let ptxas hoist the load out of the m loop — SASS shows **LDG 363 →
> 53, so 85% of the loads were deleted, not cached**. The ablation therefore prices
> X's traffic *plus* its latency *plus* 85% of the LDG issue slots together. The
> conclusion "X is where the kernel's time goes" stands; "X's per-row loads are 89%"
> was too strong for the instrument. `abl=4` (PIPELINE, same 363 LDG) is the honest
> split and is running.

## Context

ncu is denied on this pod (ERR_NVGPUCTRPERM), and five mechanisms had been excluded
by A/B: FMA chain, `n_partition`, L1 capacity, L1 bandwidth, occupancy. Ablation
needs no counters: keep the instruction count and the load count identical, remove
one suspect's *cost*, read the delta. Each variant returns wrong numbers by
construction, which is what makes it a measurement rather than a candidate.

| flag | what it removes | how the counts stay equal |
|---|---|---|
| `abl=1` X_REUSE | X's traffic and latency | every row reads row 0's X — same 2 LDG + 8 FMA per row, L1 hit after the first |
| `abl=2` NO_SCALE | the per-tile widen + scale | accumulator stays live, drops HADD2.F32 + FADD/FFMA (837 of 2936 instructions) |
| `abl=3` NO_DECODE | fp4 dequant | raw words stand in for decoded halves, same registers |

## Results

`scripts/ab_gemv_ablate.py`, per-pass totals weighted by launches/token:

| M | base | X_REUSE | NO_SCALE | NO_DECODE |
|---:|---:|---:|---:|---:|
| 1 | 21.3 ms | 0.97× | 1.03× | 0.95× |
| 8 | 73.0 | **3.05×** | 0.98× | 0.98× |
| 32 | 270.8 | **8.75×** | 0.92× | 0.98× |

Per shape at M=32: X_REUSE 8.93 / 10.63 / 8.68 / 5.60 / 8.41 / 5.67×.

**X is where the M=32 kernel's time goes** — but see the correction above for how
much this number can carry: `nvdisasm` on the ablation's own cubin shows **LDG 363 →
53**, because a loop-invariant address is hoistable, so 8.75× includes 85% of the
LDG issue slots being deleted along with the traffic and the latency. It is an
upper bound on X-related cost, not a per-row load measurement.

What *is* clean, because these variants keep their loads: the scale tail — 28.5% of
the instruction stream by SASS count — is **free** at 0.92×, and so is the fp4
decode at 0.98×. Both are entirely hidden behind whatever X is doing. That retires
the "reduce instructions per flop" direction, which the previous entry derived as
the only remaining lever.

The M-dependence is the signature: 0.97× at M=1 (nothing to hoist or reuse), 3.05×
at M=8, 8.75× at M=32. Only a per-row cost can do that, and it is why M=1 sits at
83% of its bandwidth roofline while M=32 sits at 17.6% of its FLOP peak — those are
two different kernels wearing one template.

## Two rejections this reverses

**SMEM staging** (errors/2026-09-03-smem-staging-rejected-one-flop-per-x-byte.md)
was rejected because a two-K sweep showed no L1-capacity knee. The ablation says X's
loads dominate the time, so staging them is back on the table — I rejected the
right target on a wrong prediction.

**"L1 bandwidth is not binding"** rested on X reading at 5.51 TB/s = 35% of the
15.7 TB/s port. That number is a rate, not a limit: the loads can dominate the
kernel while running below the port's peak, which is what latency-bound means.

## Two models, both refuted by their own controls

**(a) X spills L1 (total working set).** Predicts a knee at `M = 131072/(2K)`:
M≈12.8 at K=5120, M≈51.2 at K=1280. Both measured curves flatten at M≈12-24. Dead,
and the ablation shows the target was right anyway — so the *test* was wrong, not
the suspicion.

**(b) Reuse distance.** What governs an X hit is not X's size but how much W streams
between two reads of the same X word: `n_partition · K/2`, which is **independent of
M** — that is why both knee curves flattened together. Predicts K=1280 (51 reuses
deep in L1) beats K=5120 (12 deep). Measured %FMA says the opposite at every M:

| | L1 depth | M8 | M16 | M24 | M32 | M64 |
|---|---:|---:|---:|---:|---:|---:|
| K=1280 | 51 reuses | 5.2% | 8.9% | 13.3% | 13.9% | 13.9% |
| K=5120 | 12 reuses | 14.9% | 15.6% | 16.6% | 17.6% | 18.2% |

4× deeper and slower everywhere. Dead too.

A third check bounds where X is served from: at M=32, `down` reads 5.70 GB of X.
From HBM that is 6338 µs; the kernel runs in 976 µs, so X is already cached
(effective ~5.8 TB/s, between the L2 and L1 rates). `X_REUSE` lands at 92 µs. So the
prize is real and sits between "cached in L2" and "cached in L1", but which
structural property of the access pattern puts it there is open.

## What this decides

The lever is X and nothing else at M=32; both other candidates measured free. The obvious form is staging the tile's X slice in SMEM so
the M row-reads hit it, which is what was rejected on model (a). **Before writing
it**, the open question is why the loads cost what they do, because a fix aimed at
the wrong mechanism is how the last two ticks were spent. Concretely: `abl=1`
removes both X's *traffic* and its *latency* at once, so it does not separate them.
An `abl=4` that keeps per-row addresses but prefetches the whole tile's X into
registers before the FMA block would split those.

## Rule

**Ablate before modelling.** Five A/Bs and two derivations went into ranking
candidates by plausibility; one ablation ranked them by cost, and the ranking was
nothing like the guesses (the 28.5% instruction tail is free; X dominates).
An ablation that returns wrong numbers is cheap, needs no counters, and answers
"how much does this part cost" without requiring a theory of why.

Second: **a wrong prediction refutes the prediction, not the target.** "X spills L1"
was falsifiable and false; "X's loads are the cost" was never tested and is true. I
retired the second on the death of the first, and lost two ticks to it.

Third, from the correction: **check that an ablation removed the cost and not the
instructions.** `abl=1` made the address loop-invariant, and ptxas did the obvious
thing — hoisted the load, LDG 363 → 53. The delta then prices the deleted issue
slots too. Every ablation needs its instruction count read back off the cubin, which
is one `nvdisasm | grep -c`, and I published the 89% figure without doing it.

Fourth: **an ablation that changes what the compiler can prove is not a
measurement of the hardware.** Loop-invariance is a compiler-visible property; the
variant that keeps addresses per-row (`abl=4`) changes only scheduling, which is
why it is the one that can actually split traffic from latency.

## Gate

The `abl` flag raises `ValueError` without `xh=True` and is never a serving path;
182 tests pass unchanged. The ablation kernels are compiled only by
`scripts/ab_gemv_ablate.py`.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 852f319 | V100 | cuda sm70 | GEMV M=32 | X_REUSE ablation | **8.75×** |
| 2026-09-03 | 852f319 | V100 | cuda sm70 | GEMV M=8 | X_REUSE ablation | 3.05× |
| 2026-09-03 | 852f319 | V100 | cuda sm70 | GEMV M=1 | X_REUSE ablation | 0.97× |
| 2026-09-03 | 852f319 | V100 | cuda sm70 | GEMV M=32 | NO_SCALE / NO_DECODE | 0.92× / 0.98× |
| 2026-09-03 | 81c83ff | V100 | cuda sm70 | GEMV M=32 | LDG, base vs X_REUSE | **363 → 53** |
