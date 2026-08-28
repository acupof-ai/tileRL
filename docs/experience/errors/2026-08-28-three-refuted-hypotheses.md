# Three predicted wins that measured flat — 2026-08-28

## Context

One afternoon, three changes were made on an argument rather than a
measurement. All three were correct code. None moved the metric. They are
recorded together because the pattern is the finding.

## Root Cause

**1. mma8 block scales in bf16/f16.** Argument: four lanes of a group read the
same 4-byte f32 scale, so L1 sees ~2.5x the weight bytes; e4m3 scales are exact
in bf16, so halve them. Measured B=8 (the only rows that enter mma8): 0.998x
and 0.991x. Batched decode is not bound on that traffic. Reverted — it also
added a real footgun (the fp4 arm multiplies in bf16x2, the fp8 arm in f16x2,
so the storage dtype has to match each arm's `cvt`, which cost two debug
cycles).

**2. GDN norm reduce by block allreduce.** Argument: inside
`gdn_chunk_fused`'s serial token loop, thread 0 alone sums K=128 twice for the
q/k L2 norms — 256 dependent FMAs on the critical path of every token, "roughly
half the kernel" at T=512, with 127 threads idle. Measured: **1773.6 us vs
1775.7 us**. Not half the kernel; not measurable at all. The cost is the
per-thread serial K=128 delta-rule chain, which only the chunked (matmul)
formulation removes. Kept anyway — unlike (1) it adds no invariant, it deletes
a serial loop in favour of the idiom `rmsnorm_fused` already used.

**3. Frozen-backward output tile 64 -> 128 columns.** Argument: 64 columns is
32 packed bytes against an 8704-byte row stride, and the kernel measured 58
GB/s, 3% of HBM. Measured: the 1x256 train row went 0.962x, past the gate.
Reverted.

A fourth, from the same afternoon and the same habit, is recorded separately:
the fp4 dequant kernel was predicted to fix an 80 s training step and bought
1.18x — the step was launch-bound
([wins/2026-08-28-gdn-backward-launch-bound.md](../wins/2026-08-28-gdn-backward-launch-bound.md)).

## Fix

Measure the bound before optimizing for it. A roofline fraction (3% of HBM,
2.5x the weight bytes) says a kernel is *inefficient*; it does not say which
resource it is waiting on. `ncu` or an A/B on the one variable answers that; an
arithmetic argument does not.

## Rule

An estimate is a hypothesis, not evidence — and the cost of testing it is one
chain run, while the cost of shipping it is a permanent invariant. Order the
day by (measured gap x confidence in the mechanism), and when the mechanism is
a guess, spend the run before the diff. Three of four guesses were wrong on a
codebase this agent has worked in all day.
