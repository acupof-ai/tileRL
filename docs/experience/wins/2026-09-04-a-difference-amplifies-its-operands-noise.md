# The draft-forward number amplifies tick noise 12.9x — sm70, 2026-09-04

> Status: method finding, no code change. It bounds how precisely Task #22's
> block-parallel verdict can ever be stated by this instrument, and says what a
> tighter one would have to do.

## Context

Every block-parallel number in this repo descends from one quantity: the cost of a
single draft forward. It is obtained by subtracting two tick means that share a rung —
depth 2 and depth 3 both verify on rung 4, so their difference is exactly one draft
forward. That subtraction is what makes the number honest (it cancels the verify cost
without needing to model it) and it is also what makes it fragile.

## What Worked

**The subtraction amplifies relative error by the ratio of the operands to their
difference.** Two runs of the identical command, three wikitext passages each:

| run | rung-4 tick @ depth 2 | @ depth 3 | difference = one draft forward |
|---|---:|---:|---:|
| ds8 | 56.9 | 62.2 | **5.30 ms** |
| ds10 | 56.9 | 61.7 | **4.80 ms** |

The depth-3 tick means agree to **0.81%**. The draft forwards derived from them differ
by **10.42%** — an amplification of **12.9x**, which is just 62/(62−57), the operand
over the difference. Nothing is wrong with either run; this is what subtraction does.

**Propagated into the quantity the verdict rests on**, holding rung-4 verify at its
independently-derived 46.09 ms:

| draft forward | draft share of a depth-3 tick | block-parallel ceiling |
|---:|---:|---:|
| 3.93 ms (the gate's constant) | 20.4% | 1.157x |
| 4.80 ms (ds10) | 23.8% | 1.189x |
| 5.30 ms (ds8) | 25.6% | 1.206x |
| 6.40 ms (B=2, a different batch) | 29.4% | 1.244x |

So the ceiling is **1.16x to 1.21x** on wikitext at B=1, and quoting it to three
figures is quoting the run, not the system.

**The verify constants do not have this problem.** They are measured directly, not
differenced, and the offset between the gate's `VERIFY_MS` and tonight's numbers is
*systematic* rather than noisy — 1.0716 / 1.0744 / 1.0498 at rungs 2 / 4 / 8. A
uniform ~5-7% across three independent rungs is a prompt difference, not instrument
error, and it is conservative in the direction the gate needs (the gate over-states our
own cost, so a bound that holds there holds for the faster tree). Left alone
deliberately.

**The draft is also not a per-batch constant**, which is a second reason a single
figure cannot be quoted: 4.80-5.30 ms at B=1, 6.40 at B=2 (solved from depths 2 and 3
both on rung 8), and **11.10-11.40 at B=4** (depth 1's rung-8 tick against B=2's
verify — two runs of that arm read 94.5 and 94.8 ms, so this difference carries the
same amplification). It is a batched forward like any other.

The amplification factor is just operand/difference, and it checks out both ways:
62.2/(62.2−56.9) = 11.7, 61.7/(61.7−56.9) = 12.9, against a measured 12.9.

## Rule

**A difference of two large measurements needs its amplification factor stated beside
it.** The factor is operand/difference — compute it, and if it exceeds ~10, the
derived number carries a full order of magnitude more relative error than the
measurement it came from. Concretely here: 0.81% tick agreement became 10.4% draft
disagreement, so a verdict resting on a 1.7% margin cannot be settled by this
instrument at all, whatever the tick repeatability says.

The cheap fix is not more runs — averaging two noisy differences converges slowly.
It is to **measure the draft forward directly** rather than by subtraction: time
`DraftHead.step` alone with CUDA events, or profile by kernel and sum the draft's own
launches, both of which have no cancellation. The per-kernel profiler already
attributes this way; it was not used for this number.

Second: **an offset is not noise, and the two want opposite treatment.** A discrepancy
that is uniform across independent measurements has a cause worth naming and often no
fix; one that varies run to run is instrument error and needs the instrument changed.
The `VERIFY_MS` gap is the first kind and stays; the draft gap is the second kind and
is why the block-parallel verdict is stated as REJECT-on-parameter-count rather than
REJECT-on-margin.

## What this does NOT license

**It does not reopen Task #22.** The reject is carried by DSpark's 4.08x parameter
count against a 2.36x budget, which no draft-cost precision touches. What this entry
retires is the *margin* argument: 1.016x, 1.035x and 1.062x have all been printed for
that margin at various times, and none of them is resolvable by this instrument.

**It does not invalidate the tick numbers.** Ticks repeat to 0.3-0.8% across runs and
1.16% is the harness noise floor. Only quantities *derived by subtraction* carry the
amplified error.

## Results

No runtime change; no rate row. Source numbers: `$HOME/tilerl-logs/ds8.log` and
`ds10.log` on the V100 (the two agreeing wikitext runs), `ds12.log` (B=4).
