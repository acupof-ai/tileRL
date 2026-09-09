# A net is one equation: step 50 and step 100 differ on 37 problems, and the score says 11

**Date:** 2026-09-09
**Session:** v100-sm70-fp4-cc
**Run:** `86a06dc8c420`, card 3, coarse curve 25/50/75/100, per-problem rows from `eval-curve-*.jsonl`

## Context

The four-point curve (same 500 GSM8K problems at every point, greedy) —
[the gsm8k reward-vs-step curve](../wins/2026-09-08-the-gsm8k-reward-vs-step-curve.md) —
scored
466 / 467 / 412 / 456 correct at steps 25 / 50 / 75 / 100. Step 75 lost 55
problems against step 50; step 100 gained 44 back against step 75. The question
was whether step 100 walked back to step 50's policy — whether the +44 was the
same 55 (or so) problems returning.

The first pass asked for three sets: `L` = 50-right→75-wrong (62),
`R` = 75-wrong→100-right (56), and their intersection (49). Those three left
the books unclosed: the 50→100 net is 456 − 467 = **−11**, but
−|L\R| + |R\L| = −13 + 7 = **−6**. A fourth set was inferred from the net as
5 problems. The measured fourth set is **11**, and the net is satisfied by a
fifth cell nobody had named: 6 problems wrong at 50, right at 75, still right
at 100. −13 − 11 + 7 + 6 = −11.

## The full table

Per-problem decomposition over steps 50 / 75 / 100 (✓ = correct):

| 50 | 75 | 100 | count | reading |
|---|---|---|---:|---|
| ✓ | ✓ | ✓ | 394 | stable |
| ✓ | ✗ | ✓ | 49 | collapsed at 75, recovered by 100 |
| ✓ | ✗ | ✗ | 13 | collapsed at 75, never returned |
| ✓ | ✓ | ✗ | 11 | survived the dip, lost **after** step 75 |
| ✗ | ✗ | ✓ | 7 | wrong at 50 and 75, newly right at 100 |
| ✗ | ✓ | ✓ | 6 | wrong at 50, recovered by 75, held |
| ✗ | ✓ | ✗ | 1 | |
| ✗ | ✗ | ✗ | 19 | stable wrong |

All three marginals close: 394+49+13+11 = 467, 394+11+6+1 = 412,
394+49+7+6 = 456; the eight cells sum to 500.

## Root Cause

**A net score change is one equation, and a 2×2×2 movement has eight cells.**
"Step 100 recovered" was read from a net of +44, but a net cannot distinguish
problems returning from new gains offsetting new losses. The first request
compounded this: it asked for one intersection (49) and two set differences,
which is four cells of the eight, and the remaining gap was then *solved from
the net* — an underdetermined equation presented as a check. The predicted 5
and the measured 11 both satisfy −13 + 7 − x + 6 = −11 with the right
companion cell; the net has no opinion about which decomposition produced it.

Two findings the table settles, both invisible in the score column:

1. **Step 50 and step 100 are different policies on 37 problems**
   (13 + 11 + 7 + 6) — 7.4% of the set changed hands while the scores differ
   by only 11 problems (2.2 pt). This does not depend on the dip: even without
   step 75, "two points score nearly the same" is not "two points are the same
   policy". Every checkpoint-by-score selection, `best_curve_point` included,
   selects a score, and a score does not identify a policy.
2. **Training past the dip loses problems on its own.** The 11
   (✓,✓,✗) problems survived step 75 and died afterwards — nearly as many as
   the 13 the dip permanently destroyed. The case for early stopping does not
   need the dip; continued training past the plateau is a second, independent
   loss.

The 49 recoveries are 79% of the 62 collapsed problems — at the pre-registered
A/C boundary (≥80% would have read "reversible"), inside the ±1-problem
cross-process noise floor, so it is reported as a boundary, not rounded up.

## Fix

The curve has recorded per-problem rows since #323; the fix is procedural.
Before any "recovery" / "same policy" / "walked back" claim, build the full
2^k contingency table over the points being compared and check every marginal
closes. The alignment that makes the table meaningful is verified, not
assumed: 500/500 gold answers identical across the four files, and the code
slices `curve_rows` once outside the scoring loop (`cli.py:860`) and writes
`per_problem` in input order (`eval.py`, `enumerate` over the rows), so row `i`
is the same problem at every point within a run. Every cell here is ≥ 6
problems, far above the eval's 1-problem cross-process floor, so none of the
movements is noise.

## Rule

**A net is one equation; k binary outcomes have 2^k cells, and only the full
table pins the decomposition. Ask for the whole table, never one intersection
count.** The coordinator's first request named three sets and the follow-up
solved a fourth from the net — two passes, same mistake shape: a quantity the
net cannot determine was reported as if the net had determined it.
