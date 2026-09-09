# The best curve point is saved, and a tie keeps the earlier point — 2026-09-09

**Status:** shipped. `src/tilerl/` runtime change on the ledger read path and the
curve-point write path — not kernel/engine/step. The correctness evidence is the three
negative controls below, not timing; there is no perf arm in this entry.

## Context

The 2026-09-08 GRPO run scored 87.4 / 93.2 / 93.4 / 82.4 / 91.2 and **shipped 91.2** — the
last point, 2.2 below the peak it had reached 50 steps earlier. The peak weights were
unrecoverable: `AdamW.step_one` ends in `p.copy_()`, in place, so every intermediate
policy is destroyed by the next step (the same property that lets the engine keep its
captured graphs). The early-stopping design doc (Q4: snapshot first) concluded that a snapshot is the
prerequisite for any stopping rule, and that it should land first, independent of one.

## What changed

`score_curve` (`cli.py`) now snapshots the trainable tensors when a new curve point
becomes the best, and the run writes them as `adapter-best.safetensors` beside the final
adapter. The manifest records `best_curve_point` = `{step, score, mean_len,
tok_per_correct, se_kind, every}`.

Two decisions, both forced by measurements:

1. **The criterion is "significantly greater", not "numerically higher".** This eval's own
   floor is 0.2 pt: one fixed set of weights, re-scored across processes at temperature 0,
   moved 438/500 to 437/500. The run's step-50 point led step 25 by exactly one question;
   taking the numerically higher point would have bought 501.2 s of extra training for a
   reading inside the instrument. A tie keeps the earlier point — `time_to_score` is the
   objective, so at equal score the cheaper point wins.

2. **The width is paired.** Every curve point scores the same 500 rows, so the SE of a
   difference between two points is `100 × sqrt(b+c)/n` over the discordant pairs — 1.9x
   narrower than the two-arm unpaired width at the measured 8.6% discordant rate. The
   unpaired width would make the criterion never fire on a slow rise: a selection that
   always keeps the first point and does not say so. Rows missing (old runs) fall back to
   the conservative width, marked `se_kind: "unpaired (conservative)"` in the manifest so
   nobody reads a conservative "not greater" as "the two points are the same".

The criterion lives in `ledger.new_best_point`, shared by the run and its post-hoc
readers — two formulas would let the run keep a point the ledger then calls
indistinguishable.

## Correctness evidence: three negative controls

The runnable check is `python -m tilerl.ledger`. Each control mutates the criterion, goes
red at the named cell, and was reverted green:

| control | mutation | result |
|---|---|---|
| 1 | criterion as plain `>` | red — selects the numerical peak (step 15, 94.6) |
| 2 | every point wins (take the last) | red — selects step 100, 91.2: exactly what the run shipped without the snapshot |
| 3 | paired SE ignored, always unpaired | red — a 3.0 pt gap fires paired (2×1.31 = 2.6 < 3.0) and not unpaired (2×2.02 = 4.0 > 3.0); this cell is the one that proves which SE is used |

Cells 1–2 cannot distinguish the paired from the unpaired width — both give the same
answer on them — so cell 3 exists for that.

## Rule

A selection criterion's width must match the quantity it compares: points scored on the
same rows get a paired SE, and an eval's floor is measured on the instrument itself, not
assumed. A check that can always return the same answer — the unpaired criterion on a slow
rise — is not a check; the negative control that proves it can fire is the important part
of the change, not the implementation.

## Open dependency

Whether this snapshot is "rescue one accident" or "a prerequisite for this recipe" is not
yet known: a second curve (`--seed 1`, otherwise identical, run by tilerl-9b) decides
whether the step-75 collapse reproduces. Also note no run in this tree has ever stored
completion text (`train.py:578-582` keeps step/p/g/tokens/reward/advantage only), so the
2026-09-08 collapse can never be replayed at the text level — #328 closes that. This entry
describes the mechanism; the population question belongs to the seed-1 run.
