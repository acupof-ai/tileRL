---
question: What is the most that batching the DFlash2 drafter could buy, with the drafter's own cost removed?
source: H20 sm90 card 6, tilelang 0.1.13, torch 2.11.0+cu129, 27B NVFP4 + Qwen3.8-27B-DFlash2, main ac9eaee, scripts/probe_verify_ceiling.py
---

# The drafter-batching ceiling is 3.64x, and 3.64x is a division

**Read this first: `6.20 / 1.70 = 3.65`.** The ceiling is the measured forward
reduction divided by the measured tick cost. It is not a third number standing
beside those two, and it is not an independent confirmation of anything. Both
figures are in the table below so the quotient cannot be mistaken for a finding
of its own.

What the probe contributes is that **the schedule is measured rather than
assumed**: 1719 verified row-ticks, all of them W=8, 10651 committed tokens,
reconciled against the engine's own counter, with no draft head in the timed
process. The 6.20x forward reduction was previously an inference from a
profile's self-time shares; now it is a replayed schedule.

## The numbers

`record` ran the real DFlash2 arm and logged `(width, n_ok, committed)` per
verified row. `price` rebuilt the schedule on an engine holding **no draft head**
(asserted, not intended) and replayed the captured decode graph at each width the
trace used, 20 reps each.

| | |
|---|---:|
| tokens committed through verify | 10651 |
| verified row-ticks | 1719, **all W=8** |
| spec engine ticks (B=8) | 214.9 |
| base engine ticks (W=1) | 1331.4 |
| **forward reduction** | **6.20x** |
| W=1 graph replay | 26.027 ms |
| W=8 graph replay | 44.309 ms |
| **tick cost, W=8 vs W=1** | **1.70x** |
| spec trunk time | 9.52 s |
| base trunk time | 34.65 s |
| **ceiling, free drafter** | **3.64x** |

An independent extrapolation from a py-spy profile of the same configuration
(drafter 68.4% of the arm, trunk replay 13.2%) gave ~3.4x. That agrees to 7% —
but it is **not a second route to the ceiling**, because it rests on the same
tick-cost ratio. Two routes to the tick's composition; one division on top of
both.

## What it licenses

Building the batching tranche. The prize is ~3.5x, not ~1.2x, and that was the
open question: DFlash2 measures **1.67x slower end to end** with the drafter at
68.4% of the arm, so whether removing that share lands anywhere useful decided
whether the work was worth doing.

## What it does not license

Any claim about the achievable speedup. Two limits, both structural:

- **A batched drafter is not free.** The ceiling removes the drafter entirely.
  The achievable number is the ceiling minus the batched drafter's own cost, and
  nobody has measured that subtrahend. It is the whole remaining question.
- **Acceptance is an input, and batching may move it.** 74.75% (8995 of 12033)
  was measured with the per-row drafter. `path()`'s walk is serial with a data
  dependency on `prev`; batching across rows keeps the arithmetic identical, but
  restructuring the walk itself would change acceptance, which changes the tick
  count, which changes this number.

## The tick cost is 1.70-1.81x, not 1.79x

Measured today at **1.70x**. The recorded values from earlier the same day are
**1.79x** (card 7) and **1.81x** (card 5), from `profile_verify_replay.py`.

Three runs, two cards, two harnesses, and a `main` that moved in between
(#34's drafter wiring, #35's pad-row reservation). The honest object is the
spread; the point estimate was never entitled to three significant figures.

Candidates, unseparated by a single run: host contention (cards 0-3 at 100%
throughout, card 7 at 31%); the code moving; or the harness differing — this
probe builds its pools directly with `keep=W` explicit, `profile_verify_replay`
goes through `build_engine`. The cheap discriminator is the two harnesses on one
card back to back, which separates the third candidate from the first two in one
run.

**It does not move the ceiling's magnitude**: at 1.79x the ceiling is 3.46x
instead of 3.64x, so ~3.5x either way. It matters for anything that fits a
constant to it — a cost model cannot claim more precision than the quantity has.

## Rule

When a headline figure is a quotient of two measured figures, print all three
together. A ratio quoted alone acquires the authority of a measurement, and the
next reader adds it to the list of confirmations instead of dividing.

A ceiling is not a forecast. "The most this could buy with the cost of the thing
being changed set to zero" is a different claim from "this is what it will buy",
and the gap between them is exactly the quantity nobody has measured yet.
