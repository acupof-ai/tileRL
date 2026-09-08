# A default that cannot resolve the effect its own feature exists to find

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55
**Status:** open — the default is unchanged; see the Fix section for what a change would be.

## Context

`--eval-curve-n` defaults to 20 (`cli.py:1433`). The curve exists to answer one question:
at which step does the run first reach its target score. Tonight's target is the +5.6 pt gain
recorded for the 100-step GSM8K run
([P1 GRPO 27B](../wins/2026-09-05-p1-grpo-27b-run.md), 88.0% → 93.6%).

## Root cause

The curve's width is the binomial SE on its subset size, which `ledger.py:127` already
computes and prints:

| n | SE | against a +5.6 pt effect |
|---:|---:|---|
| **20 (default)** | **11.18 pt** | **unusable — 2.0x the effect** |
| 50 | 7.07 pt | unusable |
| 100 | 5.00 pt | 1.1 σ |
| 200 | 3.54 pt | 1.6 σ |
| 500 | 2.24 pt | 2.5 σ |

At the default, each point's sampling noise is twice the signal being located. The curve
runs, records well-formed points, and prints a crossing step — and that step is chosen by
which 20 rows fell where, not by the policy. The feature does not fail; it answers with a
number that carries none of the information it was built to carry.

`ledger.py:122-128` already knows this: it notes that n=20 "carries a binomial SE of 11.2 pt
at p=0.5, against P1's +5 pt target", and `_se_note` (`cli.py:1229`) prints a warning when
`se_pt >= 5.0`. **So the code both computes the defect and describes it, while the default
value sits below the threshold its own warning uses.** The warning is the mitigation, and a
warning printed after a 99-minute run is not a substitute for a default that resolves.

There is a real reason the default is small: the flag's help says "keep the scoring under 5%
of a step", and at n=500 on this model one point costs ~986 s against a 56.88 s step —
1733%. **The two constraints are in genuine conflict, and 20 was chosen for the cost side
without the resolution side being computed.** That is the actual defect: not a wrong number,
a number chosen against one of two binding constraints.

## Fix

Not applied tonight — changing it is work on the denominator, and the measurement it would
serve has not been made yet.

What a fix looks like, in the order I would try it:

1. **Refuse rather than warn.** If `--eval-every` is on and the implied SE exceeds the effect
   the caller is looking for, the run should not start. That needs the target as an input,
   which `--time-to-score` already takes at read time (`cli.py:1493`) — moving it to run time
   makes the precondition checkable before 99 minutes are spent.
2. **Price the two constraints together in the help text.** The 5%-of-a-step criterion and
   the SE are both functions of n in opposite directions; the flag documents one.
3. Only then consider a different default. n=100 costs 35% of the training time and resolves
   at 1.1 σ; neither figure is comfortable, which is the honest state of this tradeoff.

## Rule

A default has to be checked against the question its feature exists to answer, not only
against its cost. Both constraints here are real, and picking a value that satisfies the
cheap one produces a feature that runs, reports, and cannot answer.

Where the code already computes the quantity that condemns the default — as `ledger.py:127`
does here — that is the strongest available signal and the weakest available guard: a value
that is computed, printed, and then not used as a precondition.
