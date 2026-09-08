# Difficulty is a property of the task; hesitation is a property of the model — 2026-09-08

**Status:** the reason level 5 was the wrong task for P1, stated as the selection rule rather
than as a fact about that dataset. Written on `tilerl-27`'s instruction; the measurements are
`tilerl-0a`'s.

## What happened

MATH level 5 was chosen for P1 because it is harder than GSM8K. Three successive readings of
its base accuracy:

| reading | value | what it actually was |
|---|---:|---|
| first | 45.8% | a truncated-generation artifact |
| second | 64.0% | the **lower bound** of `[64%, 96%]`, reported as a point |
| third | **91.0%** | cap 6144, the 32 capped questions re-run |

GSM8K's base is 88%. **Level 5 is easier**, and the belief that it was harder rested on two
instrument artifacts in a row.

Worse, the gate is then unsatisfiable. At cap 6144 the interval is `[91%, 94%]` — 91 known
correct, 6 known wrong, 3 still hitting the new cap — and P1 requires base + 5 = **96%**. No
adapter can pass: perfecting all 6 wrong answers and all 3 unknowns reaches 94.

## The selection error underneath it

Fixing the gate would not fix this, because the premise of the choice is gone. And the premise
was wrong in a way that survives changing datasets: **we selected on difficulty, and GRPO
consumes hesitation.**

- **Difficulty** is a property of the task. It is model-independent, comparable across
  checkpoints, and it is what a benchmark's level labels encode.
- **Hesitation** is a property of *this model on this task*: does it produce different answers
  across samples of the same question. That is what makes a GRPO group carry gradient — a
  group whose eight rollouts agree has zero advantage regardless of whether they agree on the
  right answer.

A hard task the model always fails and an easy task it always passes are the same thing to the
optimizer: no signal. **Difficulty predicts nothing about which of those a task is.**

## Why base accuracy cannot stand in for tied fraction

The natural repair is to keep base accuracy and map it to tied fraction. That fails for a
reason worth stating precisely, because it is stronger than "the proxy is noisy".

The mapping runs through the group's *effective* independence. An earlier measurement put eight
rollouts at the behaviour of 3.4 independent ones
([the cap was the gradient](../wins/2026-09-04-the-cap-was-the-gradient.md)), so at base 0.91
the all-correct ties alone are 0.91^3.4 = **72.6%**. But 0a's point is that effective
independence is *itself a function of base*: a higher base means a more homogeneous group,
which means fewer effective samples, which means more ties. So 72.6% is a **lower bound**, and
the base → tied curve has a slope that changes with base.

**A proxy whose mapping slope varies with the proxy does not even preserve ordering.** Two
tasks can rank one way on base accuracy and the other way on tied fraction. That is why this
is not a calibration problem.

## What replaces it

The distribution of correct-counts over 8 samples per question, measured directly. One
measurement yields base (the mean), tied fraction (the 0/8 and 8/8 mass), and the
gradient-bearing subset (1/8 through 7/8) — **the target quantity rather than a better proxy
for it.**

Also: the searched-for property is not a harder dataset. It is questions this model is
uncertain about, which may well be easy ones.

## Rules

- **Select a training task on the quantity the optimizer consumes, not on a property of the
  dataset.** GRPO eats within-group disagreement; difficulty labels do not measure it, and a
  task the model always fails is as useless as one it always passes.
- **A proxy whose mapping to the target has a base-dependent slope cannot even rank
  candidates.** Check whether the mapping is monotone in the proxy before treating a proxy
  comparison as a target comparison.
- **Check a gate against its feasible range, not only against its strictness.** Two P1 gate
  defects in one day: one looser than the roadmap's spec, one demanding 96% where 94% is the
  ceiling. Neither was "too strict" — nobody had computed whether they were satisfiable.
- **Report a capped sample's contribution as unknown, not as its worst case.** `[64%, 96%]` was
  32 points wide, 6.4x the effect P1 exists to detect; quoting 64% as the base chose one end
  and discarded the interval.
