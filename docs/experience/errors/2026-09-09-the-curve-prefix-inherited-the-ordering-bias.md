# The curve prefix inherited the ordering bias — 2026-09-09

**Status:** fixed before any curve was read at n=200. The curve subset was the
file's first N rows; the eval file is ordered, so that prefix runs 5 pt low — a
bias the size of the effect the curve exists to measure.

## Context

The steps-to-score curve now spends ~90% of its wall clock measuring, not
training: the fine curve paid 68 min eval against 7.7 min training, and one
eval point costs 614–1104 s (linear in tokens generated, not rows). Every
future curve — the second run, two-arm controls, the λ control, the early-stop
validation — pays this, so halving the eval rows is the highest-leverage lever
on iteration price.

`--eval-curve-n 200` was the obvious cut, and it was refused earlier the same
day for a reason that is still correct: `gsm8k_test.jsonl` is **ordered** — its
first 200 rows run 5 pt low (z=3.05, [2026-09-04](2026-09-04-the-eval-cap-measured-itself.md)).
A 5 pt bias against a +6.6 pt effect is not noise; it is a confound that can
flip a crossing step's verdict.

## Root cause

`cli.py` took the curve subset as `eval_rows[:args.eval_curve_n]` — the file's
prefix. The 2026-09-04 finding was recorded against the eval arm; the curve
code, written later, copied the prefix idiom and the bias with it. The
instrument's own help text said the subset "must be FIXED ACROSS RUNS … rather
than a sample" — pairing was the requirement, and a prefix was only *one* way
to satisfy it. A fixed-seed shuffle satisfies it just as well and is unbiased.

## Fix

`_curve_rows(eval_rows, n, seed)` shuffles a copy with `random.Random(seed)`
and takes the first `n`. `--eval-curve-seed` (default 0) lands in the manifest
inputs and the run id, so the subset is fixed across steps within a run,
comparable across runs, and impossible to change silently. The before/after
arms still use `--eval-n` untouched — they compare against the historical
anchor, so their sampling cannot move.

## Verification

**Negative control:** replacing the fixed seed with `random.shuffle` makes the
determinism test fail; restoring it passes.

**Unbiasedness on real data** (run 86a06dc8c420's `eval-before.jsonl`, base
policy, 500 rows):

| subset | score | vs full 500 |
|---|---|---|
| full 500 | 87.4% | — |
| file prefix 200 (old) | 85.0% | **−2.40 pt** |
| shuffle(0) prefix 200 (new) | 88.0% | +0.60 pt (2×SE of the diff = 5.55 pt) |

The old prefix is low on this policy too; the shuffled subset is unbiased
within noise. (The −2.40 pt here is a different policy and measurement from the
−5 pt / z=3.05 recorded in 2026-09-04 — same direction, not a replication.)

## The price of the cut, stated plainly

n=500 → 200 makes each eval point 2.5x cheaper. The paired-difference SE grows
√(500/200) = 1.58x, from 1.31 to ~2.07 pt. The base→step-5 effect (+6.6 pt)
goes from 5.03σ to 3.2σ — still clearly detectable. **What n=200 cannot
adjudicate is a ~0.2 pt floor comparison** between two late curves. Hence the
split: curve points at n=200 to find the crossing step cheaply; the anchors
(base and the last point) stay at n=500 for comparison with history.

## Rules

- **A fixed subset and an unbiased subset are two requirements; a prefix buys
  the first and silently violates the second when the file is ordered.** Name
  both when choosing the sampling.
- **A bias the size of the effect is a confound, not noise.** The 5 pt ordering
  bias against the +6.6 pt learning effect is why n=200 needed the shuffle
  first; without it, the cheaper instrument would have been the more
  misleading one.
- **State what the cheaper instrument cannot see.** n=200 cannot adjudicate
  0.2 pt floor differences; the n=500 anchors exist for exactly that.
