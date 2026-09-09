# reward_rises and ce_falls stay green during eval regression — 2026-09-09

**Status:** run `d6447c0abe5b` (train on MATH L5, eval on GSM8K, λ=0.1,
cap 6144) passed its `reward_rises` validity gate while GSM8K eval
correctness was flat-to-down. `ce_falls` did not object either — it is
skipped on every RL run. These are the second and third empty gates,
after `tied` (see [the three-defects entry](2026-09-09-math-l5-run-three-defects.md)).
`reward_rises` is worse than `tied`: `tied` is always green and says
nothing, while `reward_rises` is green *because the training batch moved
the wrong way*, which reads as a pass signal.

## Context

The gate compares `reward_last > reward_first` (cli.py:1368), where both
are means of the per-step training reward over the first and last
`--patience-window` steps. The training reward is the length-aware
quantity GRPO optimizes: `correctness − λ·tokens/cap` (cli.py:580-583),
averaged over 8 completions per step on the MATH L5 training batch.

On this run:

| step | mean reward | mean tokens | n |
|---|---|---|---|
| 1 | 0.7460 | 246 | 8 |
| 10 | 0.9941 | 363.5 | 8 |

`reward_rises`: 0.9941 > 0.7460 → green.

Over the same run, GSM8K eval correctness on the same 100 problems
(curve subset, evaluated at base, step 5, step 10):

| point | correctness | mean tokens |
|---|---|---|
| base (500-row batch) | 0.880 | 1925.7 |
| step 5 | 0.790 | 1353.3 |
| step 10 | 0.800 | 1252.8 |

Step 5→10 is noise (r2w=4, w2r=5, McNemar p=0.180). Base→step 10 is −8
pt, but base and curve are different batch compositions, so the net sits
inside the cross-batch floor (Defect 3 in the three-defects entry). The
honest read: eval correctness was flat across the training steps and
down from base, and the gate was green throughout.

## Root cause

`reward_rises` asks a training-population, length-aware, n=8 reward to
guard eval correctness. Three layers of removal:

1. **Wrong population.** The training batch is 8 MATH L5 completions;
   the eval is 100 GSM8K problems. The model can improve on the training
   distribution while the eval holds or drops — that is the
   generalization gap, and the gate cannot see it.
2. **Length-confounded.** At λ=0.1 the reward rises when completions
   shorten, independent of correctness. Eval tokens fell 1925.7→1252.8
   across this run; the length term alone improved reward by +0.014,
   enough to mask a correctness drop of 1.4 pt. On the training batch
   tokens rose (246→363.5), so the reward rise there was
   correctness-driven — but the gate's *green state* is still
   uninformative about eval correctness, because the gate reads the
   training batch and the length term can inflate either side.
3. **Underpowered.** Eight completions per step. Step-to-step swings are
   large (step 6: 0.2165, step 7: 0.9880). Comparing step-1 and step-10
   means, each n=8, is a coin flip on any real signal.

The design comment (cli.py:1308-1314) is explicit that `reward_rises`
must never *make* P1 pass — it is validity-only, and "reward not rising
is informative" (run 2 collapsed that way). That is true as far as it
goes. The hole is the converse: reward rising is not informative about
eval correctness, and the gate is green by default whenever the training
batch improves — which is every run that does not collapse. A validity
gate that cannot object during an eval regression is not guarding
anything; it is decorating the run with a green check.

`ce_falls` is emptier. On the RL path `ce_first` is never written (the
GRPO branch does not produce it, cli.py:1355-1358), so the gate is
skipped on every RL run — it cannot object even when it should. On the
SFT path it reads cross-entropy, which has the same length confound:
shorter completions drop the hardest tokens, and CE falls.

## Fix

`reward_rises` should read an eval-population correctness quantity, not
the training reward. The curve eval already produces per-step
correctness on a fixed 100-problem subset; a gate on that (e.g.
`curve_correctness_last > curve_correctness_first`, or a paired test on
the fixed subset) reads the population P1 cares about and is not
length-confounded. `ce_falls` should be removed on RL (it is already
skipped — delete the gate definition for that path) or, on SFT, read a
length-normalized CE.

## The gate that is sound, and the arithmetic that proves it

`gsm8k_improves` (cli.py:1370) reads eval-population correctness —
GSM8K 500-row, `after ≥ before + 5 pt`. It is the one verdict gate whose
threshold has a measured floor, and the floor is worth recording because
the next person to tune the threshold needs it.

Cross-batch floor on GSM8K-500 (run `76a17ea6e10a`, prior session):
**52 flips / 10.4% gross, net 2.** Under the symmetric null the net is a
random walk: SD = √52 ≈ 7.2 questions = **1.44 pt**. The +5 pt threshold
is 5 / 1.44 = **3.5 SD**, two-sided p ≈ 0.0005. The threshold cannot go
below ~3 pt or it falls inside the cross-batch noise and the gate starts
passing batch-composition flips as policy gains.

`gsm8k_after − gsm8k_before` must also report gross flips, not just net —
the net and its composition can point in opposite directions (the same
reason McNemar exists). A net of +2 built from 27 r2w / 29 w2r is noise;
a net of +2 built from 2 r2w / 4 w2r is a real shift in a small
subgroup. The net alone cannot tell them apart.

## Rule

A validity gate must read a quantity that moves with the condition it
guards, on the population it guards. `reward_rises` reads a
length-aware, n=8, training-population reward and asks it to guard eval
correctness — three layers of removal, and green by default. A gate that
cannot object during the regression it exists to catch is not a gate; it
is a green check that costs nothing and means nothing.

## See also

- [The MATH-L5 run exposed three defects in its own instrumentation](2026-09-09-math-l5-run-three-defects.md)
  — the `tied` empty gate, the patience-rule noise, and the cross-batch
  floor this entry's arithmetic builds on.
