# The 256 cap was not only brevity pressure — it was what kept the reward from saturating

**Date:** 2026-09-04
**Run:** `grpo2k` — grpo-gsm8k-27b, `--max-new-tokens 2048 --max-think-tokens 1792`,
thinking on, group 8, 100 steps, card 3. The loose-cap control for
[the thinking cap](2026-09-04-the-thinking-cap.md).

## Context

The thinking-cap result trained at a 256-token cap and read out uncapped: +5.2
points and −22.8% tokens. The mechanism was written up as brevity pressure —
cap the rollout below what the model would freely spend and it finds the
shorter path.

That explanation predicts its own control. Train at 2048, where the measured
median completion is 283 tokens and only 0.5% of rollouts reach the cap, and
there is no budget pressure; the token saving should mostly disappear.

The control was run to test that. It found something the brevity story did not
predict.

## The measurement

Registered before the run's counters were read: tie rate ≈ 0.30, from `p⁸` at
p = 0.86. **It came back 0.887 at step 62, and 0.920 over the full run.** The
run has since finished; the table reads at step 62 so the prediction is scored
against the number that was available when it was registered.

| | 256 cap | 2048 cap |
|---|---:|---:|
| per-rollout accuracy, train prompts | 0.86 | **0.966** |
| steps with zero gradient | 72% | **88.7%** (55/62) |
| steps that trained | 17 of 61 | **7 of 62** |

Loosening the cap did not improve the training signal. It destroyed three
quarters of what was left.

## Why the prediction was wrong — three separate errors

1. **Wrong p.** 0.86 is the accuracy *under the 256 cap*. Uncapped the policy
   solves 0.966 of training prompts. The control's own input was the number the
   control was meant to change.
2. **Wrong event.** `p⁸` counts all-correct groups. A group is tied when it is
   all-*same*; all-wrong ties too. The correct independent form is
   `p⁸ + (1−p)⁸`.
3. **Wrong independence.** Even corrected, `p⁸ + (1−p)⁸ = 0.757` against 0.887
   observed. Eight rollouts on one prompt are not eight independent draws.

Solving `pⁿ + (1−p)ⁿ = 0.887` at p = 0.9657 gives the effective group:

    asked for                      8 rollouts
    effective independent group    3.44

**Eight correlated rollouts buy the diversity of 3.4 independent ones.**

## The full run, and the direction it moved

100 steps finished. The tie rate **rose** across the run — 88.7% over the first
62 steps, **92.0% over all 100** — which is the mechanism running forward: each
gradient step makes the policy slightly better at the training prompts, so the
next group is slightly more likely to agree, so fewer steps carry gradient. A
binary reward against a strengthening policy is self-extinguishing.

| | value |
|---|---|
| steps with zero gradient | **92 of 100** |
| mean reward across the run | 0.9775 |
| peak allocated | 82.37 GiB |
| adapter | 170.8M params, `runs/8ca073e54686` |
| MMLU before → after | 75.2% → **73.0%** |

**The MMLU move is not read as an effect.** It is n=1, from a run in which 92
of 100 steps trained on nothing; the 8 steps that did carry gradient are not
enough signal for a 2.2-point downstream claim in either direction. What the
number does support is the opposite of a result: an arm this starved should not
be expected to move a downstream metric at all, and it did not.

## Group size is the wrong lever, and sampling has no headroom

Scaling that ratio forward, against ~4 min/step at group 8:

| group | effective | tie rate | step cost |
|---:|---:|---:|---|
| 8 | 3.4 | 0.887 | 4 min |
| 16 | 6.9 | 0.787 | ~8 min |
| 32 | 13.7 | 0.619 | ~16 min |

Quadrupling the group quadruples step time to move ties 0.887 → 0.619, still a
majority dead. That is a tax, not a lever.

Nor is the correlation a sampling misconfiguration. `prompt.py:14` sets
thinking-mode sampling from the model card: `temperature 1.0, top_p 0.95,
top_k 20`. Temperature is already at 1.0 — there is no hotter setting that does
not change the policy being measured.

The rollouts agree because at 96.6% accuracy the answer is not in doubt. No
sampling setting fixes "the model already knows this."

## What this does to the thinking-cap claim

It strengthens the causal reading and corrects the mechanism.

The cap has two functions, and the write-up named only one:

1. **Brevity pressure** — the policy finds the shorter path. Measured: −22.8%.
2. **Difficulty pressure** — it holds per-rollout accuracy at 0.86 instead of
   0.966, which is what keeps groups mixed and GRPO fed. Unmeasured until now.

Function 2 is why the loose arm cannot be a second treatment. Seven gradient
steps in 62 makes `grpo2k` a base-model control **by construction** — which is
the more useful thing to have. Its own before-eval independently reproduced
`448/500 = 89.6%`, the same base number as the 256 run, from a separate process
on a separate card.

Consequence for the step-preservation metric, when it is built: point it at the
2048 adapter **first**. Base and adapter should score the same after 7 gradient
steps. If they differ, the metric is reading noise, and that is worth knowing
before it is aimed at the 256 arm.

## Registered predictions for the judge arm — written before it exists

The data points at a third arm: 2048 cap, `--judge` on (landed default-off in
#64). The judge reorders inside the all-pass subgroup, which is exactly the 55
dead steps, and `cli.py:376` asks it for "clearer steps, no unjustified leaps,
**no wasted work**" — a brevity criterion. If it works, the cap's effect is
recoverable without the cap: the cap gets brevity by truncating mid-derivation
and scoring the truncation wrong, the judge by preferring the shorter correct
path among correct ones. Same direction, one an artifact and one the objective.

**Prediction A (near-certain, proves nothing).** Tie rate collapses from 0.887
toward 0. The judge breaks ties inside the subgroup *by construction*, so this
falls whether or not it is measuring anything real.

**Prediction B (load-bearing).** Total completion tokens fall against the 2048
base arm, with the cap absent from training. **If A holds and B does not, the
judge is manufacturing gradient from noise** — worse than 55 dead steps,
because it looks like progress on every metric currently logged.

This has an implication to act on before any such run: with `--judge` on,
`tied_group_fraction` becomes structurally incapable of being bad, and it is
currently the health metric for a GRPO run. **Total completion tokens must be
logged per step alongside it**, or the run cannot tell its two outcomes apart
afterward.

Prediction B and its inversion are tilerl-48's; recorded here because they
change how the result reads, not merely whether it is confirmed.

## Rule

**A control inherits the treatment's parameter, so check it is not inheriting
the effect.** The 0.86 that produced the failed prediction was measured under
the very cap the control was built to remove. When a control's input is a
number the treatment moved, the control is not independent of the treatment,
and its prediction will be wrong in the treatment's direction.

Second: **a saturating reward is a training-signal failure, not a difficulty
setting.** 88.7% of steps carrying no gradient is the headline, not the 96.6%
accuracy that causes it. Any GRPO run here should report the fraction of steps
that trained, next to the reward.
