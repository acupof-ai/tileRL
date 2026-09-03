---
question: The 100-step GRPO run went to zero reward at step 9. Is the default lr=1e-3 the cause?
status: hypothesis not confirmed, and the experiment run to test it could not have confirmed it
source: H20 sm90 GPUs 4/5/6, one lr per card, tilelang 0.1.13, 27B NVFP4, grpo-gsm8k-27b, 20 steps each
---

# Per-step GRPO reward compares prompts, not policies

The completed 100-step run (`lr=1e-3`, the CLI default) had its last nonzero
reward at **step 8** and returned exactly 0.0 for the remaining 92 steps. The
obvious reading is that the policy collapsed under a learning rate 100-1000x
TRL's `1e-6` and AReaL's `6e-6`-`1.7e-5`, which `docs/rl-sota-parity.md` had
already flagged as an unmeasured gap.

A three-point sweep was run to test it: 20 steps each at `1e-6`, `1e-5`,
`1e-4`, one per card.

```
1e-6  1.00 0.63 0.75 0.75 0.50 0.00 1.00 1.00 0.00 0.00 0.00 0.00 1.00 0.13 0.13 0.00 0.00 0.00 0.00 0.00
1e-5  1.00 0.63 1.00 1.00 0.75 0.00 1.00 0.75 0.13 0.00 0.13 0.00 0.88 0.75 0.25 0.00 0.38 0.00 0.75 0.00
1e-4  1.00 0.63 0.75 1.00 0.88 0.00 1.00 1.00 0.38 0.00 0.13 0.13 1.00 1.00 0.00 0.00 1.00 0.25 0.88 1.00
```

**The hypothesis is not confirmed.** All three keep nonzero reward through step
20, including `1e-4`. If a 1e-3 policy collapses by step 9, a 1e-4 policy
showing 1.00 at step 20 does not follow from a simple dose-response.

## Why the experiment could not have answered it

Look down the columns. Step 6 is 0.00 in all three arms. Step 7 is 1.00 in all
three. Step 10 is 0.00 in all three. Steps 1 and 2 are 1.00 and 0.63 in all
three.

`grpo_loop` draws `prompts[step % len(prompts)]`, so every arm sees the same
question at the same step, and **question difficulty dominates the per-step
reward**. The columns agree because the prompt agrees, not because the policies
do. Three curves plotted against a quantity that is mostly a property of the
input cannot separate three policies, and 20 steps is far too few for the
learning signal to emerge from that variance.

The design error is mine and it is not subtle in hindsight: the metric moves
with the input, so it cannot be the axis a policy comparison is read off.

## What would answer it

A held-out eval set, scored every N steps, with the training prompts and the
eval prompts disjoint. That separates policy quality from which question came
up. The per-step training reward stays useful as a progress trace and is not a
comparison instrument.

## What still stands

The 92 consecutive zeros at `lr=1e-3` remain unexplained and remain worth
explaining — nothing here excuses them. `secs_per_step_median=75.86`,
`peak_gib=38.63` and the 100 completed steps stand as the first end-to-end 27B
GRPO run.

## Rule

Before comparing arms, ask what else moves the metric. If the answer includes
the input, the metric is not a comparison instrument no matter how many arms
are run. Run the arms on a fixed evaluation, or hold the input constant across
arms and compare like for like.
