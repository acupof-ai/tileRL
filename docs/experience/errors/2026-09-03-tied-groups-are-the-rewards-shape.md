---
question: What actually decides whether a GRPO group ties, and what does a tied-group failure mean?
status: measured
source: tileRL, tiny model, measured 2026-09-03 while attributing the sampler/policy bias
---

# A tied group is about the reward's shape, not the sampler's width

`cli.py:254` fails a run when `tied_group_fraction` exceeds 0.5. The gate exists
because a group whose rewards are all equal yields advantage exactly 0 and no
gradient. The question was whether a narrow sampling nucleus causes that. It does
not. Completion length and reward shape do, and one of them ties **every** group.

## Measured, tiny model, group=8, 12 steps, card sampler throughout

| completion length | reward shape | tied groups | mean reward std |
|---|---|---|---|
| 4 | dense | 12/12 (100%) | 0.0000 |
| 4 | sparse | 12/12 (100%) | 0.0000 |
| 16 | dense | 0/12 | 0.0759 |
| 16 | **sparse** | **12/12 (100%)** | 0.0000 |
| 64 | dense | 0/12 | 0.0507 |
| 64 | **sparse** | **12/12 (100%)** | 0.0000 |

"dense" = fraction of completion tokens satisfying a predicate. "sparse" =
all-or-nothing, 1.0 only if the whole completion is correct.

Same sampler in every row. Two earlier measurements of "the tied-group rate"
disagreed wildly (0/12 and 63%) and neither was about the sampler; they were run
at different lengths and reward shapes. When two measurements of the same
quantity disagree, the quantity was not what was being varied.

## Why sparse ties everything

The tiny model never produces a fully correct completion, so every rollout scores
0, so the group std is exactly 0. This is not a defect. It is what a verifiable
reward does when the policy's pass rate on a prompt is near 0 — or near 1.

**GSM8K is a sparse reward.** `grpo-gsm8k-27b` scores a final answer. So on the
first steps of a cold LoRA against hard prompts, the gate fires — for a reason
that is not a bug, with an exit code indistinguishable from a broken gradient.

`tests/test_rl.py:99` already carries the finding, one line above the test that
depends on it: "Dense reward: an untrained policy's group needs variance for a
gradient at step 0." The test is correct and it never exercises the sparse case
the recipe will hit.

## The discriminator

A tied-group failure has two causes and the exit code cannot tell them apart:

1. the policy cannot solve any prompt in the set yet;
2. the gradient is wrong, so the policy is not improving.

**Run the same prompts with a dense reward.** A policy that solves nothing still
varies token-wise, so cause 1 shows non-zero reward std under a dense reward
while cause 2 stays flat under both.

## Open question for the parity doc

`docs/rl-sota-parity.md` §4 credits us for gating on tied groups where TRL only
logs `frac_reward_zero_std`. That credit is worth re-examining rather than
banking: a gate that fires on the expected first step of a sparse-reward task is
a gate that gets switched off by whoever hits it, and then it is not a gate.
TRL's choice may be the better one for exactly the run we are about to make. Not
proposing a change — recording the argument next to the credit.

## Rule

**Vary the thing you claim to be measuring.** Two conflicting numbers for one
quantity mean the axis under test was not the axis that moved.

And when a run-level gate can fire for a benign reason, write down the
discriminator before the gate fires, not after someone disables it.
