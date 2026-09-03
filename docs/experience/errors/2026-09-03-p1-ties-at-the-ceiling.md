# P1 at lr=1e-4 runs all 100 steps and still FAILs: 86 of them carry no gradient

source: H20 GPU 7, tilelang 0.1.13, Qwen3.8-27B NVFP4, `grpo-gsm8k-27b`,
`--lr 1e-4 --eval-mmlu 0 --eval-n 0`, 100 steps, group 8, run
`b9b6e2d950d6`, 2026-09-03T10:38:58Z.

## Context

[The 1e-3 default destroys the policy](2026-09-03-grpo-default-lr-flattens-the-reward.md)
fixed the collapse: at 1e-3 the run scored 0.75-1.00 for seven steps and then
**exactly 0.0000 for all 92 remaining**, `tied_group_fraction=0.97`. This is the
same recipe at the pinned 1e-4.

The collapse is gone. Step 9 — where the 1e-3 arm went flat — scores 0.5000 with
`tied 0.00`, a live gradient. The run reaches step 100.

    reward_first=1  reward_last=1  ce_last=2.507
    secs_per_step_median=60.45  tied_group_fraction=0.86  peak_gib=38.51
    ledger: FAIL

## Root Cause

**The reward is saturated, so the group ties at the ceiling instead of the
floor.** `tied_group_fraction=0.86` is not the 1e-3 trap wearing a different
number:

| | steps | reward on those steps |
|---|---:|---|
| tied (zero advantage, no gradient) | **86** | **80 at 1.0000**, 6 at 0.0000 |
| untied (a gradient) | 14 | mean 0.6071 |

At 1e-3 the 0.97 was 92 consecutive steps at 0.0000 — every rollout failing, and
failing in a way that removed the signal that would have fixed it. Here 80 of
the 86 ties are every rollout **solving** the question. Those are opposite states
with one observable, which is the distinction the 1e-3 entry names and this run
is the other side of it.

The mechanism is the step's shape, not the learning rate. `train.py` submits
`group` rollouts of **one** prompt per step (`prompts[step % len(prompts)]`), so
a step's advantages come from 8 attempts at a single GSM8K question. At this
model's accuracy that question is almost always solved 8/8 or missed 8/8, and
`group_advantages` returns zeros for both. 14 of 100 steps had a question the
policy could half-solve; only those 14 moved a weight.

`reward_first=1` and `reward_last=1` is what the ledger gates on, and it is
reporting this honestly: the metric it measures cannot move when 86% of the
steps produce no gradient and the reward is already at 1.0 on 80 of them.

Cross-entropy spans 1.0148 to 3.9655 across the run and ends at 2.507. It is not
read here as a trend — 14 gradient steps against that spread supports no claim
about direction.

## Fix

Not applied; both candidates are design changes that need their own gate.

- **More than one prompt per step.** A group spanning questions of differing
  difficulty cannot tie at the ceiling on all of them. This is the change the
  `tied_group_fraction` column has been asking for since it was added, and it
  makes the field mean what its name says — with one prompt per step there is
  exactly one group, so `tied` is a boolean and the "fraction" is over steps.
- **A reward with more than two outcomes.** Exact-match on the final number is
  0 or 1 per rollout, so a group of 8 has 9 reachable means and two of them are
  absorbing. Partial credit on the derivation would break ties the current
  reward cannot see.

Neither is worth doing blind. The measurement that picks between them is the
per-question accuracy distribution over the prompt set: if most questions are
solved 8/8, prompt diversity is the lever; if most are 0/8, the reward's
resolution is.

## Rule

A tied group is two different states and the reward's value tells them apart.
Tied at 0.0 is collapse and is terminal. Tied at 1.0 is a saturated reward and
the run is healthy and learning nothing. A gate that reads only
`tied_group_fraction` cannot distinguish them, and 0.97-at-the-floor and
0.86-at-the-ceiling call for opposite fixes.

Fixing the failure a run reports is not the same as fixing the run. 1e-3 was a
real defect and pinning 1e-4 was the right change; it moved the blocker rather
than removing it, and the second blocker was visible in the same column the
whole time.
