# P1 GRPO fails on its own default learning rate — 2026-09-03

> Status: fixed in the recipe (`lr=1e-4`). The 100-step confirmation run is in
> flight; the finding below rests on a step-aligned paired comparison, not on
> that run.

## Context

`tilerl train --recipe grpo-gsm8k-27b` ran 100 steps on the 27B and failed its
gate:

```
FAIL reward_first=1 reward_last=0 ce_last=3.028
     secs_per_step_median=75.86 tied_group_fraction=0.97 peak_gib=38.63
```

Steps 1-7 score 0.75 to 1.00 with `ce` climbing 2.13 → 3.63; step 8 is 0.375 at
`ce` 5.25; **steps 9 through 100 are exactly 0.0000, all 92 of them.**

The recipe carried no `lr`, so it inherited the CLI default of **1e-3**
(`cli.py:461`).

## Two wrong readings, and why the second one was testable

The first reading was truncation: `max_new_tokens=256`, and the 27B needs more
than that to finish a GSM8K answer — measured the same day, 170/200 at 512
tokens against 1/16 at 256. **Wrong**: truncation starves the reward from step
1, and this run scores 1.0 for seven steps first.

The second reading was that the reward column is noise. `grpo_loop` draws
`prompts[step % len(prompts)]`, so consecutive steps are different questions
and a within-arm trajectory compares prompts, not policies
([lr-sweep-cannot-attribute](2026-09-03-lr-sweep-cannot-attribute.md)). True,
and it is exactly what makes the *across-arm* comparison sound: the prompt is
`step`-indexed and the seed is `seed + step*group + g`, both identical across
arms, so at a fixed step every arm sees the same question with the same
sampling noise and **the weights are the only difference between them.**

## The paired comparison

The 100-step run's first 20 steps against a 20-step sweep at three lower rates,
same tree, same data, same seed, aligned by step:

| step | lr 1e-3 | lr 1e-4 | lr 1e-5 | lr 1e-6 |
|---:|---:|---:|---:|---:|
| 1-8 | 1.00 … 0.375 | 1.00 … 1.00 | 1.00 … 0.75 | 1.00 … 1.00 |
| 9 | **0.0000** | 0.3750 | 0.1250 | 0.0000 |
| 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 11 | **0.0000** | 0.1250 | 0.1250 | 0.0000 |
| 12 | **0.0000** | 0.1250 | 0.0000 | 0.0000 |
| 13 | **0.0000** | **1.0000** | 0.8750 | **1.0000** |
| 14 | **0.0000** | **1.0000** | 0.7500 | 0.1250 |
| 15 | 0.0000 | 0.0000 | 0.2500 | 0.1250 |
| 17 | **0.0000** | **1.0000** | 0.3750 | 0.0000 |
| 19 | **0.0000** | 0.8750 | 0.7500 | 0.0000 |
| 20 | **0.0000** | **1.0000** | 0.0000 | 0.0000 |

Steps 1-8 agree across all four arms. From step 9 the 1e-3 arm is flat zero for
twelve consecutive steps while the others score.

**Step 13 is the single cleanest discriminator.** Same question, same sampling
noise: 1e-4 and 1e-6 both score 1.0000 and 1e-3 scores 0.0000. A question three
policies answer and one cannot is not a hard question.

Steps 1 and 2 are bit-identical across all three sweep arms, which is the
control that says the harness is doing what it claims: step 1 is
`tied 1.00`, so its advantage is zero and no update happens, and step 2 is
therefore still the base policy in every arm. They diverge from step 3, after
the first non-tied update.

## Why the sweep alone could not find this

That entry is not wrong; it was under-powered against the wrong hypothesis. It
compared 1e-4, 1e-5 and 1e-6 and correctly found them indistinguishable — they
are one cluster
([lr-sweep-cannot-attribute](2026-09-03-lr-sweep-cannot-attribute.md)). The arm
it did not include is **1e-3, the value the recipe actually ships**, and that
one sits outside the cluster. A sweep that omits the shipped value can only ever
conclude that the arms it did run are the same.

## Reproduced the same day, and the reason it cannot recover

A 20-step arm at 1e-3 on the current pod tree, hours after the 100-step run and
on a different card:

```
step  1-7   reward 1.00 .. 1.00   ce 2.13 -> 3.63   tied 1.00 mostly
step  8     reward 0.3750         ce 5.2451         tied 0.00
step  9-20  reward 0.0000 x12     ce 3.32 .. 6.11   tied 1.00 EVERY step
```

Same shape, same step. And the last column is the part the 100-step gate line
only hinted at with `tied_group_fraction=0.97`: **from step 9 on, every group is
tied, on every step.** A tied group has zero advantage, so there is no gradient
at all. The policy is not learning slowly at 1e-3 — it is frozen at a broken
state, and it cannot recover, because scoring 0 on every rollout of a group is
exactly the condition that removes the signal that would fix it.

That also explains why `ce` wanders 3.3-6.1 without trend after step 9 rather
than continuing to climb: nothing is updating the weights.

**The freezing is self-sustaining**, which is why 92 of 100 steps are exactly
0.0 rather than noisy-small:

    zero reward -> every rollout ties -> zero advantage -> no gradient -> zero reward

A trap, not a slow arm. Nothing in the loop can leave that state.

### Two causes, one observable, told apart by the tied *value*

[tied-groups-are-the-rewards-shape](2026-09-03-tied-groups-are-the-rewards-shape.md)
measures the same observable from the other end: GSM8K's reward is
all-or-nothing, so it ties groups on the tiny model at any learning rate
(36/36 group-steps, against 0/12 under a graded reward). Tying is therefore
*expected* here and is not by itself evidence of anything.

What separates the two is the value the group is tied at, and **the log already
carries it.** `group=8` with a batch of 8 gives exactly one group per step, so
`tied` is only ever 0.00 or 1.00 and the printed mean reward *is* that group's
reward whenever `tied` is 1.00:

| line | reading |
|---|---|
| `reward 1.0000  tied 1.00` (step 1) | the reward's shape — every rollout solved it |
| `reward 0.0000  tied 1.00` x12 (steps 9-20) | collapse — every rollout failed |

No new field is needed; the pair of columns is the discriminator, and this
entry and the tied-groups entry describe the same mechanism reached from
opposite directions.

## Fix

`lr=1e-4` in the `grpo-gsm8k-27b` recipe. Not in the CLI default, which also
serves SFT and pretrain — there is no evidence here about those.

`grpo-tiny-smoke` is unchanged: 2 steps on the tiny cell, nowhere near step 9.

## Limits

- 1e-4 is inside the surviving cluster; **it is not established as the best
  value in it.** 1e-4, 1e-5 and 1e-6 remain indistinguishable at n=20 steps.
- The comparison runs on the pod's `/work/tilerl-train`, which predates `micro=1`
  in the recipe on `main`. Internally consistent — both arms are the same tree —
  but the collapse is not independently confirmed on current `main`.
- 20 steps covers the window where 1e-3 dies. It says nothing about whether
  1e-4 survives 100.

## Rule

A sweep over a parameter must include the value that ships. Three arms agreeing
with each other says they are one cluster, not that the parameter does not
matter — the default was never in the room.

When a per-step metric is confounded by a rotating input, the fix is not to
abandon it. Align the arms on the input: the prompt and the seed are both
functions of the step, so at a fixed step the weights are the only difference,
and the comparison the trajectory could not support the column can.
