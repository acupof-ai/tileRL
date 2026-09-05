# The rollouts grew into the cap that was measured before they started — 2026-09-06

**Date:** 2026-09-06
**Session:** tilerl-25
**Task:** MATH run 2, `grpo-math-27b`, run `0f7006c74ea0`, card 0

## Context

Run 1 of this recipe died at step 6 with every group tied at the floor: a 512
rollout cap against a base policy averaging ~1029 tokens, so nothing reached its
`\boxed{}`. #131 added `_refuse_short_rollouts` — the before-arm has already
measured the base policy's mean completion when training starts, so a mean above
0.8x the cap exits with both numbers named. I wrote in the CHANGELOG that this
made the failure **"impossible by construction"**.

Run 2 launched at `max_new_tokens=2048` against a measured 1029, cleared the
guard, and trained. By step 41 it was producing floor ties again.

## Root Cause

**The guard runs once, before step 1. The policy lengthens as it trains.**

Completion length over the first 42 steps, OLS on the per-step mean:

| steps | mean tok | max |
|---|---:|---:|
| 1–10 | 890 | 1451 |
| 11–20 | 1122 | 1709 |
| 21–30 | 1198 | 2028 |
| 31–42 | 1347 | **2048** |

**Slope +11.96 tokens/step, projecting a mean of ~2100 at step 100** — the mean
itself reaching the cap, not a tail of outliers.

The three floor ties are the three longest steps at the time they ran:

| step | tok | reward |
|---|---:|---:|
| 26 | 1836 | 0.0000 |
| 32 | 1664 | 0.0000 |
| **41** | **2048** | **0.0000** |

Step 41 sits exactly on the cap. All eight completions were cut before the
answer, which is the run-1 failure with a different number.

**Why the policy lengthens is the interesting half, and it is not settled.**
Reward conditioned on length, n=42:

| | n | mean reward |
|---|---:|---:|
| `tok < 1200` | 21 | **0.893** |
| `tok >= 1200` | 21 | **0.488** |

Short rollouts score nearly twice as well, and the policy still moves toward
long ones. A group advantage that worked as intended would push the other way.
So this is a question about the advantage or the learning rate as much as about
the cap, and the after-arm plus the full per-step lengths are the data for it.
Recorded as open, not diagnosed.

## Fix

Not yet fixed — the run is being allowed to finish, because the after-arm of the
policy GRPO actually produced (drift included) is the number that turns 100 steps
into a verdict, and a bad number with evidence is a result this project has never
had for a full run.

What lands after it:

1. **A periodic drift guard.** Compare the rollout mean against the cap every N
   steps, not once at launch. The number is already printed in the step line, so
   this costs a comparison.
2. **Padding width buckets first** (`train.py:428` pads every row to
   `prompt + max_new_tokens`, so the backward costs the cap and not the tokens —
   measured flat at ~127 s while `tok` ranged 242 to 2048). Raising the cap
   without that doubles a fixed cost.
3. **Then** a cap decision, with both in place.

The CHANGELOG's "impossible by construction" is corrected in the same commit that
adds this entry.

## Rule

**A pre-flight measurement bounds the starting state, not the trajectory.** A
guard that samples a quantity once and then hands control to a process that
changes that quantity has checked an initial condition, and saying it makes the
failure impossible is a claim about every later step that was never measured.

The tell is grammatical: "impossible by construction" is a statement about all
future states. If the check runs at one instant, the honest phrasing names the
instant — "refuses this configuration at launch" — and the gap between the two
sentences is exactly where run 2 failed.

Related: [the eval cap measured itself](2026-09-04-the-eval-cap-measured-itself.md)
and run 1's floor ties are the same family — a length parameter set without
measuring the quantity it bounds. This is the fourth form: measured once, then
outgrown.
