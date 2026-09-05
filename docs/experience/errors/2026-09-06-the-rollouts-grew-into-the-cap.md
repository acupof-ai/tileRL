# The rollouts grew into the cap that was measured before they started — 2026-09-06

**Date:** 2026-09-06
**Session:** tilerl-25
**Task:** MATH run 2, `grpo-math-27b`, run `0f7006c74ea0`, card 0

> Status: REJECTED. Killed at step 45 of 100, no after-arm. The policy collapsed
> onto the rollout cap at step 41 and stopped producing a gradient.

## Context

Run 1 of this recipe died at step 6 with every group tied at the floor: a 512
rollout cap against a base policy averaging ~1029 tokens, so nothing reached its
`\boxed{}`. #131 added `_refuse_short_rollouts` — the before-arm has already
measured the base policy's mean completion when training starts, so a mean above
0.8x the cap exits with both numbers named. I wrote in the CHANGELOG that this
made the failure **"impossible by construction"**.

Run 2 launched at `max_new_tokens=2048` against that measured 1029, cleared the
guard, and trained. By step 41 it was producing floor ties again.

## Root Cause

**The guard runs once, before step 1. The policy lengthens as it trains.**

| steps | tied | floor ties | mean tok | mean reward |
|---|---:|---:|---:|---:|
| 1–10 | 0.40 | 0 | 890 | 0.887 |
| 11–20 | 0.20 | 0 | 1122 | 0.662 |
| 21–30 | 0.50 | 1 | 1198 | 0.700 |
| 31–40 | 0.50 | 1 | 1277 | 0.625 |
| **41–45** | **0.60** | **3** | **1923** | **0.225** |

The five floor ties are the five longest steps at the time they ran: 26 at 1836,
32 at 1664, **41 at 2048**, 43 at 2012, **44 at 2048**. Two sit exactly on the
cap — eight completions each, all cut before the answer, which is run 1's failure
at a different number.

**The drift is superlinear, and the linear fit understated it badly.** Fitted on
steps 1–40 the slope is 7.74 tok/step, predicting **1312** tokens at step 45.
The actual mean over 41–45 was **1923**. Fitted over all 45 steps the slope reads
15.92, double the first estimate — the sign that a straight line is the wrong
model, not a noisy one.

I published the linear number before the collapse and it was wrong within three
steps. A trend fitted inside a regime does not survive the regime changing.

**Why it lengthens is the part worth carrying forward.** Reward conditioned on
length over the 45 steps:

| | n | mean reward |
|---|---:|---:|
| `tok < 1200` | 21 | **0.893** |
| `tok >= 1200` | 24 | **0.458** |

Short rollouts score nearly twice as well and the policy still moves toward long
ones. A group advantage doing its job would push the other way. The reward is
`boxed_match` — pure correctness, no length term — so nothing in the objective
prefers the shorter of two correct answers, and nothing penalises running to the
cap and scoring zero. That is the mechanism: **GRPO with a correctness-only reward
and no length term lengthens until truncation, then ties at the floor.**

## What the run did establish

- **MATH level 5 does not exhaust the way GSM8K did.** At step 35 — P1's
  comparison point, where GSM8K was 0.87 tied at mean reward 0.975 — this run was
  **0.34 tied** with reward near 0.70. Ties here came from difficulty producing
  all-right and all-wrong groups, not from the task being solved. That was the
  question the run was launched to answer and it is answered.
- **Cost at a 2048 cap: s/step median 229.2** (mean 216.8, range 147.7–271.9),
  45 steps in 2.71 h. Against P1's 56.88 s/step at a 256 cap, 4.0x.
- **The backward is cap-bound, not token-bound.** Median 126.9 s with a
  117.4–177.6 range while `tok` ran 242 to 2048 — a 8.5x spread in tokens against
  1.5x in backward. `train.py:428` pads every row to `prompt + max_new_tokens`
  for a fixed JIT rectangle, so the backward costs the cap. Rollout is the part
  that scales: 20.0 s to 132.7 s.
- **Before-arm: 400/500 = 80.0%**, 514554 tokens, mean 1029.1, 1286.4 per correct.
  On a levels 3-5 file, not level 5
  ([the eval file was not the level it was named](2026-09-05-the-eval-file-was-not-the-level-it-was-named.md)).
- **`eval-before.jsonl` survived a run that never reached `_finish`** — 500 rows
  on disk after a SIGTERM. That is #131's fix working: the row writer used to
  return early unless the run directory existed, and the directory is created
  after both eval arms.

## Fix

Killed at step 45 rather than run to 100. A policy sitting on the cap at reward
0.06 would have spent ~4.5 h more producing 2048-token rollouts with no gradient,
then ~3 h on an after-arm measuring a collapsed policy — a number already known
without spending the card on it, and card 0 is needed for run 3.

In order, none of them done here:

1. **Padding width buckets.** At most log2 kernel sets, one JIT each, same
   `seq_lens` and mask. Without this a larger cap doubles the fixed 127 s.
2. **A periodic drift guard.** Compare the rollout mean against the cap every N
   steps instead of once at launch; the number is already in the step line.
3. **A length term in the reward, or a length-aware advantage.** This is the
   actual cause and the other two only contain it.
4. Then a cap decision, with all three in place.

## All 45 steps

The full curve, because the collapse is only visible as a shape and the
linear fit above is the thing it disproves.

| step | reward | tied | tok | s/step | rollout | backward |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7500 | 0.00 | 242 | 147.7 | 19.984 | 127.605 |
| 2 | 1.0000 | 1.00 | 1361 | 220.4 | 91.885 | 128.393 |
| 3 | 0.7500 | 0.00 | 1368 | 236.8 | 102.654 | 134.026 |
| 4 | 0.8750 | 0.00 | 783 | 183.2 | 59.541 | 123.532 |
| 5 | 0.7500 | 0.00 | 1451 | 239.9 | 112.711 | 127.058 |
| 6 | 0.8750 | 0.00 | 1257 | 244.7 | 106.251 | 138.308 |
| 7 | 1.0000 | 1.00 | 689 | 177.2 | 50.530 | 126.531 |
| 8 | 1.0000 | 1.00 | 448 | 226.6 | 48.911 | 177.552 |
| 9 | 1.0000 | 1.00 | 832 | 181.8 | 60.381 | 121.275 |
| 10 | 0.8750 | 0.00 | 469 | 163.0 | 35.928 | 126.914 |
| 11 | 0.8750 | 0.00 | 1351 | 233.0 | 103.011 | 129.869 |
| 12 | 0.5000 | 0.00 | 1597 | 247.5 | 121.336 | 126.048 |
| 13 | 1.0000 | 1.00 | 565 | 164.9 | 38.345 | 126.441 |
| 14 | 0.2500 | 0.00 | 1709 | 247.9 | 120.862 | 126.944 |
| 15 | 0.6250 | 0.00 | 1159 | 209.0 | 82.142 | 126.754 |
| 16 | 0.8750 | 0.00 | 1192 | 232.7 | 108.528 | 124.065 |
| 17 | 0.3750 | 0.00 | 662 | 171.5 | 48.153 | 123.210 |
| 18 | 1.0000 | 1.00 | 453 | 169.0 | 33.850 | 135.052 |
| 19 | 0.3750 | 0.00 | 1494 | 268.6 | 132.703 | 135.793 |
| 20 | 0.7500 | 0.00 | 1041 | 217.9 | 79.197 | 138.605 |
| 21 | 0.1250 | 0.00 | 2028 | 260.7 | 122.591 | 138.018 |
| 22 | 1.0000 | 1.00 | 1306 | 230.3 | 102.828 | 127.337 |
| 23 | 0.5000 | 0.00 | 1434 | 229.2 | 110.204 | 118.857 |
| 24 | 0.8750 | 0.00 | 1008 | 195.3 | 74.280 | 120.896 |
| 25 | 0.7500 | 0.00 | 1195 | 215.0 | 96.060 | 118.846 |
| 26 **floor** | 0.0000 | 1.00 | 1836 | 271.9 | 122.652 | 149.102 |
| 27 | 0.7500 | 0.00 | 1290 | 208.7 | 89.511 | 119.095 |
| 28 | 1.0000 | 1.00 | 668 | 178.5 | 47.623 | 130.725 |
| 29 | 1.0000 | 1.00 | 774 | 184.7 | 50.382 | 134.248 |
| 30 | 1.0000 | 1.00 | 445 | 149.1 | 31.579 | 117.391 |
| 31 | 1.0000 | 1.00 | 787 | 179.1 | 58.843 | 120.181 |
| 32 **floor** | 0.0000 | 1.00 | 1664 | 249.3 | 118.300 | 130.864 |
| 33 | 0.3750 | 0.00 | 1871 | 244.6 | 121.219 | 123.223 |
| 34 | 0.7500 | 0.00 | 1678 | 241.3 | 120.162 | 120.983 |
| 35 | 0.3750 | 0.00 | 1630 | 251.2 | 114.062 | 137.032 |
| 36 | 1.0000 | 1.00 | 685 | 170.0 | 44.955 | 124.933 |
| 37 | 1.0000 | 1.00 | 1019 | 187.9 | 69.295 | 118.437 |
| 38 | 0.3750 | 0.00 | 1546 | 246.5 | 120.157 | 126.183 |
| 39 | 1.0000 | 1.00 | 415 | 182.0 | 32.783 | 149.078 |
| 40 | 0.3750 | 0.00 | 1478 | 244.2 | 117.250 | 126.809 |
| 41 **floor** | 0.0000 | 1.00 | 2048 | 255.7 | 122.710 | 132.846 |
| 42 | 0.2500 | 0.00 | 1830 | 254.2 | 120.795 | 133.272 |
| 43 **floor** | 0.0000 | 1.00 | 2012 | 252.1 | 119.230 | 132.694 |
| 44 **floor** | 0.0000 | 1.00 | 2048 | 250.9 | 120.371 | 130.465 |
| 45 | 0.7500 | 0.00 | 1676 | 239.1 | 120.640 | 118.318 |

## Rule

**A pre-flight measurement bounds the starting state, not the trajectory.** A
guard that samples a quantity once and then hands control to a process that
changes that quantity has checked an initial condition. Calling that "impossible
by construction" is a claim about every later step that was never measured.

The tell is grammatical. "Impossible by construction" quantifies over all future
states; the check runs at one instant. The honest phrasing names the instant —
"refuses this configuration at launch" — and the gap between those two sentences
is exactly where run 2 failed.

**And a trend fitted inside one regime predicts nothing about the next.** The
1–40 fit missed step 45 by 47%. Extrapolating it into the entry before the run
finished was the same error one level up: a number measured over an interval,
quoted as if it described the whole.

Related: [the eval cap measured itself](2026-09-04-the-eval-cap-measured-itself.md),
and run 1's floor ties. Same family — a length parameter set without measuring
the quantity it bounds. This is the fourth form: measured once, then outgrown.
