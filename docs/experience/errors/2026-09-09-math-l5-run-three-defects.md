# The MATH-L5 run exposed three defects in its own instrumentation — 2026-09-09

**Status:** run `d6447c0abe5b` (MATH Level 5, cap 6144/6144, λ=0.1, patience=1
raw, eval-curve-n=100) was stopped at step 10 by coordinator decision — not
because it failed, but because continuing would produce an unexplainable
number. The run's real output is three instrumentation defects, each of which
made a green signal or a stop step carry less information than it appeared to.

## Context

The run scored base 85/100 (curve subset), step 5 = 79/100, step 10 = 80/100.
Patience=1 raw reset on the +1 from step 5 to step 10, so the run would have
continued to step 15. McNemar on the paired subset: step 5 (r2w=9, w2r=3,
p=0.146), step 10 (r2w=7, w2r=2, p=0.180) — neither is distinguishable from
noise. The 2×2 at step 10: kept 78, lost 7, fixed 2, untouched 13; gross flips
9/100. The net −5 and the gross 9% sit inside the cross-batch floor measured
on GSM8K (52 flips / 10.4% gross, net 2; run `76a17ea6e10a`).

## Defect 1: the validity gate reads a field that is structurally zero

The P1 validity gate checks `tied < 0.50`. `tied` (train.py:570) is the
fraction of groups whose 8 rewards are *exactly* equal. The length-aware
reward is `correctness − λ·tokens/cap` — a continuous value. Eight rollouts
almost never produce identical token counts, so `tied` is 0.00 at every step
by construction. The gate cannot turn red.

**Root cause:** the gate was designed for λ=0 (binary rewards, where ties are
common and informative). At λ=0.1 the length term makes exact ties impossible,
but the gate still reads the λ=0 field.

**Fix:** compute `tied_correctness` — the fraction of groups whose binary
correctness is all-same (all right or all wrong) — before λ, tiebreak, and the
live mask are applied. This is the quantity comparable to the roadmap's 0.650
(which was λ=0, cap 2048, 32% truncation). Measured on this run: 6/10 = 0.60.

**Rule:** a gate's field must be able to turn red under the condition it
guards. Verify the field can change before trusting the gate.

## Defect 2: the stop rule's trigger is smaller than the instrument's noise

Patience=1 raw resets on any strict improvement. Step 5→10 was +1.0 pt
(79.0→80.0), and McNemar on that pair gives p=0.180 — the +1 is noise. On an
n=100 subset with ±3-5 pt noise, a rule that resets on +1 is a random-walk
stop: when it stops depends on which direction the noise happens to point, not
on whether the policy learned anything.

**Root cause:** the stop rule was carried over from a step-function curve
(GSM8K: +6.6 pt at step 5, then flat) where a +1 reset is harmless because the
signal is 6× the noise. MATH L5 at n=100 has no such signal-to-noise ratio.

**Fix:** smooth the metric (3-point moving average) before applying patience,
or raise patience to ≥3, or gate on significance. Do not use raw-score
patience=1 on a subset this noisy.

**Rule:** a stop rule's trigger threshold must exceed the metric's noise.
Measure the noise before setting the threshold.

## Defect 3: the base comparison's churn floor is larger than the effect

The before-arm scores the eval file in order (500 rows, file order, batched
alongside MMLU). The curve eval scores a shuffled 100-row subset alone. These
are different batch compositions, and the cross-batch floor is 52 flips /
10.4% gross (net 2) on the same weights. The run's −5/−6 pt net changes sit
inside that floor: the gross churn (9-12%) is the floor, and the net is
uninterpretable without a paired test.

**Root cause:** the base and the curve points were never measured in the same
batch. The net score difference conflates policy movement with batch-composition
non-determinism.

**Fix:** re-measure base in the same batch as the after-arm (same weights,
same order, same concurrency — the same-batch instrument, which measured 0
flips on run `76a17ea6e10a`), or use McNemar on the paired subset instead of
comparing net scores.

**Rule:** a cross-batch score difference needs a churn floor before it can be
read as an effect. The net change and its composition can point in opposite
directions — only the net gets quoted.

## What the run did produce

The bucket-width–seconds table is the one output that feeds the ruler
directly:

| width | steps | secs |
|---|---|---|
| 512 | 1, 8, 10 | 31.7, 90.1, 37.1 |
| 2048 | 4, 7, 9 | 113.8, 99.9, 111.2 |
| 4096 | 2, 3 | 271.5, 244.8 |
| 6144 | 5, 6 | 409.2, 416.4 |

Cost follows actual completion lengths, not the cap. The 6144 bucket ran 2
steps without OOM. This overturns the "6144 cap = 3× cost" extrapolation —
the training rectangle is the next power-of-two of the group's longest
completion, clamped to the cap, so a 6144 cap only costs 3× when the
completions actually fill it.

## Token reduction without accuracy gain

The run cut mean tokens 1720 → 1253 (−27%) while the score fell 85 → 80 (−5).
The README's thinking-cap result (GSM8K, cap 256, λ=0) reports +5.2 pts with
−22.8% tokens. The token reduction reproduced; the accuracy direction reversed.

The two runs differ in four ways — task (GSM8K vs MATH L5), cap (256 vs 6144),
reward (correctness-only vs length-aware λ=0.1), and measurement (trained
tight / measured uncapped vs trained and measured at 6144) — so this is not a
controlled contradiction. But it is a data point where token economy arrived
without accuracy gain, which the README's "buys accuracy *and* economy" framing
does not predict. The two may be separable: the length penalty buys the token
cut on its own, and the accuracy gain in the GSM8K run may come from the tight
cap's gradient effect (the control arm at 2048 reached 96.4% anyway — README
L71-77), not from the token cut itself. This run cannot settle that; it adds
one observation to the question.

## See also

- [The collapse did not replicate; the plateau did](2026-09-09-the-collapse-did-not-replicate-the-plateau-did.md)
  — the cross-batch floor calibration and the patience-rule pricing this entry
  builds on.
