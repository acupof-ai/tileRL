# The target score X = 91.0, and why a bigger effect is the harder one to time — 2026-09-08

**Date:** 2026-09-08
**Machine:** local, arithmetic on 55's curve run. No card.
**Subject:** picking the `time_to_score` target. Replaces a `f = 0.3` default that was an
analogy to GSM8K's shape and is now dead.

## The question

`time_to_score` asks: at which step does the policy first reach score X. Picking X is the
whole measurement — every wall-clock number downstream is the crossing step times s/step.

Inputs, all measured on the live run except where marked:

```
base         87.4%  (437/500, test split, n=500)
step 25      93.2%  (466/500), cumulative train 537.4 s
step 100     93.6%  <- the ANCHOR, from 2026-09-05's run, not this one
ceiling      <= 98.7%  (0/500 at the 2048 eval cap, so truncation no longer binds)
paired SE    1.43 pt  (10.2% discordance measured, not an assumed 5%)
```

## X = 91.0

Two constraints, and they leave a 1.5 pt window:

```
lower  X >= base + 2*SE     = 87.4 + 2.86 = 90.26   (the score difference is resolvable)
upper  X <= step25 - 1*SE   = 93.2 - 1.43 = 91.77   (the crossing lands in the steep region)
window [90.26, 91.77], width 1.51 pt, midpoint 91.02
```

**The lower bound is the obvious one.** Below it, the difference between X and base is
inside the noise and no run can say it was reached.

**The upper bound is the one that is easy to miss, and it points the opposite way.**

## Crossing-step noise is score noise divided by the local slope

```
base -> step 25     0.2320 pt/step
step 25 -> 100      0.0053 pt/step        43.5x flatter
```

The quantity `time_to_score` reports is a *step*, so its error bar is not the score's error
bar:

```
crossing-step noise = score noise / slope at the crossing
  in the steep region   1.43 / 0.2320 = +-6.2 steps
  in the flat region    1.43 / 0.0053 = +-268 steps
```

**So X = 93.0 is the worse choice despite being a larger, more significant effect.** It sits
3.92 SE above base — better resolved as a score — and it lands where the curve is flat, so
the step it names is not a number. Score-resolvable and step-resolvable are two different
properties, and only the first is what a "≥ 2 SE" rule checks.

At X = 91.0 the crossing is **step 15.5 ± 6.2**, i.e. [9, 22] — inside the region already
measured, so it is interpolation, not extrapolation.

## What the f = 0.3 default would have given

`X = base + f × total_gain` with f = 0.3 gives **89.45**, which is **below the 90.26 floor**:
the run could not tell you whether it got there. The f came from GSM8K's curve shape by
analogy, before any point of this curve existed.

**A parameter reasoned from an analogy and a parameter derived from a constraint look
identical until data touches them.** 0.3 was not obviously wrong this morning; it was
unfalsifiable. It died the moment a real SE existed, and it died by 0.8 pt — not a rounding
matter.

## Two nominal figures, flagged rather than smoothed over

**The flat slope substitutes the anchor's endpoint.** `(93.6 − 93.2)/75` uses 2026-09-05's
step 100, not this run's, which has not landed. The conclusion survives the substitution
with room to spare:

| this run's step 100 | flat slope | steep/flat | crossing noise if X were there |
|---:|---:|---:|---:|
| 93.6 (anchor) | 0.0053 | 43.5x | ±268 steps |
| 94.5 | 0.0173 | 13.4x | ±83 |
| 95.0 | 0.0240 | 9.7x | ±60 |
| 96.0 | 0.0373 | 6.2x | ±38 |

Even at 96.0 the flat region is 6x flatter than the steep one and its crossing noise is 6x
worse than X = 91.0's ±6.2. **Recheck when this run's step 100 lands; the ordering does not
depend on it.**

**The ±6.2 is nominal.** The steep slope is an average over two points, base and step 25.
If the real curve bends before step 25 — which it probably does — the crossing is earlier
and the band narrower. That direction favours X = 91.0, and it is unmeasured, so ±6.2 is a
placeholder for a number the step 5/10/15 points will supply.

## Rule

**A threshold on a curve needs the slope where it will be crossed, not only the effect size
at it.** The error bar on the answer is the error bar on the measurement divided by that
slope, so the target belongs on the steep part. Choosing by significance alone picks the
flattest place the significance test still passes.

**An analogy-sourced parameter and a derived one are indistinguishable until data arrives**,
and the analogy tends to look reasonable, because it was fitted to a curve that really
existed — just not this one.
