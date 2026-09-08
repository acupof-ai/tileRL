# ISO's 2.7x step win is a 1.28x wall-clock win, and below 2.0x it is a loss

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55

## Context

**The only output of this line of work is a measured `steps_to_score`.** Everything below —
the frame cost C, the break-even, the precision guard — exists so that number can be obtained.
It is worth stating in that order because the conclusion here is entirely a function of one
value the project has never measured: `r`, the ratio of steps Adafactor needs to steps ISO
needs. `design-rl-stack.md:27` reports ~2.7x on a pretrained Qwen3-4B/8B. Nobody has run it on
the 27B.

The project's objective is `time_to_score = steps_to_score × seconds_per_step`. The right factor
is measured at 85.617 s (#273). ISO reduces the left factor and adds a fixed cost, so the
objective is not monotone in the step ratio.

## Root cause

ISO reparameterizes each 2D weight by its own SVD and caches the frames. That cache is a
per-instance dict keyed by `id(p)` with no persistence (`iso.py:52`, written at `:79`, read at
`:66`, no save/load anywhere in the file), so its cost is paid **once per arm, every run** —
not once per project. And it must be computed on the host, because cuSOLVER's f32 SVD misses
ISO's own orthonormality guard by 4.2x at k=5120
([the guard read cuSOLVER, not f32](2026-09-08-the-iso-guard-read-cusolver-not-f32.md)).

Measured over all 546 master matrices in 10 shape classes, host f32:

| shape | n | per s | class s |
|---|---:|---:|---:|
| (5120, 17408) | 64 | 24.90 | **1593.6** |
| (17408, 5120) | 128 | 6.09 | 779.9 |
| (5120, 6144) | 64 | 7.30 | 467.0 |
| (10240, 5120) | 48 | 6.16 | 295.6 |
| (6144, 5120) | 48 | 5.10 | 244.9 |
| (12288, 5120) | 16 | 5.82 | 93.2 |
| (248320, 5120) | 2 | 22.19 | 44.4 |
| 3 smaller classes | 176 | ≤0.68 | 22.4 |

**C = 3541 s = 59.0 min.** Break-even against Adafactor at N steps:

```
C < N × 85.617 × (1 − 1/r)      ⇒      N > C / (85.617 × (1 − 1/r))

r = 2.7 → N > 66      r = 2.0 → N > 83
r = 1.5 → N > 124 ✗   r = 1.3 → N > 179 ✗
```

At P1's 100 steps and r = 2.7: Adafactor 8562 s (2.38 h), ISO 6712 s (1.86 h) — **1.28x**.
A 2.7x step win is a 1.28x time win. At r = 1.5 the same arm is a *loss* despite winning on
steps.

**Two corrections to figures I produced earlier, both against my own interest.** I first
reported C ≈ 2732 s from three classes with counts 3/72/144; the measured counts are 2/64/128,
because the earlier counts came from the pre-`drop_quantized` inventory where **a served-bytes
tensor shares its shape with the master it encodes**, over-counting every class by 1.15x. That
is the third distinct consequence of the same shape-sharing fact today — the first was 20.38 GiB
of resident memory, the second was the drift gate nearly taking the spectrum of fp4 byte
patterns. The corrected C is larger, which makes ISO harder to justify, not easier.

## Fix

P3's exit criterion moves from "ISO needs fewer steps" to "ISO's `time_to_score` beats
Adafactor's, against the measured C = 3541 s", with **r ≥ 2.0 as the threshold inside a
100-step budget**. Recorded by tilerl-27 in the roadmap.

The old criterion would have scored a 1.4x step win as a pass. On the wall clock that arm
loses. This is the third exit criterion today that was written against an intermediate quantity
rather than the one being optimized — the others being P1's `after > before` (looser than its
own spec) and its +5 pt target (unreachable by the test judging it). The general fix is that
every exit criterion must reduce to an inequality in `time_to_score`, and none may stop at a
proxy.

Not done, deliberately: `(5120, 17408)` costs 24.90 s against `(17408, 5120)`'s 6.09 s at the
same k = 5120 — a **4.1x transpose-direction asymmetry** holding 45% of C, so transposing before
the SVD and swapping U/V back is the obvious first cut from 3541 s to under 2000. It is
optimizing a constant on a path that may not be adopted, so it waits behind `steps_to_score`.

## Rule

A speedup in an intermediate quantity is not a speedup. When a mechanism trades a fixed cost
for a rate improvement, the acceptance threshold is a break-even in the objective, and it has
to be computed before the arm runs — otherwise the arm produces a number that reads as a win in
the units it was measured in and a loss in the units that matter.

State the sensitivity with the verdict. Here the entire conclusion flips between r = 2.0 and
r = 1.5, and r is the one value not yet measured, so "ISO is affordable" is a statement about
an assumption, not a result.
