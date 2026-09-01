# A one-width warmup made 52.7 tok/s read as 1.3 — 2026-09-01

## Context

With the packed-f16 GEMV in, `--draft --depth 3` served **1.3 tok/s** against a
41.6 ms W=2 replay that predicted ~39. Dense on the same server measured 32.7,
matching its own replay. So the trunk was fine and something in the speculative
path looked 25x broken.

`prof_spec_tick.py` agreed: 79% of the wall in `_draft_step`, 371 ms per depth
step against 4.98 ms in isolation. That reads like a real defect in the draft.

## Root Cause

There was no defect. Timing ticks one by one across a generation:

| tick | seq_len | ms |
|---:|---:|---:|
| 1 | 31 | 2589 |
| 2 | 32 | 906 |
| 3 | 35 | 732 |
| 5+ | 37..317 | **71-78, flat** |

300 tokens in 84 ticks at ~78 ms = 3.57 tok/tick = 45.8 tok/s. The engine was
never slow. The first three ticks are **CUDA graph capture**, and a speculative
run captures one graph PER ACCEPTED-CHAIN WIDTH — so W=1,2,3,4 each pay once.

`bench_b1_decode.py` warms up with a single `one(prompt, args.lo)` call. At
depth 3 that reaches only width 1. The remaining captures then landed inside
the timed `lo` point, inflating it by ~4.2 s. The two-point slope
`(ghi-glo)/(thi-tlo)` divides by a `thi-tlo` that the inflated `tlo` has driven
near zero — and the rate collapses.

The same script already carried a comment about warmup, from the day a missing
one produced 289 tok/s above a 64 tok/s roofline. Same bug, opposite sign: an
unwarmed `lo` can make the rate absurdly high (compile in `lo` only) or absurdly
low (compile dominating `lo`).

`prof_spec_tick.py`'s 371 ms/step was the same three ticks averaged over a
15-tick run.

## Fix

Warm up to `--hi`, not `--lo`, so every chain width captures before timing:

```python
one(prompt, args.hi)   # was args.lo
```

Measured after: **52.7 tok/s at 31 ctx, 35.4 at 1K** (dense 32.7 / 27.4).
100% draft acceptance in serving, 2.95 tok/forward.

## Rule

A warmup is only a warmup for the code paths it actually reaches. Speculative
decode has a graph per chain width, so warming the narrow case warms nothing.
When a two-point slope disagrees with a per-kernel replay by more than ~2x,
suspect the timing points before the system — and time the ticks individually,
which is what settled this in one run after two profilers pointed at the wrong
component.
