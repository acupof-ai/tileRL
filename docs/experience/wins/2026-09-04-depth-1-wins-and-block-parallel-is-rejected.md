# Depth 1 wins at ctx=2048, the shipped default is depth 3, and block-parallel drafting is REJECTED with room to spare

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 + draft, B=1, wikitext x3, `--tokens 128`, ctx=2048
**Task:** #71, closing the gating number for #22

## The rule, registered before the data

Written into #71 before the run launched, because #60's failure was a contaminated
sweep producing a confident answer:

> **Clean-row test** — a row counts only if (i) a deeper depth on a dearer-or-equal
> rung is not *faster* per tick, and (ii) the two-term model
> `verify(own rung) + fpt·draft` holds within 1.3x.
> **Flip** the depth default only if, on clean rows, the best deeper depth beats the
> incumbent by **>1.02x** (~3x this harness's measured 0.17–0.74% spread).
> **Inconclusive, not reject**, if the winning row is contaminated.

## Every row is clean, by the widest margin this harness has produced

| depth | W | rungs | ms/tick | tok/fwd | tok/s | model / measured |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 2 | r2 x215 | 37.48 | 1.78 | **47.5** | 1.0018x |
| 2 | 3 | r2 x5, r4 x161 | 57.99 | 2.32 | 40.0 | 1.0002x |
| 3 | 4 | r4 x144 | 64.30 | 2.73 | 42.5 | 1.0008x |
| 4 | 5 | r2 x2, r4 x29, r8 x105 | 96.15 | 2.91 | 30.3 | 1.0017x |

Monotone in tick (37.48 → 57.99 → 64.30 → 96.15), and the two-term model closes to
**0.18% / 0.02% / 0.08% / 0.17%**. #60's contaminated rows read 5.46x and 1.82x on the
same model, so the test discriminates rather than passing everything.

**The decomposition is not circular**, which is what makes these numbers usable. The
harness times the draft directly (`-- timed, not differenced`) and derives verify as
tick minus its *own* rung's draft, so a rung measured at several independent depths is
a free cross-check:

| rung | depths | spread |
|---|---|---:|
| r2 | 30.44 (d1), 30.40 (d2), 30.39 (d4) | **0.16%** |
| r4 | 47.10 (d2), 47.50 (d3), 47.68 (d4) | **1.23%** |
| r8 | 82.70 (d4) | one depth |

Three independent depths agreeing on rung 2 to 0.16% is the strongest internal check
this harness has produced. And `verify r2 = 30.44` against #60's independently measured
**30.58** is 0.46% — a fourth instrument, on a different day.

## Depth 1 wins, and the shipped default is depth 3

```
depth 1  47.5 tok/s
depth 2  40.0 tok/s   0.8424x
depth 3  42.5 tok/s   0.8940x     <- cli.py:496 default
depth 4  30.3 tok/s   0.6373x
```

**Depth 1 is 1.1186x faster than the shipped default**, on clean rows, at 5.6x the
registered 1.02x threshold. Depth 4 is the worst of the four, at 0.64x.

Why depth 2 costs so much for what it adds — the staircase, not a slope:

```
gains  +0.54 tok/fwd (1.78 -> 2.32)
costs  the rung 2->4 verify step 16.70 ms, PLUS one more draft forward 5.91 ms
       tick 37.48 -> 57.99 = 1.5472x
needs  tok/fwd >= 1.78 x 1.5472 = 2.754 to break even.  Got 2.32.
```

The **16.70 ms** rung step is measured *within one depth* here (r4 47.10 − r2 30.40 at
depth 2), so it no longer depends on the cross-depth subtraction #60 used for its
16.11.

Depth 3 recovers some of depth 2's loss because it fills rung 4 exactly — 144 of 144
ticks, no rung-2 remainder to pay the step twice — which is the reasoning
`cli.py:496`'s help text gives. That reasoning is right about the *ladder* and wrong
about the *winner*: filling rung 4 beats straddling it, and not entering rung 4 beats
both.

## #71's premise had the sign backwards, again

#71 was opened because #60 saw depth 4 read 30.5 tok/s at ctx=2048 against 24.7 at
ctx=1024, acceptance rising monotonically, and asked whether **longer context makes
deeper chains pay**. Measured clean, at the same ctx=2048: **it does not.** Depth 1
wins outright, and the acceptance rise is real but too small to buy the rung.

Depth 4 *does* improve with context — that part of #60 holds. It is simply irrelevant
to the default: at ctx=2048 depth 4 reads **30.3 tok/s against depth 1's 47.5**, so a
depth that gets better with context is still 0.64x the best depth *at that context*.
The premise compared one depth across two contexts and read it as a statement about
which depth to ship.

The context term itself, measured at depth 1 where both contexts sit on **one rung**
(r2 x218 and r2 x215) so no staircase argument is needed:

| | ctx 1024 | ctx 2048 | ratio |
|---|---:|---:|---:|
| ms/tick | 36.13 | 37.48 | 1.0374x |
| tok/fwd | 1.75 | 1.78 | 1.0171x |
| **tok/s** | **48.4** | **47.5** | **0.9805x** |

Doubling the context costs 3.7% of the tick and buys 1.7% more acceptance — a **2.0%
net loss**. Longer context does not pay for a wider chain; it does not even pay for
itself.

That ctx=1024 row is also the run's tightest external check: **36.13 against #60's
36.06** on a different day, 0.19% apart, tok/fwd identical at 1.75.

Implied per-token acceptance from tok/fwd is **0.780 at depth 1 and 0.753 at depth 2** —
essentially one draft quality, not a chain that gets better with depth. Acceptance was
never the lever; the rung step is.

This is the second time in two days a #60-derived premise has been refuted with the
sign reversed. Both times the mechanism was the same: a figure taken from a
contaminated row, or from a ratio of two harnesses.

## The draft cost has two terms, and the four depths separate them

Depth 1's draft reads **7.18 ms/forward** against 5.91 / 5.79 / 5.75 at depths 2-4,
which sit within 2.78% of each other. That is not noise and not an anomaly in depth 1 —
it is a **fixed per-call cost divided by a different number of forwards.**
`_time_draft` (`engine.py:970-977`) brackets the whole `_draft.step(rows)` call with
CUDA events **plus a `synchronize()`**, then divides by the forwards inside it, so
whatever the call pays once is amortized over 0.99 forwards at depth 1 and 3.80 at
depth 4. Fitting `elapsed_per_tick = fixed + pf·n` across all four:

```
fixed  1.60 ms per tick        pf  5.28 ms per draft forward
  n=0.99  predicted 6.89 vs measured 7.18   0.960x
  n=1.93            6.11         5.91       1.033x
  n=2.91            5.83         5.79       1.006x
  n=3.80            5.70         5.75       0.991x
```

Four points, one line, worst residual 3.3%. Two numbers rather than one, and the one
that matters for a parallel head is the **marginal** 5.28 ms — the forward itself.

Which makes the #22 reject **stronger**, not weaker:

| draft cost used | share of tick | ceiling if free |
|---|---:|---:|
| as-measured 7.18 (carries the harness's sync) | 19.0% | 1.2340x |
| **marginal 5.28 (the forward)** | **13.9%** | **1.1620x** |

The 1.60 ms fixed term is the harness's own per-tick sync, which the captured graph
path does not pay (`engine.py:223` forces the JIT to finish before capture; a served
tick replays). So the honest ceiling for a production parallel head is **1.162x**, and
the table below — computed at 7.18 — is the *generous* arm.

## Verdict for #22: block-parallel drafting REJECTED

Depth 1 is a single-rung row, so the draft share needs **no cross-depth subtraction** —
the operation that contaminated #60. This is the first uncontaminated draft/verify
split on this arch:

```
tick 37.48 = verify 30.44 + draft 7.18 x 0.99 forwards/tick = 37.55   (0.18%)
draft share = 19.0%
```

The arithmetic #22 asked for:

| | |
|---|---:|
| draft share of the tick | **19.0%** |
| ceiling if the draft were **free** | **1.2340x** |
| parallel head at 1.00x our draft cost | 0.9982x |
| parallel head at 2.00x | 0.8393x |
| parallel head at **4.08x** (DSpark's 1.86B vs our 456M) | **0.6305x** |

Break-even tok/forward a parallel head must beat:

| head cost | needs tok/fwd | have |
|---|---:|---:|
| 1.00x | 1.783 | 1.78 |
| 2.00x | 2.121 | 1.78 |
| 4.08x | **2.823** | 1.78 |

**Even a parallel head that costs exactly what our serial draft costs breaks even at
best**, because the ceiling is 1.234x and the draft is only 19% of the tick. At
DSpark's actual 4.08x parameter count it needs acceptance to rise **1.59x**, and
block-parallel drafting changes how drafts are *issued*, not how often they are
accepted.

Same direction as #30 (tick 88% GPU-bound), but that verdict rested on a differenced
draft cost. This one is timed.

## The cache sub-question is NOT answered, and the log says why

#71 also asked whether #60's 51 recompiles are a disk-cache miss or first-visit cost.
This run shows **0 compiles** — and that is not evidence, because it shows **0
TileLang log lines of any kind**. The INFO logger that produced #60's count is not
enabled in this invocation, so "0 compiles" measures the logging, not the cache.

Recorded rather than reported as a result. What remains true from the code read: of
`(B, S, H, D, NB, Mb)`, five dims are pinned for this harness — `NB` is arithmetic
(`ab_draft_depth.py:266`, `ceil(2176/16)·1 + 8 = 144`, no `mem_get_info`), `Mb == NB`
(`engine.py:203` allocates `_bt` as `[B, num_blocks]`), `B=1`, `H`/`D` model constants
— and prefill is bucketed at `_PREFILL_BUCKET = 64` ("bounded kernel shapes",
`engine.py:59`). That bounds distinct shapes at **~7 against 51 measured**. The bound
is a code read; the 51 is a measurement; the gap is the open question.

## Rule

Time the thing, do not difference it. Every wrong draft-cost number in this task's
history came from subtracting two ticks that sat on different rungs; the one that
cross-checks to 0.13% came from a timer around the draft and a rung-local
subtraction.

And a single-rung row is worth more than a sweep. Depth 1 alone answers #22, because
215 of 215 ticks share one rung and nothing has to be held equal across a staircase.
