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

**ctx=2048**

| depth | W | rungs | ms/tick | tok/fwd | tok/s | model / measured |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 2 | r2 x215 | 37.48 | 1.78 | **47.5** | 1.0018x |
| 2 | 3 | r2 x5, r4 x161 | 57.99 | 2.32 | 40.0 | 1.0002x |
| 3 | 4 | r4 x144 | 64.30 | 2.73 | 42.5 | 1.0008x |
| 4 | 5 | r2 x2, r4 x29, r8 x105 | 96.15 | 2.91 | 30.3 | 1.0017x |

**ctx=1024**

| depth | W | rungs | ms/tick | tok/fwd | tok/s | model / measured |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 2 | r2 x218 | 36.13 | 1.75 | **48.4** | 1.0000x |
| 2 | 3 | r2 x3, r4 x174 | 56.30 | 2.16 | 38.4 | 0.9999x |
| 3 | 4 | r2 x2, r4 x159 | 61.53 | 2.38 | 38.7 | 1.0003x |
| 4 | 5 | r2 x1, r4 x15, r8 x141 | 99.48 | 2.45 | 24.6 | 0.9993x |

Eight rows, **all clean and all monotone in tick**. The two-term model
`verify(own rung) + fpt·draft` closes to within **0.18%** everywhere and to
**0.00-0.07%** on the whole ctx=1024 arm. #60's contaminated rows read 5.46x and 1.82x
on the same model, so the test discriminates rather than passing everything.

**The decomposition is not circular**, which is what makes these numbers usable. The
harness times the draft directly (`-- timed, not differenced`) and derives verify as
tick minus its *own* rung's draft, so a rung visited by several depths cross-checks for
free — and here every rung was visited by three or four independent depths:

| | rung 2 | rung 4 | rung 8 |
|---|---|---|---|
| ctx 1024 | 29.38 / 29.34 / 29.33 / 29.36 — **0.17%** | 45.43 / 45.48 / 45.55 — **0.26%** | 80.22 |
| ctx 2048 | 30.44 / 30.40 / 30.39 — **0.16%** | 47.10 / 47.50 / 47.68 — **1.23%** | 82.70 |

Four independent depths agreeing on rung 2 to 0.17% is the strongest internal check
this harness has produced. And `verify r2 = 29.38` at ctx=1024 is **byte-identical** to
#60's independently measured 29.38, on a different day.

### The rung step reproduces, and it retroactively validates #60's figure

Measured **rung-locally** — `r4 − r2` inside one depth, never across depths:

```
r4 - r2:  16.09  16.15  16.19  16.70  17.29   (5 independent measurements, mean 16.48)
r8 - r4:  34.67  35.02                        (mean 34.84)
```

#60 derived **16.11 and 34.82** by cross-depth subtraction, the method this entry
avoids. Both land **inside** the rung-local ranges. So #60's step figures were right;
what was wrong there was the draft cost and the contaminated rows, not the staircase.

## Depth 1 wins at both contexts, and the shipped default is depth 3

```
             d1      d2      d3 (shipped)   d4        d1 / d3
ctx 1024   48.4    38.4      38.7         24.6      1.2522x
ctx 2048   47.5    40.0      42.5         30.3      1.1186x
```

**Depth 1 beats the shipped default by 1.2522x at ctx=1024 and 1.1186x at ctx=2048** —
12x and 5.6x the registered 1.02x threshold. Depth 4 is the worst row at both contexts.

Note the direction of the margin: it is **larger at the shorter context**, the opposite
of #71's premise.

### Why, in one number: the step costs 60% of a tick and acceptance pays 23-30%

Depth 1 is the only depth that sits alone on rung 2. Entering rung 4 costs the verify
step **plus** one more draft forward:

| | ctx 1024 | ctx 2048 |
|---|---:|---:|
| rung step + one draft | 16.15 + 5.7 = **21.8 ms** | 16.70 + 5.7 = **22.4 ms** |
| as a share of the depth-1 tick | **60%** | **60%** |
| what acceptance buys | +0.41 tok/fwd (**23%**) | +0.54 tok/fwd (**30%**) |

A 60% cost against a 23-30% gain, at both contexts. Nothing about acceptance closes
that, which is why no context makes a deeper chain pay.

### `cli.py:496`'s help text is right about the ladder and wrong about the winner

Its argument is that depth 3 fills the rung-4 tile exactly while depth 4 spills to
rung 8. Both halves check out — and the effect is small:

```
filling rung 4 vs straddling it (d3 vs d2):   1.008x at ctx1024,  1.062x at ctx2048
NOT ENTERING rung 4 at all     (d1 vs d3):    1.252x            1.119x
```

Filling the rung is worth 0.8-6.2%. Staying off it is worth 12-25%. The help text
optimizes within the wrong branch of the staircase.

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

## The clean-row test I registered has a hole, found on the peer's B=8 row

Test (ii) — `verify(own rung) + fpt·draft` within 1.3x of the tick — reads **1.0026x on
the peer's B=8 depth-1 row**, which is contaminated beyond argument: its rung-2 verify
is **15693 ms on one tick** against my 29-30, and its tok/fwd is 12.51 against a B=1
1.78.

The test passes because the harness *derives* verify as tick minus its own rung's draft.
Verify therefore carries whatever the tick carries, and reassembling `verify + draft` is
near-tautological — it recovers the tick by construction, compile included. It caught
#60's rows only because those pooled the draft **across** rungs, which broke the
identity; a per-rung draft restores it and the test goes blind.

So the eight rows above are clean, but **this test is not what proves it.** What proves
it is the rung cross-check: the same rung measured at three or four independent depths
agreeing to 0.16-0.26%. A compile inside one rung's mean shows up there immediately —
15693 against 29.33 is 535x.

**Replacement, for any future sweep:** a row is admissible when every rung it reports
agrees with that rung measured elsewhere at the **same `(rows, W, ctx)`**, and its
tok/fwd is physically reachable. Drop the two-term identity; it is a restatement of the
harness's own subtraction.

One further correction, mine: I first flagged rung 4 and rung 8 as "split" between my
rows (45-48, 80-83) and the peer's (24, 25). That is **not** contamination. The ladder
rounds `rows × W`, so the same rung is reached from different `(rows, W)` — my rung 4 is
1 row at W=3-5, theirs is 2 rows at W=2 — and attention splits history by `KVSPLIT =
f(s)` on the query width (`registry.py:23`) with a per-row launch over per-row history.
Same rung, different kernel shape, different context. **Rung identity does not make two
verify times comparable across B**; only identical `(rows, W, ctx)` does. Rung 2 is the
exception that makes the peer's compile visible, because there both runs are 1 row at
W=2.

## Checked: the sm90 peer's p0 contamination does not reach these rows

At B=8 the peer measured `groups[0]` — the same prompts the warm pass had just run —
coming out **3.0x slower** than the two unwarmed groups, and concluded that
`measure(groups[0])` at `:291` is one insufficient warm pass, making **p0 structurally
unusable at any `--prompts`**. That would put a contaminated third inside every pooled
mean here, so it needed checking rather than assuming.

It does not hold at B=1. My p0 is the **fastest** group in 6 of 8 rows:

```
                p0      p1      p2    p0/min(p1,p2)
ctx1024 d1   34.87   38.35   35.16      0.9918x
ctx1024 d4   98.79  100.67   98.98      0.9981x
ctx2048 d1   36.18   38.11   38.16      0.9494x
ctx2048 d4   98.64   91.79   98.03      1.0746x   <- worst
```

Worst p0 excess across all eight rows is **1.0746x** against the peer's 3.0x. A compile
moves the *tick*; nothing here does.

What does vary across my groups is **tok/fwd, falling monotonically** — 1.90 → 1.79 →
1.65 at ctx2048 d1, up to 3.43 → 3.10 → 2.19 at d4. That is the passage effect the
harness already documents at `:286` (15.8% between wikitext passages) and the reason it
prints per-group rows at all. Corpus variance moves acceptance with a flat tick;
a compile moves the tick with flat acceptance. The two are distinguishable, and this is
the second.

**Neither verdict moves if p0 is dropped anyway**, which is the test that matters:

```
                 all 3 groups              p1+p2 only
ctx1024   d1 48.3  best=d1  d1/d3 1.2516x  |  46.9  best=d1  1.2468x
ctx2048   d1 47.5  best=d1  d1/d3 1.1199x  |  45.1  best=d1  1.1608x
#22 draft share / ceiling    19.0% 1.2340x  |  18.6% 1.2291x
```

Depth 1 wins on either arm at both contexts, and the #22 ceiling moves by 0.4 points.
The headline figures in this entry pool all three groups, as the harness printed them.

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
history came from subtracting two ticks that sat on different rungs; the ones that
cross-check to 0.16-0.26% came from a timer around the draft and a rung-local
subtraction.

A single-rung row is worth more than a sweep. Depth 1 alone answers #22, because 215 of
215 ticks share one rung and nothing has to be held equal across a staircase.

And a pre-registered rule is not automatically a working rule. Test (ii) here was
registered before the data, honestly applied, and still near-vacuous — it reassembled a
quantity the harness had just decomposed. Registering a rule protects against fitting it
to the answer; it does not protect against the rule having no content. What made these
rows trustworthy was a redundancy the harness happened to provide (one rung, four
depths), not the test I wrote down. When a check passes on everything, find the case it
should fail on before believing it — the peer's contaminated row was that case, and it
scored 1.0026x.
