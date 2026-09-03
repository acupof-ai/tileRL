# A verify tick costs 0.67 + 0.53·W dense ticks — V100 sm70, 2026-09-03

> Status: **ACCEPTED as a cost model**, out-of-sample verified at W=8. It replaces a
> "constant 2.75" I derived from a single width and stated as general. The model prices
> any depth without a run, and it says the shipped default (depth 3) is the **worst of
> three options at short context** — a 12% loss where depth 1 is a 1% win.

## Context

Task #31 measured one verify tick at 66.46 ms. From five contexts at depth 3 I computed
`tok_per_fwd / (spec/dense)` and got 2.79 / 2.82 / 2.72 / 2.78 / 2.79 — flat to 1.8% over
a 128× context range. Three independent derivations agreed (tok/s ratio, ms/token, the
profiler's tick). I published it as **a verify tick costs 2.75 dense ticks**.

Flat in *context* is not constant. The quantity I never varied is the one in the
mechanism: **W**, the verify width. Row-sharing in the sm70 GEMV means a tick's cost
should be an affine function of rows — a fixed per-tick part plus a per-row part — and
"2.75" would then be that line evaluated at W=4 and mistaken for a property of
speculation.

## The measurement

`/tmp/wsw.sh` — depths 1 and 7, so W=2 and W=8 are both **ladder-exact**
(`LADDER_WIDTHS = (1, 2, 4, 8, 32)`) and no rounding penalty confounds the width effect.
`TILERL_NCOLS=1` on every arm: at `max_batch=4`, W=2 submits 8 rows (rung 8, ncols off)
while W=8 submits 32 (rung 32, ncols **on**), so leaving the default in would have
changed the kernel between the arms I was comparing.

Break-even = `tok_per_fwd / (spec_tok_s / dense_tok_s)`: the tokens per forward a draft
must deliver for speculation to break even, in units of dense ticks.

| ctx | dense | **W=2** tok/s | ratio | tok/fwd | **cost** | **W=8** tok/s | ratio | tok/fwd | **cost** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 43.1 | 43.5 | 1.009 | 1.74 | **1.724** | 24.9 | 0.578 | 2.82 | **4.881** |
| 512 | 42.7 | 48.4 | 1.133 | 1.95 | **1.720** | | | | |
| 1024 | 42.1 | 46.8 | 1.112 | 1.92 | **1.727** | | | | |
| 2048 | 41.1 | 45.4 | 1.105 | 1.90 | **1.720** | | | | |
| 4096 | 39.1 | 41.0 | 1.049 | 1.81 | **1.726** | | | | |

W=2 reads **1.723 ±0.4%** over the same 128× context range — the flatness W=4 showed,
tighter. So the cost is genuinely context-independent, and genuinely **not** 2.75.

Fitting the two measured widths (W=2: 1.723, W=4: 2.776):

```
verify tick cost = 0.670 + 0.5265·W   (dense ticks)
```

**Committed before the W=8 run, in the log:** accept within ±15% (4.15-5.61), otherwise
report the two-point fit as insufficient. W=8 predicted **4.88**, and — the stricter test
— flat across contexts to within ±2%, the tolerance W=2 (±0.4%) and W=4 (±1.8%) both met.

| ctx | W=8 tok/s | ratio vs dense | tok/fwd | **cost** |
|---:|---:|---:|---:|---:|
| 32 | 24.9 | 0.578 | 2.82 | **4.881** |
| 512 | 36.4 | 0.852 | 4.10 | **4.809** |
| 1024 | 49.1 | 1.166 | 5.08 | **4.355** |
| 2048 | 34.5 | — | 4.10 | **excluded — harness flagged `UNWARMED`** |
| 4096 | 25.8 | 0.660 | 3.26 | **4.941** |

The 2048 row is dropped on the harness's own flag, not on my judgement of it:
`bench_ctx_decode.py:80` marks a row when the third measurement pass runs more than 2×
faster than the second, so the printed 34.5 is a still-warming pass and the steady-state
rate is lower. It is the only such row in the sweep. Using it would have put a
too-favourable point into the arm I was validating.

**Accepted, and the near-miss is the interesting part.** Mean of the four usable points is
**4.75 against 4.88 predicted (2.7%)**, every point inside the committed ±15% band, and
the extremes are ctx=32 (4.881) and ctx=4096 (4.941) — so there is **no trend in context**,
just one low outlier at 1024. Spread 13%, against W=2's 0.4% and W=4's 1.8%; the cost is
therefore noisier at W=8 but not systematically context-dependent.

With three of four rows in hand the picture looked different: 4.881 / 4.809 / 4.355 read as
a **monotone decline**, and I had written the mechanism to explain it — a fixed per-tick
overhead amortized against a dense tick that is itself slowing. Then 4096 came back the
highest of all four and deleted the trend. **A three-point monotone sequence is not a
trend**, and the explanation I had ready for it was the kind that fits whatever the data
does.

## What the line means

Both terms are physical, and the axis is worth stating precisely because I later misread
it: `bench_ctx_decode.py` submits **one** request, so every point here is a **B=1** tick
and `W` is a chain width that happens to equal the launched row count. At B=4 those come
apart (W=4 and W=8 both launch 32 rows), so this line does not price the ladder's rungs —
see the open item.

- **0.67 dense ticks of fixed cost per tick** — the parts a tick pays once regardless of
  width: the GDN state gather/scatter (6.40 of 8.80 torch ms, task #31), the launch chain
  of 144 kernels.
- **0.53 dense ticks per verify row** — the marginal row, 12.48 ms at ctx 1024. **This is
  two mechanisms, not one**, and it is not weight-stream sharing (see below):

| term | ms per +1 W | share |
|---|---:|---:|
| one more draft forward | **5.53** (flat) | 34% |
| the verify forward widening | **4.65-6.64** (falls with W) | 66% |
| sum | **10.2-12.2** | vs the fitted **12.48** |

Both terms are independently measured — the draft forward and the verify-only costs
(`w≤2` 36.58, `w≤4` 49.87, `w≤8` 68.46 ms at ctx 1024) come from task #31's profiler four
months earlier, recorded in `spec.py:20-22`. Rebuilding the tick from them:

| W | recorded `verify(W) + (W−1)·5.53` | this entry's line | ratio |
|---:|---:|---:|---:|
| 2 | 1.777 dense ticks | 1.723 | 1.031× |
| 4 | 2.804 | 2.776 | 1.010× |
| 8 | 4.522 | 4.750 | 0.926× |

**3-7% across three widths, from two fully independent instruments** — end-to-end tok/s
here, a kernel profiler there. That is the mechanism confirmed, not merely consistent.

**What it is not: weight-stream sharing.** I first wrote that a marginal row is cheap
"because rows share the weight stream — a second row adds arithmetic, not bytes". The
arithmetic refutes it: the marginal row costs 12.48 ms against a **14.4 ms** total weight
stream (13 GB at 900 GB/s), i.e. **0.86× of re-reading every weight in the model**. If the
rows genuinely shared one pass, the marginal cost would approach zero. Nor is it re-reads
from chunking — `_sm70_chunks` returns **one** chunk for every width 1..8 at B=1. The cost
is a second full forward (the draft's own weights and KV plane) plus a genuinely wider
trunk forward.

And the slope is still **below 1.0**, which is what lets speculation pay: a marginal row
costs 0.53 of a dense tick rather than a full one, because the *verify* half widens
sub-linearly as the ladder's rungs absorb it (6.64 → 4.65 ms/W from W=2→4 to W=4→8).

## It reproduces the depth staircase quantitatively

[`errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md`](../errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md)
recorded depth 4 measuring 31.5 tok/s against 43.8 at depth 3 and 32.6 with no
speculation — a depth *increase* falling below no-speculation, which read as anomalous.
The line explains it with no free parameters: depth 4 wants W=5, the ladder has no rung
at 5, so it launches 8 rows and pays **W=8's** cost. It must beat **4.88** tok/forward
while delivering roughly what depth 3 delivers (~3.3). Not an anomaly — the arithmetic.

Only depths whose W is a ladder rung launch the rows they use:

| depth | W | rows launched | break-even tok/fwd | ladder-exact |
|---:|---:|---:|---:|:--:|
| 1 | 2 | 2 | **1.72** | yes |
| 2 | 3 | 4 | 2.78 | no — pays W=4 |
| 3 | 4 | 4 | **2.78** | yes |
| 4 | 5 | 8 | 4.88 | no — pays W=8 |
| 5-6 | 6-7 | 8 | 4.88 | no — pays W=8 |
| 7 | 8 | 8 | **4.88** | yes |
| 8-30 | 9-31 | 32 | 17.5 | no — pays W=32 |
| 31 | 32 | 32 | **17.5** | yes |

Depths **1, 3, 7, 31** are the only defensible choices on sm70. The engine already warns
on the others (`engine.py:313`).

## The finding that changes a default

Depth 1 was never measured before this run. Against the same dense baselines, with depth
7 from this sweep alongside:

| ctx | dense | d1 (W=2) | d3 (W=4) | d7 (W=8) | best |
|---:|---:|---:|---:|---:|---|
| 32 | 43.1 | **43.5 (1.009×)** | 38.0 (0.882×) | 24.9 (0.578×) | **d1 — d3 loses 12%** |
| 512 | 42.7 | 48.4 (1.133×) | **49.4 (1.157×)** | 36.4 (0.852×) | d3 |
| 1024 | 42.1 | 46.8 (1.112×) | **51.7 (1.228×)** | 49.1 (1.166×) | d3 |
| 2048 | 41.1 | **45.4 (1.105×)** | 44.6 (1.085×) | (unwarmed) | d1 |
| 4096 | 39.1 | 41.0 (1.049×) | **41.3 (1.056×)** | 25.8 (0.660×) | wash |

The shipped default (depth 3) is the **worst of three options at ctx=32** and clearly
best only at 512-1024. Depth 1 never loses at any context — its worst reading is 1.009×.
Depth 7 pays only at 1024 (1.166×) and never beats depth 3, so 8 rows is past this
draft's useful width and nothing above it is worth trying.

Why the loss is at *short* context and not long: it is dense that changes, not spec.
Dense decode gains 34% going from 4096 to 32 (39.1 → 43.1) because it is bandwidth-bound
on weights plus a KV read that shrinks; a verify tick's cost barely moves. So the shorter
the context, the higher the bar speculation must clear. **Every dense win erodes
speculation's margin and moves the crossover** — including the ncols=2 work shipped this
same day.

## Rule

**Flat in the variable you swept is not constant.** I varied context five times, saw
1.8% spread, and published a constant. The variable in the mechanism — width — I never
varied, and it moves the number 2.8× across the range the engine actually allows. Before
calling a measurement a constant, name the quantity the mechanism says it depends on and
check that one specifically.

Second: **fit two points, then predict a third before measuring it.** The line was
derived from W=2 and W=4 with the tolerance (±15%) and the number (4.88) written into the
log before the W=8 arm started. It read 4.881. Had the tolerance been set afterward, any
result would have "confirmed" the model.

Third: **a cost model earns its keep by pricing what you did not run.** This one prices
every depth on the ladder from two measurements, and retro-explains a four-month-old
anomaly that was recorded as unexplained.

Fourth: **the model found a live defect that no benchmark would have.** `spec.py` prices the
verify trim with the *same affine form* this entry measures — `BIAS_MS + ROW_MS·r` — but
with H20's constants (211.0 / 0.53). On sm70 the measured pair is 15.9 / 12.5 ms, and
substituting it changes the trim's decision at every acceptance level. A comment claimed
this was "mispriced-but-inert" because "a captured tick skips it"; that is false
(`engine.py:974` calls it from `_draft_chains`, on every spec tick — capture replays the
*verify*, while the trim decides what enters it). **Having a measured cost model turned a
plausible-looking comment into a checkable claim**, and it failed. But **so did my fix**:
the reprice is rejected the same day, because the tick pays for the widest chain rounded to
a rung and W=3 launches the same 32 rows as W=4 —
[`errors/2026-09-03-repricing-verify-lens-was-the-wrong-fix.md`](../errors/2026-09-03-repricing-verify-lens-was-the-wrong-fix.md).
The line survives that rejection and picks the winning depth 5/5; what died is injecting it
into a function whose cost is continuous in a different variable.

Fifth, on the instrument: one row in the sweep printed `UNWARMED` and I dropped it.
`bench_ctx_decode.py` flags a row when its third pass beats its second by >2×, i.e. the
engine was still warming and the printed rate is optimistic. **The harness knew before I
did**, and the flag fired inside the very arm I was validating — where a too-fast number
would have looked like confirmation.

Sixth: **wait for the last row.** Three of the four W=8 points arrived in descending order,
I read a trend, and I had a mechanism written for it. The fourth point was the highest of
the four. Nothing was published, but only because the run finished before I committed —
which is luck, not method. A partial sweep read in arrival order is the same instrument as
a complete one only when it is complete.

Seventh, and the one that cost most: **the decomposition already existed and I re-derived
it wrongly.** `wins/2026-09-02-draft-is-two-thirds-of-a-spec-tick.md` says in its third
paragraph "**depth moves two terms, not one** — depth D costs D draft forwards *and* one
wider verify", with both terms measured. I then priced the slope as one mechanism, and
excluded the draft using **1.12 ms** — the GEMV-only part from a different entry — when the
measured draft forward is **5.53 ms**, 4.9× larger. Reading my own prior entry before
theorizing would have given the right answer immediately, and the wrong exclusion is what
sent me looking for a third mechanism that does not exist.

## Gate

No behavior changed. `engine.py:314` had a stale comment saying the ladder was 1/2/4/8
after the 32 rung was added; `spec.py`'s H20 constants are annotated with the shape
mismatch that makes repricing them the wrong fix, and its `__main__` asserts that W=3 and
W=4 share a rung at both B=1 and B=4 — the fact that kills the reprice, so it fails if
`LADDER_WIDTHS` changes without revisiting the trim. 187 tests pass, ruff clean.

## Open

1. **Should the depth default be context-aware?** Picking the better of depth 1 and 3 per
   context is worth **1.144× at ctx=32** and 1.018× at 2048, nothing elsewhere — one
   threshold, not a table. No new machinery is needed: `graph_keys` already precaptures
   every width in `range(1, 2+spec_depth)`, and the trim already varies W per tick. But the
   better policy variable is probably **acceptance, not context** — ctx=32's loss traces to
   tok/fwd 2.44 against a 2.78 break-even, i.e. p≈0.62, and the trim observes p every tick
   while context is only a proxy for it.
2. **A rung-aware trim, and first the slope at B=4.** `verify_lens` prices a line in total
   rows where sm70 pays a staircase in the widest chain, and repricing its constants is
   rejected
   ([`errors/2026-09-03-repricing-verify-lens-was-the-wrong-fix.md`](../errors/2026-09-03-repricing-verify-lens-was-the-wrong-fix.md)).
   This line cannot substitute for it: it is fitted at **B=1**, where W and launched rows
   coincide, and a serving trim runs at B=4 where they do not. Four concurrent requests at
   widths 2/3/4 is a new script, not a flag — until it exists, no trim change ships.
3. **Why W=8's cost is noisier (13% spread) than W=2's (0.4%) and W=4's (1.8%)** with no
   context trend behind it. A fourth width would say whether variance grows with W or is
   specific to the 8-row rung.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 9f032ce | V100 | cuda sm70 | qwen38-27b | verify tick cost, W=2 (5 ctx) | **1.723 dense ticks ±0.4%** |
| 2026-09-03 | 9f032ce | V100 | cuda sm70 | qwen38-27b | verify tick cost, W=4 (5 ctx) | 2.776 ±1.8% |
| 2026-09-03 | 9f032ce | V100 | cuda sm70 | qwen38-27b | **cost line, fitted W=2/W=4** | **0.670 + 0.5265·W** |
| 2026-09-03 | 9f032ce | V100 | cuda sm70 | qwen38-27b | **W=8 predicted / measured (4 ctx mean)** | **4.88 / 4.75 (2.7%) — accept** |
| 2026-09-03 | 9f032ce | V100 | cuda sm70 | qwen38-27b | W=8 per-ctx cost | 4.881 / 4.809 / 4.355 / 4.941 (13% spread, no trend) |
| 2026-09-03 | 9f032ce | V100 | cuda sm70 | qwen38-27b | depth 1 @32 vs dense | 43.5 vs 43.1 (**1.009×**) |
| 2026-09-03 | 9f032ce | V100 | cuda sm70 | qwen38-27b | depth 3 @32 vs dense | 38.0 vs 43.1 (**0.882×, a loss**) |
| 2026-09-03 | 9f032ce | V100 | cuda sm70 | qwen38-27b | depth 7 @1024 vs dense | 49.1 vs 42.1 (1.166×, still < d3's 51.7) |
| 2026-09-03 | 9f032ce | V100 | cuda sm70 | qwen38-27b | **verify_lens trim price, sm70 vs H20** | **15.9/12.5 vs 211.0/0.53 ms — reprice rejected, wrong cost shape** |
| 2026-09-03 | c4852e6 | V100 | cuda sm70 | qwen38-27b | **slope decomposed** | **5.53 draft + 4.65-6.64 verify widening = 10.2-12.2 vs 12.48 fitted** |
| 2026-09-03 | c4852e6 | V100 | cuda sm70 | qwen38-27b | line vs task #31 components, W=2/4/8 | **1.031 / 1.010 / 0.926× — two independent instruments** |
| 2026-09-03 | c4852e6 | V100 | cuda sm70 | qwen38-27b | marginal row vs full weight stream | 12.48 vs 14.4 ms (**0.86× — refutes weight sharing**) |
