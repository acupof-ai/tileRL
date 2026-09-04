# Batching is non-monotone — a padding row costs 3.3× a useful one, V100 sm70, 2026-09-03

> Status: **measured, four batch sizes.** Same harness, only `--batch` varies:
> **B=1 32.4 → B=2 41.8 (1.29×) → B=4 28.5 (0.88×) → B=8 51.6 tok/s (1.59×)**. Four times
> the batch is a **12% throughput loss**; eight times is a 1.59× win. The non-monotonicity
> is the ladder: `LADDER_WIDTHS` has no rung between 8 and 32, so **B=4's 16 useful rows
> launch as 32**. The direct test: at **identical launched rows (32)**, doubling useful rows
> 16 → 32 costs only **12% more tick time**, so the bill is on launched rows with a small
> useful-row term.

## Context

`ncols=2` was measured at **1.498× on the verify path at B=4**
([entry](../wins/2026-09-03-ncols2-is-1.5x-on-the-verify-path.md)), and pricing it against a B=1
number from a *different* run implied "4× batch buys 1.12× aggregate". That comparison
crossed harnesses, which this session had already been burned by twice, so it was not a
result. This is the same-harness version: one script, ctx=32, depth 3, `--tokens 64`,
`TILERL_NCOLS=1` pinned off so the ncols effect cannot confound the batch effect.

Pinning ncols matters here because the sweep walks **three different compiled rungs**.
`LADDER_WIDTHS = (1, 2, 4, 8, 32)` has no rung between 8 and 32, so `_sm70_chunks` rounds
any M in 9..31 *up*:

| B | useful rows (B·W, W=4) | rung → **launched** rows | occupancy |
|---:|---:|---:|---:|
| 1 | 4 | 4 | 100% |
| 2 | 8 | 8 | 100% |
| **4** | **16** | **32** | **50%** |
| 8 | 32 | 32 | 100% |

## Results

`scripts/bench_ctx_decode.py --depth 3 --batch B --max-ctx 32 --tokens 64`, `TILERL_NCOLS=1`.

| B | tok/s | ms/token | tok/forward | launched rows | tick ms | vs B=1 | per-request tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **32.4** | 30.9 | 2.10 | 4 | 64.9 | 1.000× | 32.4 |
| 2 | **41.8** | 23.9 | 4.48 | 8 | 107.1 | **1.290×** | 20.9 |
| 4 | **28.5** | 35.1 | 8.64 | 32 | 303.3 | **0.880×** | 7.1 |
| 8 | **51.6** | 19.4 | 17.52 | 32 | 339.9 | **1.593×** | 6.5 |

Tick ms is `ms_per_token × tok_per_forward`, both measured columns.

**Batching is non-monotone.** B=2 wins 1.29×, B=4 gives it all back and lands 12% *below* a
single request, B=8 recovers to 1.59×. Aggregate throughput is not increasing in batch size.

The B=4 arm was re-run in the B=8 process as a drift control and read **28.5 / 35.1 / 8.64**
— identical to three decimal places across two separate processes and two different
`max_batch` engines, so the B=8 delta is not drift.

## The discriminating test: same launch, twice the work

B=4 and B=8 launch **the same 32 rows**. B=8 fills them with 32 useful rows where B=4 fills
16 and pads the rest, and `tok/forward` confirms the tick really carried them: **17.52 vs
8.64 = 2.03×**, against 2.00 expected.

| | launched | useful | tick ms | tok/forward |
|---|---:|---:|---:|---:|
| B=4 | 32 | 16 | 303.3 | 8.64 |
| B=8 | 32 | **32** | **339.9** | **17.52** |

**Doubling the useful rows inside a fixed 32-row launch costs 12% more tick time**, for
2.03× the tokens — a 1.81× aggregate gain. So the bill is dominated by launched rows, with a
real but small term (**37 ms, 12%**) that scales with useful rows: the per-row state gathers
and KV work that padding rows do not do.

Committed before the run, the band was: ~57 tok/s → launched rows are the whole variable;
~40-45 → both matter; ~28.5 → the fit is void. Measured **51.6**, which is **9.5% below** the
launched-rows-only prediction and above the middle band. The prediction was directionally
right and quantitatively off, and the 12% useful-row term is exactly the miss.

## What it is not

**Not acceptance.** Per-request tok/forward is **2.10 / 2.24 / 2.16 / 2.19** — flat across
the sweep, so the drafts are being accepted at the same rate and B=4's loss is not a shorter
accepted prefix at higher batch.

**Not the harness.** Both endpoints come from the same script in the same loop, and
`measure()` carries the row-count spy that this session added after the B=1 disaster
([entry](../errors/2026-09-03-the-spec-ncols-ab-ran-at-b1.md)): it raises if the widest
pure-decode tick is narrower than `batch × width`, so each arm is verified to have
submitted the rows it claims.

## The scaling law, and its limit

With only B=1/2/4 the tick looked like a clean power law in launched rows. Fit the two
extremes, hold out the middle:

| form | fit | predicts rows=8 | measured | error |
|---|---|---:|---:|---:|
| affine | `30.8 + 8.51·rows` | 99.0 | 107.1 | **7.6%** |
| power | `23.2 · rows^0.74` | 108.5 | 107.1 | **1.3%** |

**B=8 kills the power form.** It has the same 32 launched rows as B=4, so the fit must
predict one tick for both — and they differ by 12%:

```
23.2 · 32^0.74 = 301.5 ms   vs  B=4's 303.3 (0.6% off)  and  B=8's 339.9 (11.3% off)
```

One input, two measured outputs 12% apart: launched rows alone is **not** a function of the
tick. Adding the useful-row term, with its coefficient measured **marginally** off the two
points that share a launch (no fit involved):

```
tick_ms = 25.6 + 7.53 · launched_rows + 2.29 · useful_rows
```

held out at L=8 to **3.3%**, exact at the three fitted points. The linear-in-launched form is
preferred over the power form for one reason: the power form cannot represent two different
ticks at the same row count.

**The marginal costs invert the intuition:**

| | ms |
|---|---:|
| marginal **launched** row (a pure padding row) | **7.53** |
| marginal **useful** row on top of a launched one | **2.29** |

A row that exists only to fill the rung costs **3.3× what the real work on top of it costs**.
Padding is the expensive part; the actual per-request work — state gathers, KV — is cheap once
the row is launched. That is why B=4 (half its rung padding) loses and B=8 (no padding) wins,
and why B=8 beats B=4 by 1.81× while doing 2.03× the work.

Caveat: four points, three parameters, one held-out check at 3.3%. A description that
survived one out-of-sample test, not a mechanism.

## What this withdraws

1. **"Batching should approach 4× because a ctx=32 tick is launch-bound"** — committed
   before this run, **wrong**. At 144 launches/token and a tiny KV the tick looked
   launch-dominated; a marginal launched row costs 7.53 ms, so rows are the bill.
2. **My 1.56× prediction for B=2.** Derived from per-rung verify ms/row (18.29 / 12.47 /
   8.56 for W=2/4/8), giving "the 8 rung is 1.46× cheaper per row". Measured **1.29×**;
   the real per-launched-row gain from 4 to 8 rows is **1.21×**, not 1.46×. The ms/row
   figures came from a B=1 width sweep, so they price *width* rungs and do not transfer to
   *batch* rungs even at equal row counts.
3. **A residual I nearly published.** I subtracted the measured 3 × 5.53 = 16.6 ms of draft
   forwards from the affine fit's 30.8 ms intercept and wrote "14.2 ms of other fixed
   work". That difference is a measured quantity minus **a rejected model's parameter** —
   never sum or subtract a fit parameter against a measurement, because the residual is
   fabricated. There is no measured decomposition of the fixed cost; getting one needs the
   in-graph profiler, not a third curve.
4. **`rows^0.74`, one tick after I committed it.** It held B=1/2/4 to 1.3% and I wrote it
   into a commit message and a CHANGELOG line as the finding. B=8 refuted it the same tick:
   the fit assigns one tick per row count and the two 32-row points differ by 12%. What
   survives is only the qualitative half — the tick is dominated by launched rows.
5. **"A padding row costs 4.1× an extra useful row."** My first version of the ratio divided
   B=4's whole tick by its 32 rows (**9.48 ms**, an average carrying all the fixed cost) and
   compared it to a marginal 2.29. Marginal against average is not a ratio. The correct
   figure from the plane is **7.53 / 2.29 = 3.3×**.

## Rule

**Fit two forms and hold a point out — and the held-out point must be reachable by a
*different* input, not a different value of the same one.** `rows^0.74` passed a held-out
check at 1.3% and was refuted one measurement later, because every point I had varied
launched and useful rows together. B=4 vs B=8 breaks that collinearity by construction: same
launch, twice the work. **A fit tested only along the direction its inputs move together has
not been tested at all** — the design that discriminates is the one where two inputs
disagree.

Second: **a per-row cost measured on one rung ladder does not transfer to another.** The
verify-width ms/row numbers are correct for their own sweep and gave a 1.46× prediction
where the answer was 1.21×. Equal row counts reached by widening the chain and by adding
requests are not the same work — different KV locality, different state gathers.

Third: **when a config loses, check whether it is the config or the rounding.** B=4 is the
one point in the sweep whose useful rows are not a rung, and it is the only point that
loses. The mechanism was in `LADDER_WIDTHS`, not in batching.

Fourth: **compare marginals to marginals.** A total divided by a count is an average and
carries every fixed cost in the tick; setting it against a difference-of-two-measurements
inflates the ratio (here 4.1× against a true 3.3×).

## Gate

Row-count spy active in `measure()` (raises below `batch × width`); GPU verified idle before
each launch; `timeout` per arm; ncols pinned identically across arms. **B=4 re-run inside the
B=8 process as a drift control: 28.5 / 35.1 / 8.64, identical to three decimals across two
processes and two `max_batch` engines.**

Not measured: B=8's context ceiling. It peaked at **31110 of 32768 MiB** at ctx=32 (1.6 GB
spare) and B=4 already OOMs from ctx≥512, so B=8 above ctx=32 is unknown — and this session
has twice guessed an OOM cause wrong, so it stays unknown until swept with `--max-ctx`.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | spec d3 @ctx32 ncols=1, B=1 | 32.4 tok/s, tok/fwd 2.10 |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | spec d3 @ctx32 ncols=1, B=2 | **41.8 tok/s (1.290×)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | spec d3 @ctx32 ncols=1, B=4 | **28.5 tok/s (0.880× — loses)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **spec d3 @ctx32 ncols=1, B=8** | **51.6 tok/s (1.593×)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | B=4 control re-run in B=8's process | 28.5 / 35.1 / 8.64 — **zero drift** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | per-request rate, B=1 → B=8 | 32.4 → **6.5 tok/s (0.20×)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | per-request tok/forward across B | 2.10 / 2.24 / 2.16 / 2.19 — **flat** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **B=8 vs B=4 at equal 32-row launch** | **tick +12% for 2.03× the tokens** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **marginal launched row** | **7.53 ms** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **marginal useful row** | **2.29 ms — padding costs 3.3×** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | `rows^0.74` fit vs B=8 | **11.3% off — refuted** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | plane `25.6 + 7.53L + 2.29U` | held out at L=8 to **3.3%** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | B=4 rung occupancy | **16 useful / 32 launched = 50%** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | B=8 precapture / peak memory | 16 graphs, 155 s / **31110 MiB** |
