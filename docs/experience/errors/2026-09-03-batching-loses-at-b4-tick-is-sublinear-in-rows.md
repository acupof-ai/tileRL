# Batching loses at B=4 — the tick is sublinear in *launched* rows, V100 sm70, 2026-09-03

> Status: **measured, one arm pending.** Same harness, only `--batch` varies:
> **B=1 32.4 → B=2 41.8 (1.29×) → B=4 28.5 tok/s (0.88×)**. Four times the batch is a
> **12% throughput loss**, and per-request rate falls 32.4 → 7.1 tok/s (0.22×). The tick
> tracks `launched_rows^0.74`, which holds the held-out middle point to **1.3%** where an
> affine fit is 7.6% off. B=8 (32 useful rows on the same 32 rung) is in flight to test
> whether launched rows is the whole variable.

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

Tick ms is `ms_per_token × tok_per_forward`, both measured columns.

**Batching pays once and then loses.** B=2 is a real 1.29× aggregate win; B=4 gives back
more than that win and lands 12% *below* a single request.

## What it is not

**Not acceptance.** Per-request tok/forward is **2.10 / 2.24 / 2.16** — flat across the
sweep, so the drafts are being accepted at the same rate and the loss is not a shorter
accepted prefix at higher batch.

**Not the harness.** Both endpoints come from the same script in the same loop, and
`measure()` carries the row-count spy that this session added after the B=1 disaster
([entry](../errors/2026-09-03-the-spec-ncols-ab-ran-at-b1.md)): it raises if the widest
pure-decode tick is narrower than `batch × width`, so each arm is verified to have
submitted the rows it claims.

## The scaling law

Fit the two extremes (4 and 32 rows), hold out the middle (8 rows), and compare:

| form | fit | predicts rows=8 | measured | error |
|---|---|---:|---:|---:|
| affine | `30.8 + 8.51·rows` | 99.0 | 107.1 | **7.6%** |
| **power** | **`23.2 · rows^0.74`** | **108.5** | 107.1 | **1.3%** |

Same number of parameters, same fitting points, and the power form is **6× better** at the
held-out point. Fitted the other way — one exponent per adjacent pair — the two ratios
agree independently: **0.72** on 4→8 and **0.75** on 8→32.

So a launched row costs about **0.74 of its own full pass**. Rows are nowhere near free,
and that single exponent reproduces both aggregate throughputs to 1%:

```
aggregate = useful_token_ratio / launched_row_ratio^0.74
  B=1→2:  2.13 / 2^0.74  = 1.28×   (measured 1.29×)
  B=2→4:  1.93 / 4^0.74  = 0.69×   (measured 0.68×)
```

B=4's loss then reads off directly: it launches **4×** the rows of B=1 for **2.1×** the
useful tokens, because half of its 32-row rung is padding.

## What this withdraws

1. **"Batching should approach 4× because a ctx=32 tick is launch-bound"** — committed
   before this run, **wrong**. At 144 launches/token and a tiny KV the tick looked
   launch-dominated, but an added row costs 0.74 of a pass, so rows are the bill.
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

## Rule

**Fit two forms and hold a point out, or the exponent is decoration.** Three points and two
parameters fit almost anything; the affine and power forms differ by 6× at the one point
neither was fitted on, and that comparison is the entire evidence for sublinearity. A curve
quoted without its held-out error is a redescription of the data.

Second: **a per-row cost measured on one rung ladder does not transfer to another.** The
verify-width ms/row numbers are correct for their own sweep and gave a 1.46× prediction
where the answer was 1.21×. Equal row counts reached by widening the chain and by adding
requests are not the same work — different KV locality, different state gathers.

Third: **when a config loses, check whether it is the config or the rounding.** B=4 is the
one point in the sweep whose useful rows are not a rung, and it is the only point that
loses. The mechanism was in `LADDER_WIDTHS`, not in batching.

## Gate

Row-count spy active in `measure()` (raises below `batch × width`); GPU verified idle
before each launch; `timeout` per arm; ncols pinned identically across arms.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | spec d3 @ctx32 ncols=1, B=1 | 32.4 tok/s, tok/fwd 2.10 |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | spec d3 @ctx32 ncols=1, B=2 | **41.8 tok/s (1.290×)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | spec d3 @ctx32 ncols=1, B=4 | **28.5 tok/s (0.880× — loses)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | per-request rate, B=1 → B=4 | 32.4 → **7.1 tok/s (0.22×)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | per-request tok/forward across B | 2.10 / 2.24 / 2.16 — **flat** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **tick vs launched rows** | **`rows^0.74`, held-out error 1.3%** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | affine fit, same held-out point | 7.6% off — **rejected** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | B=4 rung occupancy | **16 useful / 32 launched = 50%** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | B=8 (32 useful, same rung) | **pending — in flight** |
