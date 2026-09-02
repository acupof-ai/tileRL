# The ncols gate left spec decode on, and I said it didn't — V100 sm70, 2026-09-03

> Status: **the code is right, the prose was wrong.** `ncols=2` *is* active on the
> speculative verify path, which I stated three times it was not. Measured there it is
> a **wash (0.988-1.000×, worst 1.2% against a 2% threshold)**, so the gate stays as
> shipped and only the documentation changed — plus the test loop that could not have
> caught this.

## The false claim

[`errors/2026-09-03-ncols2-cost-5-percent-of-decode.md`](2026-09-03-ncols2-cost-5-percent-of-decode.md)
and its commit both said the gate leaves `ncols=2` on for prefill while

> "M=1 decode and a verify tick (M=B·W≤32, which takes the 8 rung) get the 1-column
> kernel."

The parenthesis is false. The sm70 ladder is `LADDER_WIDTHS = (1, 2, 4, 8, 32)` —
**there is no rung between 8 and 32** — so `_sm70_chunks` rounds any M in 9..31 *up*
to 32:

```
M= 8 -> rungs [8]    ncols off
M=12 -> rungs [32]   ncols ON
M=16 -> rungs [32]   ncols ON
M=32 -> rungs [32]   ncols ON
M=40 -> rungs [32,8] ncols on the first chunk only
```

The engine's defaults are `max_batch=4`, `spec_depth=3` → verify width W=4, so a full
verify tick submits **B·W = 16 rows → the 32 rung → `ncols=2` on**. The gate turns it
off for *dense decode only*.

I wrote "the 8 rung" from the phrase I had been using all tick — "top rung = prefill" —
rather than from `_sm70_chunks`, which answers it in one line. Three copies of the same
unchecked sentence: the `backend.py` comment, the error entry, and the CHANGELOG.

## Why it mattered enough to measure

Not because the doc was wrong, but because **spec decode was a third unmeasured path**,
and its row count sits between the two regimes that *were* measured. The mechanism
argument cuts both ways at M=16 and I could not predict the sign:

- **For a win**: 16 rows share one weight stream, so the tick carries more arithmetic
  per byte than M=1 — closer to prefill, where `ncols=2` pays 1.5-1.8×.
- **For a loss**: the grid still halves, onto the same small-N shapes that lost 4.9% at
  M=1, and a verify tick is 88% GPU-bound.

## Results

`bench_ctx_decode.py --draft ... --depth 3`, tok/s. Threshold committed in the script
before the run: within 2% → keep the rung gate; a >2% loss → the gate must key on the
real row count M rather than the compiled rung Mk.

| ctx | nc1 | nc2 | nc2/nc1 |
|---:|---:|---:|---:|
| 32 | 38.0 | 37.8 | 0.995× |
| 512 | 49.4 | 49.4 | 1.000× |
| 1024 | 51.7 | 51.1 | 0.988× |
| 2048 | 44.6 | 44.4 | 0.996× |
| 4096 | 41.3 | 41.2 | 0.998× |

**Worst 1.2%, inside the threshold.** No code change; `_NCOLS_MIN_M` stays a rung
threshold. (nc1 reads 51.7 at 1024 against the recorded 50.8 baseline — 1.8% above, the
right direction and size for this session's other wins, so the harness is sound.)

## What the three paths together say

| path | rows | rung | ncols=2 |
|---|---:|---:|---|
| prefill | 512 | 32 | **1.52-1.60× — win** |
| spec verify | 16 | 32 | 0.995× — wash |
| dense decode | 1 | 1 | 0.951× — loss, gated off |

The gradient is monotone in rows and matches the mechanism: arithmetic per byte rises
with M, so the same kernel goes from costing 4.9% to paying 1.6×. M=16 is the crossover
and lands on neither side — which is why measuring it was worth a tick and predicting it
would not have been.

## Fix

Docs in three places, and the test's coverage. The contract test's rung loop probed
`(1, 2, 4, 8, 32)` — **every ladder-exact width**, and therefore precisely not the
9..31 interval where the rounding surprise lives. It now probes `9, 12, 16, 24, 31`
and asserts they are ON, `40` as `[True, False]`, and `512` all-True. Two negative
controls verified (`_NCOLS_MIN_M` moved to 33 and to 9 both fail).

## Rule

**A test that only samples the exact boundaries cannot find a rounding bug.** The old
loop asserted `gated == (rows == 32)` over the ladder values, which is true and useless:
the interesting inputs are the ones that are *not* rungs, because rounding is what the
ladder does. Choose test inputs that fall between the cases, not on them.

Second: **read the function, not your last sentence about it.** The claim was one
`_sm70_chunks(16)` call away from being checked, in a tick where I had already run that
function for other reasons. Prose about code drifts from code within minutes; every
factual clause in an entry should be traceable to something executed.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | spec d3 @4096, ncols on vs off | 41.2 vs 41.3 (**0.998×**) |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | spec d3, worst point of five | **0.988× @1024 — wash** |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | spec d3 @1024 control vs record | 51.7 vs 50.8 recorded |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | verify rows at default depth 3 | B·W=16 → **rung 32, ncols on** |

Reproduce: `bench_ctx_decode.py --draft $CKPT/model-00018-of-00018.safetensors --depth 3`
under `TILERL_NCOLS=2` and `=1`.
