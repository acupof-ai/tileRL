# The ncols gate left spec decode on, and I said it didn't — V100 sm70, 2026-09-03

> Status: **the code is right, the prose was wrong — and the measurement below is
> WITHDRAWN.** `ncols=2` *is* active on the speculative verify path in serving, which I
> stated three times it was not; that correction stands. But the "wash (0.988-1.000×)"
> reading did not measure it: the A/B ran `bench_ctx_decode.py`, which submits **one**
> request, so the tick was 4 rows on the 4 rung and `ncols=2` was **off in both arms** —
> [`2026-09-03-the-spec-ncols-ab-ran-at-b1.md`](2026-09-03-the-spec-ncols-ab-ran-at-b1.md).
> The suspiciously flat five-context agreement was the tell. **Re-measured at B=4: it is a
> 1.498x WIN** (42.7 vs 28.5 tok/s, confirm arm 42.8) --
> [`wins/2026-09-03-ncols2-is-1.5x-on-the-verify-path.md`](../wins/2026-09-03-ncols2-is-1.5x-on-the-verify-path.md).
> The gate and the test-loop fix below are unaffected; only the "wash" claim dies.

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

**WITHDRAWN — this table is one kernel measured twice.** The harness submitted a single
request, so the tick ran 4 rows on the 4 rung and `Mk=4 < _NCOLS_MIN_M`: neither arm
compiled the 2-column kernel. Kept here as the evidence, because the *pattern* is the
lesson — five contexts agreeing to 0.5-1.2% is tighter than two different kernels track,
and I read it as a passed threshold instead of as a null result.

| ctx | nc1 | nc2 | nc2/nc1 |
|---:|---:|---:|---:|
| 32 | 38.0 | 37.8 | 0.995× |
| 512 | 49.4 | 49.4 | 1.000× |
| 1024 | 51.7 | 51.1 | 0.988× |
| 2048 | 44.6 | 44.4 | 0.996× |
| 4096 | 41.3 | 41.2 | 0.998× |

The nc1 column is still a valid **depth-3 at B=1** baseline (51.7 at 1024 against a
recorded 50.8, 1.8% above and in the right direction), and it is the column reused for the
depth comparisons in
[`wins/2026-09-03-verify-tick-cost-is-a-line-in-width.md`](../wins/2026-09-03-verify-tick-cost-is-a-line-in-width.md).
Only the ratio is void.

## What the three paths together say

| path | rows | rung | ncols=2 |
|---|---:|---:|---|
| prefill | 512 | 32 | **1.52-1.60× — win** |
| spec verify | 16 | 32 | **1.498x — win** (measured at B=4) |
| dense decode | 1 | 1 | 0.951× — loss, gated off |

All three paths are monotone in real rows and match the mechanism: arithmetic per byte
rises with M, so the same kernel goes from costing 4.9% at M=1 to paying **1.498× at 16
rows** and 1.52-1.60× at M=512. M=16 was exactly the point a prediction was worth least —
which is why it was worth a tick, and why running that tick at the wrong batch size wasted
it. The prediction I could not sign turned out to be a win, and closer to prefill than to
decode.

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
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | spec d3 @4096, ncols on vs off | 41.2 vs 41.3 (**0.998× — void, same kernel**) |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | spec d3, five-context spread | **0.988-1.000× — the tell, not a result** |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | spec d3 @1024 control vs record | 51.7 vs 50.8 recorded (valid, B=1) |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | verify rows in SERVING at depth 3 | B·W=16 → **rung 32, ncols on** |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | verify rows in THIS BENCH | **B=1 → 4 rows → rung 4, ncols off** |

Reproduce the corrected measurement with `--batch 4`; without it the bench runs one
request and never reaches the rung the gate keys on
([`2026-09-03-the-spec-ncols-ab-ran-at-b1.md`](2026-09-03-the-spec-ncols-ab-ran-at-b1.md)).
