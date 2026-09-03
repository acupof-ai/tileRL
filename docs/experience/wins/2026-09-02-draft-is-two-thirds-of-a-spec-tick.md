# Drafting is a quarter of a spec tick, not two thirds — V100 (sm70), 2026-09-02

> Status: SUPERSEDED 2026-09-03 by
> [errors/2026-09-03-block-parallel-drafting-is-1.016x-on-sm70.md](../errors/2026-09-03-block-parallel-drafting-is-1.016x-on-sm70.md),
> third-pass section: **3.93 ms, 19.2% of a depth-3 tick, ceiling 1.147×**. The
> 5.53 / 25.0% / 1.20× below came from each depth's MEAN tick, and `verify_lens`
> trims per tick — so each depth ran a mixture of rungs (depth 2: 15 rung-2 ticks
> and 56 rung-4; depth 3: 14 and 55) and the subtraction moved 16% of all ticks
> across the 16.73 ms rung step. The isolating-pair *method* below is right; it
> just has to be applied to same-rung ticks, not to depth means.
>
> This file's own earlier 55-68% figure, and the 1.58-1.83× verdict that rested on
> it, were superseded by the 5.53 pass. Filename kept so existing links resolve.

## Context

To price a block-parallel draft head (DFlash/DSpark) you need one number: what
fraction of a speculative tick is drafting. A block-parallel head replaces D
sequential draft forwards with one, and changes nothing else, so that fraction
is the entire prize.

This file previously answered 55-68% by regressing ms/tick on depth. That was
wrong, and the error survived because the regression's own bad fit was
rationalized into a range instead of being treated as a refutation.

## Root cause of the earlier number

Depth moves **two** terms, not one. Depth D costs D draft forwards *and* one
verify of width D+1 — and the sm70 GEMV rounds that width up the rung ladder
{1,2,4,8,32}. So a slope fitted through a depth sweep charges verify's growth to
the draft.

Measured at ctx 1024:

| depth | W | rung | ms/tick | tok/fwd | tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 42.11 | 1.92 | 45.7 |
| 2 | 3 | 4 | 60.93 | 2.65 | 43.4 |
| 3 | 4 | 4 | 66.46 | 3.34 | **50.3** |
| 4 | 5 | 8 | 90.58 | 3.74 | 41.2 |

The four-point fit gives a slope of 15.13 ms, and that number decomposes exactly:

    5.53 ms true draft + ~10.6 ms/depth verify growth = 16.1 ~ the 15.13 slope

Restricting the fit to depths 1 and 3 — the earlier "fix" — does not help: W=2
and W=4 are *different rungs*, so that pair mixes the staircase too. Both the 68%
and the 55% are the same error with different arithmetic.

## What Worked

**Depths 2 and 3 are the only isolating pair.** W=3 and W=4 both round to rung 4,
so verify is bit-identical between them and the difference is exactly one draft
forward:

    draft  = 66.46 - 60.93 = 5.53 ms
    verify = 60.93 - 2 x 5.53 = 49.87 ms
    drafting = 3 x 5.53 / 66.46 = 25.0% of a depth-3 tick

**The number is over-determined, which is why it is trustworthy.** Feed that one
draft cost back through every depth and the verify column must come out a clean
staircase — it does, and the two rung-4 depths return the identical 49.87 that
the pair assumed:

| depth | W | rung | verify = ms − D×5.53 |
|---:|---:|---:|---:|
| 1 | 2 | 2 | 36.58 |
| 2 | 3 | 4 | **49.87** |
| 3 | 4 | 4 | **49.87** |
| 4 | 5 | 8 | 68.46 |

Successive rung ratios are 1.363 and 1.373 — consistent, so the draft cost and
the rung model corroborate each other rather than resting on one subtraction.

## What it prices

**Block-parallel drafting: 66.46 → 55.41 ms/tick, 50.3 → 60.3 tok/s at the same
3.34 tok/forward — 1.20×.** Break-even is **2.79 tok/forward**: a parallel head
that drops below that loses to the head we ship.

That is an upper bound, and a loose one. It assumes a parallel head drafts as
accurately as an autoregressive one, but a parallel position cannot see what was
sampled before it, so every point of accuracy lost cuts tok/forward directly
toward the 2.79 break-even. Against that 1.20× ceiling:

- The shipped DSpark head is 5 layers / 1.86B where ours is 1 layer / 456M, so
  its single forward may cost more than the three it replaces.
- agent-infer measured that head at 13% acceptance on Qwen3.8 with no recorded
  cause.

**Verdict: reject.** A 1.20× ceiling that requires matching autoregressive
accuracy with a 4× larger head is not worth the architecture.

## Why the earlier number was larger — the staircase, and nothing else

An earlier version of this entry credited the sync fix: "the 15.13 ms was real at
the time, the loop synced per depth step, that has since been fixed". **That
attribution is wrong, and the data to refute it was already on the page.**

The two depth sweeps — before the sync fix and after — agree to within 0.5%:

| depth | before | after |
|---:|---:|---:|
| 1 | 42.36 | 42.11 |
| 2 | 61.29 | 60.93 |
| 3 | 66.76 | 66.46 |
| 4 | 90.96 | 90.58 |

Had a draft forward really dropped 15.13 → 5.53 ms, three of them would have taken
~29 ms out of the depth-3 tick and it would read ~37 ms. It reads 66.46. So 15.13
was **never a draft cost at all** — it is the slope of a line through the verify
staircase, and the entire 15.13 → 5.53 change is a change of *analysis*, not of
the system. Both sweeps measured the same machine.

The sync fix's own benefit remains **unmeasured**. It is still the right change
(a per-step host sync in a launch-bound loop cannot help), but nothing here
quantifies it, and this entry no longer claims otherwise.

The earlier recommendation — "check the cheaper lever first, the draft is
launch-bound" — was **wrong, and is withdrawn**. It rested on `_draft_step` being
called after `_run_decode_graph` returns, so all D draft forwards run outside the
captured graph, plus a byte roofline: a draft forward streams 954 MB, 1.06 ms at
900 GB/s, against 5.53 measured — "5.2×, where fully-captured dense decode sits at
1.7× of its own floor. Capturing the trunk was worth 2.66×; the same factor gives
5.53 → 2.08 ms and 50.3 → 59.5 tok/s."

Per-kernel attribution refutes it. The depth-3 tick is **88% GPU-bound** (58.5 ms
of GPU-busy against the real 66.46), 71% of that GPU time is the fp4 GEMV, and a
draft forward's 9 GEMV launches are ~1.12 ms of GPU at 125 µs each — i.e. the
byte floor was a factor of 5 below the real one, because the GEMV at M=1 is
launch-shaped, not byte-shaped. There is no host overhead in that gap to reclaim.
Capturing *every* launch caps at 1.14×.

See `errors/2026-09-02-capturing-the-draft-is-rejected.md`. Where the tick
actually is: one kernel, 41.49 ms, 125 µs per launch.

## Rule

**A depth sweep does not isolate a per-depth cost when depth moves two terms.**
Find the pair where every other term is held constant — here, two depths that
land on the same rung — and subtract. One clean subtraction beats a four-point
fit whose residuals you have to explain.

Second, and this is the one that cost the most: **a bad fit is a refutation, not
a range.** The structured residuals were visible and were reported honestly as
"55-68%", which reads like conservatism but is really two wrong numbers averaged.
Nothing about a staircase becomes more true when you widen the error bars on a
line through it.

Third: **feed a derived quantity back through the data it did not come from.**
5.53 ms was derived from depths 2 and 3; verifying it reproduces a monotone rung
staircase at depths 1 and 4 is what turns a subtraction into a measurement.

## Results

| date | commit | machine | target | model | ctx | draft ms | draft share | tok/s |
|---|---|---|---|---|---:|---:|---:|---:|
| 2026-09-02 | 2d8749c | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 1024 | **5.53** | **25%** | 50.3 @ d3 |
| 2026-09-02 | (earlier, wrong) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 1024 | 15.13 | 55-68% | 50.1 @ d3 |

Raw artifact: `scripts/ab_draft_depth.py`. Its cross-check now asserts the two
non-pair depths deviate in OPPOSITE directions (depth 1 below the rung-4 line
because rung 2 is cheaper, depth 4 above it because rung 8 is dearer) — the
earlier version wanted both above, so it flagged a correct result as suspect.
