# Drafting is a quarter of a spec tick, not two thirds — V100 (sm70), 2026-09-02

> Status: SUPERSEDES this file's earlier 55-68% figure, and with it the
> 1.58-1.83× verdict on block-parallel drafting. Measured draft forward is
> **5.53 ms, 25% of a depth-3 tick**; the block-parallel ceiling is **1.20×**.
> Filename kept so existing links resolve.

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

## Why the earlier number was so much larger

Not only the staircase. The 15.13 ms draft forward it measured was **real at the
time** — the draft loop then called `.tolist()` on the sampled token and
confidence every depth step, a device sync per step, serializing D forwards that
should queue back to back. That has since been fixed (the loop enqueues the whole
chain on-device, one drain after), and the draft is now 5.53 ms.

So the earlier entry's own recommendation — "check the cheaper lever first, the
draft is launch-bound, recovering capture takes the 55-68% down with no new
architecture" — was correct, and taking it is what collapsed the prize it was
computed against. The architecture change was never needed.

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
