# A verify tick costs its RUNG, not its useful rows — sm70, 2026-09-04

> Status: measured at B=1 and B=2. B=4 is `pending-remote` — it OOMed at
> 31.73/31.74 GiB when run in the same process as the other two arms (3 batches x 4
> depths = 12 captured (batch, width) graph pools), and is re-running alone as ds11.
>
> This is Task #40: the depth cost line was fitted with ncols=1 at B=1, and serving
> runs ncols=2 at B=4.

## Context

`LADDER_WIDTHS = (1, 2, 4, 8, 32)` and the ladder rounds **M = rows x width UP** to a
rung, so a tick with 5 useful rows launches the 8-row kernel and 3 rows are padding.
Two earlier entries priced depth against tick time without separating those, which
left the obvious question unanswered: does a padding row cost anything?

It matters because it decides what to fix. If padding is free, an inversion like a
peer's (B=1 spec wins 1.728x, B=8 loses 0.928x) is a kernel problem. If padding costs
full price, it is a *width selection* problem and no kernel work is needed.

## What Worked

**The rung is the unit of cost.** Solved from the B=2 arm without assuming anything
about the draft: depths 2 and 3 both land on rung 8 (M = 6 and 8), so their two tick
times are two equations in verify and draft.

    depth 2, rung 8:  verify + 2 x draft =  96.2 ms   (55 ticks)
    depth 3, rung 8:  verify + 3 x draft = 102.6 ms   (48 ticks)
    =>  draft = 6.40 ms,  verify = 83.40 ms

Against the same rung measured with padding, at B=1 depth 4 (M=5, 3 of 8 rows idle):

| rung 8 launch | useful rows | padding | verify ms |
|---|---:|---:|---:|
| B=1 depth 4 | 5 | 3 | 82.15 |
| B=2 depth 3 | 8 | 0 | 83.40 |

**60% more useful rows for 1.5% more time.** The kernel charges for the rung it
launches. A padding row is not merely wasteful, it is indistinguishable in cost from a
useful one.

**Which makes the rung boundary the only thing that matters for a width default.**
Every depth comparison in this sweep is decided by which side of a boundary the two
widths fall on, not by acceptance:

| B | depth | M | rung | ms/tick | tok/fwd | tok/s | tok/s/row |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **1** | 2 | 2 | 35.04 | 1.79 | **51.0** | 51.0 |
| 1 | 2 | 3 | 4 | 56.39 | 2.19 | 38.8 | 38.8 |
| 1 | 3 (shipped) | 4 | 4 | 61.42 | 2.49 | 40.5 | 40.5 |
| 1 | 4 | 5 | 8 | 99.61 | 2.59 | 26.0 | 26.0 |
| 2 | **1** | 4 | 4 | 51.63 | 3.49 | **67.5** | 33.8 |
| 2 | 2 | 6 | 8 | 94.66 | 4.40 | 46.5 | 23.2 |
| 2 | 3 (shipped) | 8 | 8 | 99.95 | 4.92 | 49.2 | 24.6 |
| 2 | 4 | 10 | 32 | 185.31 | 5.12 | 27.6 | 13.8 |

Depth 1 wins at both batch sizes, and **its margin grows with B**: 1.259x at B=1,
1.371x at B=2. Same mechanism both times — depth 1 is the last width that stays on the
cheap rung (rung 2 vs 4 at B=1; rung 4 vs 8 at B=2).

**The rung-32 step is the cliff, and ncols=2 does not pay for it.** At B=2 depth 4,
M=10 rounds to rung 32 and crosses `_NCOLS_MIN_M = 32` onto the *faster* kernel. It
still costs **1.86x** the rung-8 tick (191.1 vs 102.6 ms) for **1.04x** the tokens —
net **0.559x**. So ncols=2's 1.82x microbench win, which is real, is spent entirely on
carrying 22 padding rows.

**The draft forward is batched too.** 4.80 ms at B=1 (rung 4, depths 3-2) against 6.40
at B=2 (rung 8) — not a per-B constant, so a draft-share number is only valid at the
batch it was measured on.

## Rule

Price a verify width by the **rung its M rounds up to**, never by its useful rows. The
cost function is a staircase whose steps are (1, 2, 4, 8, 32) and whose value is flat
between them, so the only question a width default answers is which step it lands on.
Two consequences:

- **A batch-size inversion is a width-selection bug until proven otherwise.** B*W
  landing just past a rung boundary costs the whole next rung. Fix the width, not the
  kernel.
- **ncols=2's kernel win and the padding it implies are one decision, not two.** The
  gate at M>=32 buys the fast kernel by guaranteeing padding at every M in
  (10..31); measured here, the padding is 1.7x more expensive than the kernel is fast.

Second rule, from how the B=4 arm died: **one process cannot hold an arbitrary
sweep's graph pools.** 12 captured (batch, width) pairs plus the KV pool reached
31.73 of 31.74 GiB. Sweep one axis per process, or the last arm reports OOM instead of
a number.

## What this does NOT license

**No default is flipped by this entry.** Depth 1 winning on wikitext at B=1 and B=2 is
a corpus-specific result — applying a peer's GSM8K acceptance (p=0.922) to these same
tick costs makes depth **3** win at B=1
(`wins/2026-09-04-depth-default-is-wrong-on-text.md`). What transfers is the cost
staircase, not the winner.

**B=4 is the serving default and is not measured here.** At B=4 depth 1 lands on M=8
(rung 8, ncols=1, exactly filled) while depth 3 lands on M=16 (rung 32, ncols=2, half
padding) — two variables move at once, which is exactly the configuration Task #40 was
filed about.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-04 | 5fdbb46 | V100 32GB | cuda sm70 | qwen38-27b | — | 19.6 | 51.0 |
| 2026-09-04 | 5fdbb46 | V100 32GB | cuda sm70 | qwen38-27b | — | 14.8 | 67.5 |

Rows are depth 1 at B=1 and B=2, ctx=1024, wikitext-103 test, 128 new tokens,
temperature 0. Raw: `$HOME/tilerl-logs/ds10.log` (B=1, B=2, and the B=4 OOM),
`ds11.log` (B=4 alone, in flight).
