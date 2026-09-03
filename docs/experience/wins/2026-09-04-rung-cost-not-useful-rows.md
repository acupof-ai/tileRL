# A verify tick costs its RUNG, not its useful rows — sm70, 2026-09-04

> Status: measured at B=1, B=2 and B=4, on three different rungs. The B=4 arm
> needed its own process — the first attempt OOMed at 31.73/31.74 GiB sharing one
> with the other two (12 captured graph pools), and the second at a hardcoded
> `num_blocks=2048` whose draft mirror cost 3.0 GiB to hold 288 blocks of context.
>
> This is Task #40: the depth cost line was fitted with ncols=1 at B=1, and serving
> runs ncols=2 at B=4. **Answered — B=4 does not reverse the B=1 answer.**

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

**Confirmed on rung 32, the widest and the one serving actually runs.** Three arms
land there from two different batch sizes, and a rung-32 tick also carries the depth's
draft forwards, so those are subtracted first (at each batch's own draft cost — 6.40 ms
at B=2, 11.10 at B=4, since the draft is batched too):

| arm | tick ms | useful rows | draft forwards | verify ms |
|---|---:|---:|---:|---:|
| B=2 depth 4 | 191.1 | 10 | 4 | **165.5** |
| B=4 depth 2 | 190.8 | 12 | 2 | **168.6** |
| B=4 depth 3 | 201.1 | 16 | 3 | **167.8** |

**1.87% spread for 60% more useful rows**, and it does not even order with row count.
Three rungs now say the same thing: rung 8 to 1.5%, rung 32 to 1.87%, and the raw
rung-32 ticks at 10 vs 12 rows to 0.16%.

**The harness's own per-rung derivation is the cleanest version of this.** Bucketing
every tick by its own M, rung-32 verify comes out at **170.03 ms independently at depth
2 (52 ticks) and depth 3 (46 ticks), and 169.47 at depth 4 (47 ticks)** — three depths
with 12, 16 and 20 useful rows, agreeing to **0.33%**. That is the rung thesis stated
without any subtraction across batch sizes.

**And it prices Task #22 at the serving batch, where the arm gets harder.** From the same
rung-32 pair, one draft forward at B=4 is **10.36 ms** (my estimate from depth 1 against
B=2's verify read 11.10, 7.1% high — the amplification again). Drafting is **15%** of a
rung-32 depth-3 tick, so a block-parallel head doing one forward instead of three takes
201.12 → 180.39 ms, a ceiling of **1.115x**, and it must keep **89.7%** of the
autoregressive head's tok/forward to break even. At B=1 that threshold was 82.7%. The
serving batch makes the block-parallel case *worse*, not better, because the verify launch
it cannot shrink is a larger share of the tick.

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
| 4 | **1** | 8 | 8 | 91.43 | 6.71 | **73.4** | 18.3 |
| 4 | 2 | 12 | 32 | 175.45 | 8.17 | 46.5 | 11.6 |
| 4 | 3 (shipped) | 16 | 32 | 174.14 | 8.45 | 48.5 | 12.1 |
| 4 | 4 | 20 | 32 | 187.59 | 8.45 | 45.0 | 11.3 |

Depth 1 wins at every batch size, and **its margin grows monotonically with B**:
**1.259x at B=1, 1.371x at B=2, 1.513x at B=4** (73.4 vs 48.5 tok/s). Same mechanism
each time — depth 1 is the last width that stays on the cheap rung (rung 2 vs 4 at B=1;
4 vs 8 at B=2; 8 vs 32 at B=4). The ncols=2 kernel that turns on at M>=32 does not
reverse it, which is the question Task #40 was filed to answer.

**Inside rung 32, depth 3 is the optimum and depth 4 is strictly worse.** Depths 2, 3
and 4 all land there, so the verify launch cost cannot separate them; acceptance does,
and it saturates: 8.17 → 8.45 → 8.45 tok/fwd. Depth 4's extra position is rejected every
time while its draft forward is still paid, so it costs 7.7% more tick for zero tokens.
The shipped depth 3 is the right choice at B=4 for a reason no earlier number gave.

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

**The width-selection rule that falls out of it, with the correction the depth-4 arm
forced:** pick the rung on cost, then take the deepest chain that fits inside it **only
as far as acceptance keeps paying** — because the verify launch is free inside a rung but
each extra chain position costs a **draft forward, which is outside that launch**. The
B=4 rung-32 arms show both halves:

| B=4 depth | M | verify (draft subtracted) | tick | tok/fwd |
|---:|---:|---:|---:|---:|
| 2 | 12 | 168.6 | 175.45 | 8.17 |
| 3 | 16 | 167.8 | 174.14 | **8.45** |
| 4 | 20 | 166.5 | 187.59 | **8.45** |

Verify is flat across all three (1.26%), so the rung thesis holds. But depth 4 costs
**7.7% more tick for exactly zero extra tokens** — acceptance saturates at 8.45 and the
positions past depth 3 are all rejected, while their draft forwards are still paid for.
So depth 3 is the optimum at B=4, and it is the depth already shipped.

Second rule, from how the B=4 arm died twice: **one process cannot hold an arbitrary
sweep's graph pools, and a round `num_blocks` is not a small number.** 12 captured
(batch, width) pairs plus the KV pool reached 31.73 of 31.74 GiB; and a flat
`num_blocks=2048` cost 3.0 GiB across the trunk pool *and the draft's mirror of it* to
hold 288 blocks of live context, which is what the draft's f32 prefill readout then could
not find 1.88 GiB for. Sweep one axis per process, and size the pool from
`(ctx + tokens) / BLOCK_TOKENS x rows`.

## What this does NOT license

**It does not reopen Task #22 — it strengthens the reject at the batch that matters.**
The B=4 arm gives block-parallel a 1.115x ceiling and an 89.7% acceptance-retention
break-even, against 1.16-1.21x and 82.7% at B=1. DSpark's head is 4.08x our parameter
count against a 2.36x budget, and now it also has to lose less accuracy than before.
a corpus-specific result — applying a peer's GSM8K acceptance (p=0.922) to these same
tick costs makes depth **3** win at B=1
(`wins/2026-09-04-depth-default-is-wrong-on-text.md`). What transfers is the cost
staircase, not the winner.

**B=4 is now measured, and it removes the reason not to flip — but not the reason to
wait.** The worry this entry was filed with was that at B=4 depth 1 lands on M=8 (rung
8, ncols=1, exactly filled) while depth 3 lands on M=16 (rung 32, ncols=2, half
padding), so two variables move at once and the faster kernel might pay for the padding.
It does not: depth 1 reads 1.513x. What still blocks a flip is the corpus, not the
batch — every one of these rows is wikitext, and the GSM8K acceptance a peer measured
puts depth 3 ahead at B=1. A default needs the serving mix.

**Depth 4 at B=4 was measured after this entry's first draft and refuted its
prediction.** The draft said depth 4 lands on the same rung 32 so the within-rung logic
predicts it beats depth 3 on acceptance. It does not — acceptance saturates at 8.45 and
depth 4 reads 45.0 tok/s against depth 3's 48.5. The error was treating a chain position
as free inside a rung; the verify launch is free, the draft forward that produces the
position is not. Corrected above rather than deleted, because the wrong version is the
natural reading of the rung thesis and worth having on record.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-04 | 5fdbb46 | V100 32GB | cuda sm70 | qwen38-27b | — | 19.6 | 51.0 |
| 2026-09-04 | 5fdbb46 | V100 32GB | cuda sm70 | qwen38-27b | — | 14.8 | 67.5 |
| 2026-09-04 | 2218724 | V100 32GB | cuda sm70 | qwen38-27b | — | 13.6 | 73.4 |
| 2026-09-04 | 2218724 | V100 32GB | cuda sm70 | qwen38-27b | — | 20.6 | 48.5 |

Rows are depth 1 at B=1, B=2 and B=4, then the shipped depth 3 at B=4 for the
comparison; ctx=1024, wikitext-103 test, 128 new tokens, temperature 0. Raw:
`$HOME/tilerl-logs/ds10.log` (B=1, B=2, and the first B=4 OOM), `ds11.log` (B=4 depth 1,
then the second OOM at num_blocks=2048), `ds12.log` (B=4 with the pool sized to the run).
