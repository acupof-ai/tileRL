# Speculative decoding — REJECTED: measured against the eager path, not the shipped one

> Status: Rejected on throughput, twice, the second time after the M-row GEMV
> made the verify cheaper. The engine keeps the code (correct, gated, and off
> unless a draft is passed). The blocker is the width-2 verify tick, not the
> draft: at B=1 it costs 1.97x a width-1 tick and returns 1.12 tokens, so a
> free draft still loses.

## Context

Speculation landed in the engine: a decode row drafts off the trunk's last
hidden and the same forward verifies it as a `seq_q = 1+depth` row. Goodput is
real — **1.87 committed tokens per trunk forward at depth 2, 43-47% acceptance**
— and this entry first reported it as a 1.14-1.19x win.

That was wrong. The comparison was against the EAGER decode path, and tileRL
ships a CUDA-graph decode. `Engine.__init__` disables graph capture whenever a
draft is present (`self._decode_graph_on = decode_graph and draft is None`),
because a captured graph replays one T=1 step through the fused `gdn_decode`
kernel, which does not exist at T>1.

Measured, same script, same session (H20, GPU 7, 64 layers, 15 timed ticks):

| arm | B=1 ms/tick | B=1 tok/s | B=8 ms/tick | B=8 aggregate |
|---|---:|---:|---:|---:|
| eager, no draft | 91.6 | 10.9 | 138.1 | 57.9 |
| **graph, no draft (shipped)** | **11.6** | **86.2** | **60.0** | **133.3** |
| speculation, depth 2 (forced eager) | 106.2 | 17.6 | 151.0 | 92.3 |

**Against what ships, speculation is 4.9x slower at B=1 and 1.44x slower at
B=8.** The graph is worth 7.9x on its own at B=1 (785 launches collapsed into
one replay); 1.87 tokens per forward does not buy that back.

## What Was Actually Established

- The draft head works: teacher-forced top-1 agreement 84.4%, 43-47%
  acceptance in the loop, 1.87 tok/tick at depth 2.
- The verify path is correct — a mid-sequence multi-token forward matches T=1
  greedy exactly, and the block loop reproduces greedy with an always-wrong and
  an always-right draft.
- Serving the head as fp8 is worth ~75x per projection against the generic
  dense path (9.7 ms -> 0.13 ms for the same shape).

None of that is retracted. Only the throughput verdict is.

## Capturing the spec tick: done, still not enough

`_DecodeGraph` now takes a width — `[B,W]` id/position buffers, `seq_q_lens`
statically W, `keep_steps=W` so the verify's per-step recurrent state writes
land in the pool's static buffers and capture like any other kernel. Uniform W
is the price: `verify_lens`' per-row trim cannot survive capture, so rows pad
to the widest chain (any pad token is correct — a draft is accepted only when
it equals what the trunk sampled).

It works and it is worth 3x on the speculative tick: **106.2 ms -> 35.4 ms** at
B=1 depth 2. It is still a loss everywhere. Both arms below are measured by the
same suite, settled the same way — the harness's spec suite now runs its own
no-draft arm, because three earlier readings of this feature compared against a
row from a different script (eager instead of graph, B=8 instead of B=1,
un-settled instead of settled) and each time the mistake flattered speculation.

| B | depth | plain tok/s | spec tok/s | ratio | accept |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 92.3 | 39.5 | 0.43x | 71.8% |
| 1 | 2 | 92.3 | 53.7 | 0.58x | 47.2% |
| 1 | 4 | 92.0 | 56.9 | 0.62x | 29.6% |
| 8 | 1 | 310.8 | 38.3 | 0.12x | 59.5% |
| 8 | 2 | 311.6 | 149.5 | 0.48x | 42.9% |
| 8 | 4 | 311.9 | **238.5** | **0.76x** | 26.2% |

The ratio improves with both depth and batch and is best at B=8 depth 4, still
0.76x. The B=8 depth-1 row (219.9 ms/tick) is out of family with its
neighbours and is not explained.

The verify replay is ~13 ms; the two eager draft steps are the other ~22 ms,
about 11 ms each for a ONE-layer head whose projections measure ~0.13 ms and
whose lm_head measures 0.52 ms. That gap is not in the kernels and it is not
the host syncs either — removing one of the two D2H copies per draft step moved
52.7 to 52.8, i.e. nothing.

Five hypotheses about this cost have now been measured and refuted (PCIe
migration, weight dtype, weight format, the dict rebinding, the draft loop's
syncs). Stopping here: the code is correct and gated, a draft is opt-in, and
the remaining work is to find where ~10 ms per draft step actually goes.

## Settled: even a free draft loses (2026-08-29, second pass)

The M-row GEMV cut the verify's linear cost, which moved the earlier ratios up
enough to be worth re-deriving. I predicted depth-1 at B=1 would turn positive
(1.06x) from three measured pieces: a 17.54 ms W=2 verify replay, a draft step
captured at 0.67 ms (2.59x off eager, bit-identical), and 71.8% acceptance.

**Measured 0.57x.** The prediction was wrong the same way this entry's original
claim was wrong: the 17.54 ms came from `profile_verify_replay`, which builds
the graph with `keep=0`. A real verify tick uses `keep=W` — it writes the
recurrent state after every chain step — and costs 21.17 ms at B=1, not 17.54.
Adding numbers from two configurations is the exact error this file exists to
record, and it produced a 1.9x optimistic estimate.

Same process, both arms, `bench_harness --suite spec`:

| B | depth | plain | spec | ratio | accept | tok/tick | ms/tick |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 92.9 | 52.7 | **0.57x** | 55.8% | 1.12 | 21.17 |
| 1 | 2 | 92.7 | 59.9 | **0.65x** | 44.7% | 1.77 | 29.47 |
| 8 | 1 | 311.6 | 46.1 | **0.15x** | 58.8% | 1.05 | 181.30 |
| 8 | 2 | 311.1 | 69.9 | **0.22x** | 41.9% | 1.67 | 190.89 |

Depth 2 beats depth 1 and both lose, which is the same shape as the first
pass: the verify tick's growth with width outruns the tokens it buys. At B=1
a width-3 tick is 29.47 ms for 1.77 tokens against 10.78 ms for 1.

The verdict no longer depends on the draft at all. At B=1 a chain of TWO costs
21.17 ms against 10.76 for a chain of one — **1.97x the tick for 1.12 tokens**.
Set the draft's cost to zero and it is still 0.63x. The captured draft step was
reverted: it is a real 2.59x on the draft half, and the draft half is not what
decides this.

Open, and not chased: `tok/tick` is 1.12 at 55.8% acceptance where depth 1
should give ~1.56, and the discrepancy is depth-1-only (depth 2 reads 1.77
against ~1.89 implied, which is close). Either the acceptance counter or the
commit disagrees with `tokens_generated` on short chains. It does not change
the verdict (1.56 tokens in 21.17 ms is 73.6 tok/s, still under 92.9) but it
should be resolved before anyone trusts the acceptance column.

The `spec/*` baseline rows are now meaningless — the harness PASSes them at
0.15x of plain, because they were seeded when the arm was measured differently.
A row whose gate can pass while the feature loses 6.7x is worse than no row.

## Rule

Benchmark against the configuration that ships, not the one that is easy to
instrument. The eager path was the convenient baseline because
`bench_batch_decode` defaults to it; the number it produced was real and the
conclusion drawn from it was false. Two rows in one baseline file, side by
side, is what exposed it — which is the argument for one snapshot covering
every path rather than a per-feature script.
