# The draft is 55-68% of a speculative tick, and it is launch-bound — 2026-09-02

> Status: measured, not yet acted on. This entry is the price tag for two
> candidate fixes, not a win.

## Context

Whether to pursue DFlash/DSpark-style block-parallel drafting — one draft forward
emitting a whole block, instead of our D sequential forwards — turns on one
number: what fraction of a speculative tick is drafting. A head that removes D-1
draft forwards and changes nothing else is capped by exactly that fraction.

An earlier verdict rejected the DSpark head because "its selling point over our
MTP head is draft accuracy, which 96.8% top-1 says is not the bottleneck." That
answered the wrong question. Accuracy is not the bottleneck and the head's real
property is not accuracy — it is that the block comes out of ONE forward.

## Measuring it without touching the tick

The direct approach fails twice over. `prof_spec_tick.py` syncs around each
wrapped method, which breaks CUDA-graph replay: it read 0.4 tok/s where the same
config serves 48.4 (`errors/2026-09-02-synchronize-inside-a-captured-graph.md`).
Building a fresh engine per depth OOMs, because `shutdown()` only joins the
daemon thread while the KV pool and captured graphs live on.

`scripts/ab_draft_depth.py` measures the slope instead. Depth D costs D draft
forwards plus one verify, so ms/tick is affine in D — sweep it and regress. One
engine, `_spec_depth` mutated in place (it is read per tick at `engine.py:1059`,
`1079`, `1106`), each width warmed before it is timed.

ctx 1024:

| depth | ms/tick | tok/fwd | tok/s |
|---:|---:|---:|---:|
| 1 | 42.36 | 1.92 | 45.4 |
| 2 | 61.29 | 2.65 | 43.2 |
| 3 | 66.76 | 3.34 | **50.1** |
| 4 | 90.96 | 3.74 | 41.1 |

Four-point fit: `ms = 27.52 + 15.13 * depth`. One draft forward is 15.13 ms, so
at depth 3 drafting is 45.4 of 66.8 ms = **68%** of the tick.

**The fit is not clean, and the reason matters.** Residuals are structured
(-0.3, +3.5, -6.2, +2.9) because verify width is a staircase too: width = depth+1
and the sm70 rungs are {1,2,4,8}, so depth 2 pays 4 rows for 3 and depth 4 pays 8
for 5. The engine warns about precisely this (`engine.py:1366`). Restricting the
fit to the depths that land ON a rung — 1 (w=2) and 3 (w=4) — gives 12.2 ms per
draft and a **55%** share. The honest answer is a range: **drafting is 55-68% of
the tick**, and a single regression through a staircase would have reported 68%
with false precision.

## What it prices

**Block-parallel drafting**: 66.76 -> 36.5..42.3 ms/tick at the same 3.34
tok/forward = 50.1 -> **79-92 tok/s, 1.58-1.83x**. Break-even is 1.83
tok/forward: a parallel head may lose nearly half our acceptance before the
cheaper draft stops paying. That is a genuinely large prize, and the user's
argument for it is correct.

**But check the cheaper lever first.** The draft is launch-bound, not
compute-bound, and it is still outside the captured graph. `_draft_step` is called
after `_run_decode_graph` returns (`engine.py:984`), so it pays the eager price D
times per tick — where the trunk went 103.58 -> 39 ms from capture, worth 2.66x
(`errors/2026-08-31-draft-step-outside-graph.md`). 15.13 ms is ~60x the head's own
0.25 ms fp4 bandwidth floor. Recovering even half of the trunk's capture factor
takes the same 55-68% down with no new architecture, no KV/state bookkeeping
change, and no accuracy risk.

The blocker is visible and narrow: `engine.py:1100` and `:1102` call `.tolist()`
on the sampled token and confidence every step — a device-to-host sync per depth
step — and `:1094` copies that token straight back to the device via
`np.array(...)`. Everything else in the loop is already device-resident, and
`backend.greedy` returns a tensor that `backend.embedding` would accept directly.
The host roundtrip exists only to append to a Python `chains` list.

The two levers attack the same term, so whichever lands first shrinks the other's
remaining prize. Capture is the smaller diff.

## Rule

Before pricing an architecture change, measure the term it attacks — and measure
it by varying a parameter the system is linear in, not by instrumenting it. One
free variable beats any amount of in-graph probing, and a structured residual
tells you when a second staircase is hiding in the fit.

Second: when a component costs 60x its own bandwidth floor, the problem is not
what it computes. Look for the launches before redesigning the math.

## Results

| date | commit | machine | target | model | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|
| 2026-09-02 | 1cb702d | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 20.0 (spec d3) | 50.1 @ 1024 ctx |

Raw artifacts: `scripts/ab_draft_depth.py`.
