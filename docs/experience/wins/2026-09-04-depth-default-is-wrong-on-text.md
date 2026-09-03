# The spec depth default is wrong on text: depth 1 beats depth 3 by 1.266x — sm70, 2026-09-04

> Status: measured, default NOT flipped. The flip needs the ncols=2/B=4 kernel
> (task #40) priced first, because every number here is B=1 ncols=1.

## Context

Two entries left the depth default open with the same sentence: acceptance is a
property of the prompt, random-vocabulary ids are one end and `range(10, 10+ctx)`
the other, and neither is the serving distribution. Wikitext-103's test split is in
the pod's HF cache, so "real text" was one script argument away rather than a
project. `scripts/corpus.py` loads it; `--prompt wikitext|random` selects the arm.

## What Worked

ctx=1024, B=1, 3 passages per depth, one engine with depth moved in place:

| depth | W | mean chain | ms/tick | tok/fwd | tok/s | vs shipped |
|---:|---:|---:|---:|---:|---:|---:|
| **1** | 2 | 2.00 | 35.92 | 1.74 | **48.5** | **1.266x** |
| 2 | 3 | 2.97 | 56.41 | 2.17 | 38.4 | 1.003x |
| 3 (shipped) | 4 | 3.90 | 61.70 | 2.36 | 38.3 | — |
| 4 | 5 | 4.77 | 98.27 | 2.43 | 24.7 | 0.645x |

**Repeatable to 0.3%**: an independent second run of the identical command read
48.4 / 38.5 / 38.2 / 24.7, and both ran at 1530 MHz with throttle reasons 0x0.

**The mechanism is the rung, not the acceptance.** Depth 3 buys 1.356x the tokens
of depth 1 (2.36 vs 1.74) for 1.719x the tick (61.70 vs 35.92, rung 4 against rung
2), netting 0.789x. Depth 1 wins because W=2 is the last width that stays on the
cheap rung.

**Cost is prompt-independent; acceptance is not — which is the whole finding.**
Same code, same rungs, random ids against wikitext:

| depth | ms/tick random | ms/tick wikitext | ratio | tok/fwd random | tok/fwd wikitext |
|---:|---:|---:|---:|---:|---:|
| 1 | 34.98 | 35.92 | 1.027 | 1.64 | 1.74 |
| 2 | 54.43 | 56.41 | 1.036 | 2.62 | 2.17 |
| 3 | 61.63 | 61.70 | 1.001 | 2.99 | 2.36 |

Tick cost agrees within 3.6% at every depth, so the two arms are running the same
kernels at the same shapes and the harness is measuring the same thing. Acceptance
does not agree: **wikitext accepts worse than uniform random ids at every depth
past 1** (2.36 vs 2.99 at depth 3, 0.79x), and that difference is what inverts the
choice — on random ids depth 3 wins at 48.4 tok/s, on text it loses to depth 1.

That ordering is the surprise and it is worth stating plainly: a draft head that
predicts uniformly-random token ids better than English prose is predicting
something other than language. The 1-layer MTP head sees `fc([embed(t), h_trunk])`,
and on random ids consecutive positions are independent, so a head that learns
"continue whatever the trunk's hidden implies" has no wrong-continuation to be
punished by. Not measured here, and it does not change the verdict: text is the
distribution we serve.

## Rule

Quote acceptance with the corpus it was measured on, and settle a depth or width
default on text only. A cost measured on any prompt transfers (1.001-1.036x here);
an acceptance measured on synthetic ids transfers to nothing, and it inverted this
default.

Second: two runs of the identical command, or the number is not a number. The first
wikitext run read 97.28 ms and 24.3 tok/s at depth 3 against 61.70 and 38.3 for the
two runs after it — a 1.58x outlier with identical tick counts, i.e. the same work
taking longer, on a host whose GPU was idle by the time it was checked. It produced
a *negative* verify cost (-24.33 ms), which is what caught it; had the contention
been 20% instead of 58% the run would have published a plausible depth curve.
`ab_draft_depth.py` now samples SM clock, throttle reasons and load per depth, and
prints per-passage rows, so a contaminated run says so.

## What this does NOT license

**The default is not flipped.** Every number here is B=1 with ncols=1, and the
serving default is B=4 with ncols=2, where a padding row costs 3.3x a useful one
(`errors/2026-09-03-batching-is-non-monotone-padding-rows-cost-3x.md`) and W=2 at
B=4 launches 8 rows against W=4's 32. The rung arithmetic that makes depth 1 win at
B=1 is a different arithmetic there, so flipping on this evidence would trade a
measured B=1 win for an unmeasured B=4 regression. Task #40 is that measurement.

**Wikitext is not chat traffic either.** It is encyclopedic prose; a chat or code
workload will accept differently, and code especially so. This entry replaces
"synthetic vs synthetic" with "one real corpus", which is progress, not an answer.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-04 | d5ecca0 | V100 32GB | cuda sm70 | qwen38-27b | — | 20.6 | 48.5 |
| 2026-09-04 | d5ecca0 | V100 32GB | cuda sm70 | qwen38-27b | — | 26.1 | 38.3 |

Rows are depth 1 (W=2) and the shipped depth 3 (W=4), single-stream, ctx=1024,
wikitext-103 test, 3 passages x 128 tokens each, slots 4, max_batch 4.

Raw artifacts on the V100: `$HOME/tilerl-logs/ds8.log`, `ds9.log` (the two
agreeing runs), `ds5.log` (the contaminated one), `bud6.log` (per-kernel profile of
the wikitext arm: 49.83 ms/forward GPU against 84.54 wall, so a third of the tick
is host-side and outside the captured graph).
