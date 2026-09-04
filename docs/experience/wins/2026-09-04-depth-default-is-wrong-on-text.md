# The spec depth winner flips with the corpus: depth 1 on wikitext, 3 on GSM8K — sm70, 2026-09-04

> Status: measured, default NOT flipped and now believed CORRECT as shipped for
> reasoning traffic. The title of this file said "wrong on text" when it was written
> against wikitext alone; the same-day section at the bottom corrects that with a
> GSM8K acceptance number, and the filename is kept so links resolve.
>
> Every number here is B=1 ncols=1. Task #40 is the B-axis measurement.

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

Quote acceptance with the corpus it was measured on, and never carry a
depth or width default from one corpus to another. A cost measured on any prompt
transfers (1.001-1.036x across corpora here, 2.5% across runs); an acceptance does
not transfer at all — synthetic ids, wikitext and GSM8K give p = 0.81 / 0.59-0.69 / 0.92
at the widths each was measured on, and depth 1 wins at the first two while depth 3
wins at the third.

The corollary that cost me a headline: **"real text" is not one distribution.**
Replacing synthetic ids with wikitext felt like the end of the question and was only
the end of one arm of it. One corpus licenses a statement about that corpus.

Second: two runs of the identical command, or the number is not a number. The first
wikitext run read 97.28 ms and 24.3 tok/s at depth 3 against 61.70 and 38.3 for the
two runs after it — a 1.58x outlier with identical tick counts, i.e. the same work
taking longer, on a host whose GPU was idle by the time it was checked. It produced
a *negative* verify cost (-24.33 ms), which is what caught it; had the contention
been 20% instead of 58% the run would have published a plausible depth curve.
`ab_draft_depth.py` now samples SM clock, throttle reasons and load per depth, and
prints per-passage rows, so a contaminated run says so.

Third, and it is a defect I introduced: **repeating one prompt is not repeating the
measurement.** Adding `--batch` replaced `for p in prompts` with `prompts[:B]`, so at
B=1 the four "per-passage" rows all measured passage 0 and printed 2.49 four times.
Identical rows read as a clean instrument and are actually a stuck one — they hid the
15.8% between-passage spread that is the only honest error bar on any p here. The
script now cuts `prompts` into disjoint groups of B, refuses a `--prompts` that is not
a multiple of `--batch`, and prints the group count so a single-group run says on its
own header that it shows no spread.

## 2026-09-04, same day: the winner flips with WHICH text, and one corpus is not "text"

A peer session measured the same B=1 config on **GSM8K** with real generated
continuations: 6.12 tok/forward at W=8, i.e. **p = 0.922** per position. Wikitext at
W=4 reads 2.36 as a 3-passage mean, and its three passages read **2.49 / 2.44 /
2.15** individually — a **15.8% between-passage spread**, p = **0.689 / 0.676 /
0.592**. That spread is the resolution limit on any corpus-based depth claim. The two
corpora differ by 1.43x at W=4 (3.55 vs 2.49 tok/fwd), well outside it.

Run-to-run, by contrast, is nil: a re-run of passage 0 read 2.49 to three digits,
the same value the first run read. Cost repeats, acceptance does not — and the
variation is across passages, not across runs.

Applying both acceptance regimes to **this entry's own measured tick costs** (an
independent B=1 re-run: 35.04 / 56.39 / 61.42 / 99.61 ms at depths 1-4, reproducing
the table above to within 2.5%):

| depth | W | tok/s at p=0.689 (best wikitext passage) | tok/s at p=0.592 (worst) | tok/s at p=0.922 (GSM8K) |
|---:|---:|---:|---:|---:|
| **1** | 2 | **48.2** | **45.4** | 54.8 |
| 2 | 3 | 38.4 | 34.4 | 49.1 |
| **3** | 4 | 40.5 | 35.0 | **57.9** |
| 4 | 5 | 27.3 | 22.8 | 42.9 |

**The depth winner flips: 1 on wikitext, 3 on GSM8K** — and depth 1 wins on wikitext
by 1.19x on its most-predictable passage and 1.30x on its least, so the 15.8%
acceptance spread does not reach the flip. The corpus does. So this entry's headline
— "the depth default is wrong on text" — is too strong, and is corrected here: it is
wrong on *wikitext*, and right as shipped on GSM8K. The shipped default of 3 is the
better choice for the reasoning traffic this stack is being built for.

Why the sign is what it is: GSM8K completions are the model's own arithmetic
scaffolding, highly self-predictable; wikitext is encyclopedic prose the draft head
has to guess at. A 1-layer MTP head does much better on the former. Nothing here
measures chat, code, or a mixed workload, and those are what a served default
actually faces.

**What this leaves standing, and it is the useful half:** tick cost is
prompt-independent (measured within 3.6% across corpora, and 2.5% across two runs of
the same one), so the cost column transfers and only acceptance has to be re-measured
per distribution. The rung mechanism is also unchanged — depth 3 costs **1.753x**
depth 1's tick because W=4 crosses rung 2 → 4, whatever the acceptance is. What is
*not* established by any measurement here is a single depth default; that needs the
serving mix, not another corpus.

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
