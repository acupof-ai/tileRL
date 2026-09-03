# Single-stream B=1 on the V100: 45.9 tok/s peaks at ctx=1024, and the tick is flat, 2026-09-03

> Status: **baseline, no runtime change.** First V100 B=1 context curve on this branch.
> **45.9 tok/s at ctx=1024** is the best single-stream figure measured on this card; ctx=32
> is **32.7**, reproducing the recorded 32.4 to **+0.9%** (noise floor 1.16%). The rate peaks
> in the middle of the range, and not because anything got faster: **the tick is flat at
> 62-70 ms across a 128× context span** and the whole curve is acceptance.
>
> Also confirmed, as predicted from the code before the run: **the last three commits are
> worth zero at B=1.**

## Why this run existed

Every number this branch produced for two days was **B=8 aggregate**. The user's target is
one concurrent request, where an aggregate is not just unhelpful but misleading: the B=8
88.5 tok/s is **~11 tok/s per request**, worse than the B=2 serve default's 20.9. So the
single-stream curve had to be measured rather than divided out of something else.

The last V100 B=1 measurement was **32.4 tok/s** (ctx=32, depth 3, `ncols` off) from the
batch sweep, taken before `16382c8` (draft readout), `70b53f2` (PO f16) and `a158672`
(per-width KVSPLIT) landed.

## The curve

B=1, depth 3, `--tokens 64`, `expandable_segments` on, pool 554 blocks (510 MB), 8 graphs
precaptured in 71 s:

| ctx | tok/s | ms/tok | tok/forward | tick ms |
|---:|---:|---:|---:|---:|
| 32 | 32.7 | 30.6 | 2.10 | 64.3 |
| 512 | 43.5 | 23.0 | 2.86 | 65.8 |
| **1024** | **45.9** | **21.8** | 2.86 | **62.4** |
| 2048 | 39.8 | 25.1 | 2.62 | 65.8 |
| 4096 | 34.7 | 28.8 | 2.42 | 69.7 |

`tick ms = ms/tok × tok/forward`, so it is derived from the two measured columns rather than
timed separately.

**The tick is flat: 62.4 to 69.7 ms over ctx 32 → 4096.** A 128× context span costs 11.7% of
tick time. That is the split-KV attention entry's claim — "at KVSPLIT=32 a block owns ≤128
positions, below launch overhead, so the context slope is gone" — holding at B=1 on the real
model, not a microbenchmark.

So the rate curve is **entirely acceptance**: `tok/forward` runs 2.10 → 2.86 → 2.86 → 2.62 →
2.42. The draft head conditions better on 512-1024 tokens of context than on 32, and falls off
again past 2048. Peak throughput sits where acceptance peaks.

## The prediction, and what it settles

Committed in the run script before launch: *ctx=32 should be unchanged within noise, because
B=1 depth 3 = 4 rows → the 4 rung, below `_NCOLS_MIN_M = 32` (`backend.py:511`), so `ncols=2`
never engages; PO at 1 row is ~96-192 MiB, nowhere near the ceiling the f16 and per-width
KVSPLIT work removed; `last_only` reduces 4 rows, not 32.*

Two independent quantities confirm it:

| | recorded (before the three commits) | now | delta |
|---|---:|---:|---:|
| ctx=32 tok/s | 32.4 | 32.7 | **+0.9%, inside 1.16% noise** |
| ctx=32 tok/forward | 2.10 | **2.10** | **identical** |

`tok/forward` matching exactly is the stronger half — a rate can coincide, an acceptance count
matching to the hundredth means the same tokens were drafted and accepted. **Three commits of
memory work bought single-stream nothing**, which is what the code said they would.

That is not a defect in those commits: they removed an OOM at B=8 ctx=512, which is a capacity
result and was reported as one. It does mean **the single-stream path has had no attention paid
to it on this branch**, and its levers are different ones.

## 45.9 is the V100 record, and 87.5 is not a V100 number

Checked against every B=1 figure in the tree, because several are easy to misquote:

| number | machine | what it is |
|---:|---|---|
| **45.9** | **V100 / sm70** | **this run, ctx=1024 — the best measured single-stream on this card** |
| 32.4 / 32.7 | V100 / sm70 | ctx=32, the previous and current short-context point |
| 20.3 | V100 / sm70 | the cross-machine comparison figure, on an older kernel |
| 52.6 | H20 / sm90 | the 27B baseline entry |
| 74.9, 87.5, 90.9 | H20 / sm90 | the sm90 decode progression |

**87.5 must never be quoted as a V100 number.** The recorded sm70-vs-sm90 ratio is 4.31×,
logged as **97% of the two cards' bandwidth ratio** — so most of that gap is the memory system,
not missing optimization.

## What this decides for the demo

A page driven by one user typing will sit at **~44-46 tok/s** for realistic prompts (hundreds
to ~1000 tokens of context), not the 32.7 a toy prompt shows and not anything resembling 88.5.
That is the number a demo may display, labelled per-request.

Two things found while reading for it, both filed rather than started:
`server.py:244` `_stream` emits the **whole completion as one SSE delta** (its own ponytail
marker calls incremental streaming "day-2"), so a viewer sees a pause and then a block of text
— the rate is invisible. `_Req.output` (`engine.py:134`) is a live per-tick list and
`_await_completion` already polls at 20 ms, so this needs an accessor, not an event system.

## Harness fix that made the run honest

`bench_ctx_decode.py` had `b = max(4, args.batch)`, so `--batch 1` built the engine with
`max_batch=4`: **990 MB of block pool instead of 510, and four times the decode graphs**, for
rows that could never be admitted. Floored at 2 (`61ee8c6`) — the comment above it documents
the padding-slot leak as a `num_slots = b + 2` constraint, not a `max_batch` one, so 4 had no
recorded justification. **B ≥ 4 is byte-identical**, so no earlier measurement on this branch
moved; verified locally across submit counts 1/2/4/8.

## Rule

**A per-request target makes an aggregate a wrong answer, not a rough one.** 88.5 tok/s and
"11 tok/s per request" are the same measurement, and only one of them answers "how fast is it
for me". When the question changes from throughput to latency, the previously headline number
has to be re-derived, not carried over — and here the carried-over number would have pointed
at a configuration that is **half the speed** of the shipped default for this use.

Second: **a prediction that names the mechanism can be checked on two axes.** Predicting only
"ctx=32 will not move" would have been satisfied by a coincidence. Predicting *why* — the rung
is below the ncols gate, so the same kernel runs — also predicts `tok/forward` is untouched,
and that matched to the hundredth. The second axis is what makes it a confirmation rather than
a null result.

## Gate

Prediction committed in `/tmp/b1base.sh` before launch, quoted above. GPU verified idle before
launch. `tok/forward` cross-checked against the #41 sweep's recorded 2.10. Every quoted B=1
figure checked for its machine before use. No runtime code changed.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | **B=1 d3 ctx=1024** | **45.9 tok/s — best single-stream on this card** |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | B=1 d3 ctx=512 | 43.5 tok/s |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | B=1 d3 ctx=2048 | 39.8 tok/s |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | B=1 d3 ctx=4096 | 34.7 tok/s |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | **B=1 d3 ctx=32** | **32.7 vs recorded 32.4 — +0.9%, inside noise** |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | ctx=32 tok/forward | **2.10 vs recorded 2.10 — identical** |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | **tick ms across ctx 32→4096** | **64.3 / 65.8 / 62.4 / 65.8 / 69.7 — flat, 11.7% over 128×** |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | tok/forward across the same span | 2.10 / 2.86 / 2.86 / 2.62 / 2.42 — **the whole curve** |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | **value of the last 3 commits at B=1** | **zero, as predicted from the code** |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | B=8 88.5 agg expressed per request | ~11 tok/s — **below the B=2 default's 20.9** |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | precapture at max_batch 2 | 8 graphs, 71 s |
| 2026-09-03 | 61ee8c6 | V100 | cuda sm70 | qwen38-27b | pool at B=1 after the floor fix | 510 MB, was 990 |
