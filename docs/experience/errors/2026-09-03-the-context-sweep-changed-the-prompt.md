# The context sweep changed the prompt, not just its length — 2026-09-03

## Context

With the KV pool fitted to the card, contexts past 4096 became reachable for the
first time, so `bench_ctx_decode.py` got `CTXS` out to 65536 and two points came
back: **38.0 tok/s at ctx=8192** and **25.5 at ctx=16384**, against 45.9 at 1024.

Decomposing 1024 → 16384, using `tick = ms/tok × tok/forward`:

| ctx | tok/s | ms/tok | tok/fwd | tick ms |
|---:|---:|---:|---:|---:|
| 1024 | 45.9 | 21.8 | 2.86 | 62.3 |
| 8192 | 38.0 | 26.3 | 2.74 | 72.1 |
| 16384 | 25.5 | 39.2 | 2.03 | 79.6 |

The 1.80x rate drop splits into a **1.278x tick** term and a **1.409x acceptance**
term. The tick half is explained: 1.13 ms per 1K of context, which is 1.89x the
byte floor for a 4-row verify reading 128 KiB/token of f32 KV — the right order,
no second mechanism needed.

The acceptance half is not explainable that way. `tok/forward` is how many drafts
the trunk accepts, a property of what the draft head is predicting. Nothing about
a longer context should cost 0.83 accepted tokens per forward.

**What follows establishes that the sweep cannot answer the question — not that a
long-context acceptance effect is absent.** The confounded rows are withdrawn; the
re-measurement has not been run.

## Root Cause

The prompt was `list(range(10 + i * ctx, 10 + (i + 1) * ctx))` — consecutive token
ids starting at 10. So the prompt's **content** is a function of `ctx`:

| ctx | ids | share of the 248320 vocab | mean id |
|---:|---|---:|---:|
| 1024 | 10..1033 | 0.41% | 522 |
| 8192 | 10..8201 | 3.30% | 4106 |
| 16384 | 10..16393 | 6.60% | 8202 |

Low ids in this checkpoint are byte-level and special tokens; higher ids are real
word pieces. Every row of the sweep therefore mixed "the context is longer" with
"the prompt is drawn from a different part of the vocabulary", and after the fact
the two cannot be separated. The 1.409x acceptance term is some unknown mixture of
a real long-context effect and an artifact of which tokens the draft was asked to
predict.

This was invisible for as long as the sweep topped out at 4096, where every arm sat
inside the first 1.6% of the vocab and the ranges overlapped heavily. Extending the
sweep is what made the confound bite — the instrument broke at the moment it
started being used for the thing it was extended for.

## Fix

`_prompt(ctx, i, vocab)` draws from one fixed distribution over the whole vocabulary
at a per-request seed, so a shorter prompt is a prefix of a longer one in
distribution and length is the only variable. Verified: `_prompt(1024,0,V)` and
`_prompt(16384,0,V)` share their first five ids, and mean id is 122212 vs 124186
rather than 522 vs 8202.

`vocab` is a required argument with a floor rather than a defaulted one — `vocab=0`
would raise from inside a 20-minute pod run, and a too-small value would silently
re-introduce exactly the narrowing this removes.

**The three long-context rows are withdrawn**, not corrected: 38.0 and 25.5 were
measured on the old prompt and there is no way to rescale them. The tick-term
arithmetic survives (1.13 ms/1K against a 1.89x byte floor) because tick cost does
not depend on which tokens are in the prompt; the acceptance column does not.

## Rule

A sweep's control variable has to be the only thing that varies. When the harness
derives the input from the swept parameter — prompt from context length, batch
content from batch size, shape from a size knob — every row differs in two ways and
the second one is invisible in the output. Check what else the parameter reaches
before extending a sweep's range, because a confound that overlaps harmlessly in the
old range can dominate in the new one.
