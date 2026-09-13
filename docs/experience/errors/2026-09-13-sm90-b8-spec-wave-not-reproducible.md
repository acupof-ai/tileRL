# An sm90 B=8 spec wave is not reproducible across identical cold waves — 2026-09-13

> Status: **open.** Owner: cc. The warm-adoption gate (#564) is B=1-exact; this
> defect is what its B=8 wave was measuring.

## Context

The #564 card gate compared one warm-adopting B=8 wave against one cold B=8
wave and found 7/8 followers differing (one at token 0). Before blaming warm
adoption we ran a placement control: the SAME 8 followers through TWO separate
fresh COLD engines, each in one B=8 wave. With no warm adoption anywhere, the
two cold waves still disagree.

27B NVFP4, H20 card-4, head f6882ac6 (the #563 unfused-writer guard ACTIVE),
sparse k=128, spec_depth=1, greedy (temperature 0.0, seed 0), 404-token prompts
(384 shared + a 20-token per-row tail), 64 generated tokens,
`scripts/probe_warm_control.py` with one engine per subprocess.

## What happened

Cold wave A vs cold wave B: **3/8 followers token-equal** (rows 2, 3, 4); the
other five diverge, at decode positions:

| row | 0 | 1 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|
| first differing token | 27 | 30 | 31 | 5 | 30 |

Same code, same inputs, same greedy settings, independent waves in separate
processes. Greedy decode from bit-identical first-token logits cannot diverge,
so the per-row state already differs silently before the position shown.

The B=1 control is exact: the same 8 followers run SEQUENTIALLY (one follower
per tick), warm vs cold, are 8/8 token-equal with first-token logits max_abs
0.0. The defect is B>1-only and needs several sparse spec rows sharing ticks.

## Why it is not the fused-writer defect

The #563 guard forces every sparse tick through the unfused K/V writer at this
head; #567's later correction was a routing bug in the FUSED attention branch,
which the guard never takes. A divergence under the guard cannot be that cell.
This is a separate defect: B=8 spec-wave nondeterminism / non-reproducibility
on sm90 with the exact unfused path.

Also distinct from the guarded residual gap in
[2026-09-12-sm90-fused-attn-prep-sparse-packed-prefill.md](2026-09-12-sm90-fused-attn-prep-sparse-packed-prefill.md)
(0.7986 vs 0.915 accuracy on one matched set): that compares sparse against
dense; this one is sparse-vs-sparse across two identical runs, so it is
reproducibility, not fidelity-to-dense.

## Candidate cells (not yet localized)

- a packed prefill / verify tick shared by >1 ragged sparse rows writing or
  reading the wrong slot under the unfused writer (the same class as the
  earlier fused B>1 bug, different path);
- B>1 batched draft proposal / admit ordering;
- uninitialized padding in the packed `[selected;own]` block table or packed
  seq_len differing between two batches that should be equivalent.

The CPU target does not reproduce: the same B=8 warm wave is 8/8 token-equal
on CPU tiny, so localization needs a served-shape sm90 probe that dumps
per-row first-token logits in two identical cold B=8 waves.

## Rule

A warm-vs-cold mismatch under batching is not evidence about the warm path
until cold-vs-cold in the same batch shape is reproducible. A one-wave
comparison inherits every B>1 nondeterminism in the engine and attributes it
to the change under test.
