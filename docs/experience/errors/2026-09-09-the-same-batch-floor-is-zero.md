# The same-batch floor is zero, and the 2xSE ruler is a sampling question — 2026-09-09

## Context

Two floor claims stood in the tree, both wrong:

1. "Same-batch gross-flip floor 15-16/500" — read from adjacent plateau points of one
   run, written into `curve_churn`'s docstring (#339) and the three-calibers entry
   (#338). The flips were called the instrument's own jitter.
2. "Instrument floor 0.2 pt (1 question / 500)" — read from a cross-process re-score,
   written into `new_best_point`'s docstring (#334/#336) and the early-stopping wins
   entry. A one-question lead was called a reading inside the instrument.

## Root cause

One quantity conflated with another.

- The **instrument floor** asks: does re-scoring the same weights on the same rows
  move? Same-batch: **0**. Cross-batch (different batching): 52 flips / 500 = 10.4%.
- The **paired SE** (`100·√(b+c)/n`, ~1.0-1.4 pt at n=500) asks: would the difference
  survive a different set of problems? It is a sampling width, not an instrument width.

The 15-16 flips were measured between adjacent curve points of one run — same batch —
so under a zero instrument floor every one of them is a real policy change; calling
them jitter used a sampling word for an instrument question. The 0.2 pt reading came
from a cross-process re-score, which batches the problems differently: a cross-batch
comparison, an instance of the 10.4% cross-batch floor, misclassified as an instrument
floor.

## Evidence

Run `76a17ea6e10a` on the pod (`/work/tilerl-realrun`): two arms
(`eval-before.jsonl` / `eval-after.jsonl`) scoring the same step-5 adapter in the same
process, same order, concurrency 8. Read directly: 500 rows each, **zero rows differ**
in correct/tokens, sums 471/500 and 125100 tokens both arms. (GSM8K; MATH is assumed
the same by engine determinism, not yet measured.)

## Fix

Docstrings and comments rewritten to name the two quantities separately (this commit):
the 2xSE ruler is stated as a sampling question everywhere it appears — `paired_se`,
`new_best_point`, `significant_decline`'s ruler, the P1 gate comment,
`require_paired_width`; `curve_churn` states same-batch floor 0 and the cross-batch
10.4% as the only operating floor. No logic changes. The #338 three-calibers passage's
"jitter" wording gets its own errata (cc). The early-stopping and churn wins entries
cite the old floors; this entry corrects them — snapshots are not overwritten.

## Rule

Instrument floor and sampling width answer different questions — never use one where
the other is asked. Same-batch: 0 (bit-identical, GSM8K 500, 2026-09-09). Cross-batch:
52/500 = 10.4%. Paired SE at n=500: 1.0-1.4 pt, the sampling ruler. A "floor" cited
without naming which comparison it was measured on is not a floor yet.
