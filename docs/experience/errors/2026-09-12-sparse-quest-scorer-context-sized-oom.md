# Sparse Quest scorer allocated the whole context at once and OOMed past 32k — 2026-09-12

> Status: **fixed on `fix/sparse-scorer-page-chunk`.**

## Context

After the #528 capacity/keying fixes, the V100 128k sparse run died ~34k tokens
into prefill with `CUDA out of memory. Tried to allocate 4.19 GiB` (4.08 GiB
free), in `quest_scores` (`sparse_engine.py`).

## Root cause

`quest_scores` built the bound for every candidate page simultaneously:

```
qi[:, None] * kmin[None]   # [Tq=512, Cp, Hkv=4, D=256] f32
```

The kmin and kmax operands are each `Tq*Cp*Hkv*D*4` bytes — at Cp≈2048 that is
2.1 GiB each, 4.2 GiB live, and it grows with context. The 32k run (Cp≈2048 at
its deepest) barely fit the V100's ~4 GiB post-weights headroom; 128k crossed
it. It is also an avoidable 17 GiB transient on the H20. The selector only
needs one score per page.

## Fix

Score candidate pages in chunks of 64 (`_SCORE_PAGE_CHUNK`). The reductions —
max over the chunk's queries, sum over KV heads and head dim — commute with a
split over the page axis, so the chunked result is bit-identical to the
all-at-once computation (gate asserts `torch.equal`, Cp=201 to cover the
non-multiple tail). Operand peak stays under ~270 MiB regardless of context.

## Rule

A selector that scores a growing candidate set must not hold the f32 outer
product for the whole set; rank-one bound products reduce over the same axes
the score sums, so split the candidate axis and accumulate — exact, bounded.
