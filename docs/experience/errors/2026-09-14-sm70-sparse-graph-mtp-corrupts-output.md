# sm70 sparse decode graph + MTP d1 corrupts multi-token output — 2026-09-14

> Status: open. H1 (warmup/capture scribbling live block 0) disproved on
> device; #585 closed. H2 (a capture at a cmax-bucket transition) TESTED BAD
> on device 2026-09-17 — the first replay at every sparse bucket and both
> widths corrupts the first token; see
> [2026-09-17-sm70-sparse-decode-graph-replay-corrupts-first-token.md](2026-09-17-sm70-sparse-decode-graph-replay-corrupts-first-token.md).
> The hybrid engine (#586) avoids the defect by forcing the sparse graph off.

## Context

V100 sm70, sparse k=128 + MTP depth 1 + `--decode-graph` deterministically
corrupts multi-token decode; 1-token answers stay exact. Under B=4 the same
configuration hit an illegal memory access with leaked slots. All eager
variants and dense+graph soaks are exact. Sparse graphs are captured lazily
per `(B,W,cmax,own_w)` key on the first matching tick.

## Root cause

Unknown. H1 — warmup forwards on zeroed static buffers addressing physical
block 0 and flipping `win_parity` on live slot 0 — was #585's hypothesis;
the pad-frame fix still emitted degenerate loops on V100 (the sm70
win_parity probe stayed inconclusive — instrument blind), disproving it.
H2, a capture firing at a cmax-bucket transition while live rows change
buckets, was tested 2026-09-17 (#700 probe) and CONFIRMED — see the
2026-09-17 entry.

## Fix

None. Workaround in force: the hybrid (#586) keeps sparse ticks eager even
with `--decode-graph` on (`_sparse_graph_on` forced false under hybrid,
asserted in a CPU gate). Next arm: capture at a forced bucket transition and
compare first-logit and `win_parity` state to eager.

## Rule

A lazy capture keyed on a live-traffic shape is a live-mutation hazard until
every key transition is captured against a pad frame, not just the first.
