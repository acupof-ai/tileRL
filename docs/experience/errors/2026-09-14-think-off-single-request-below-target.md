# Think-off single request runs at 48.1 tok/s, below the 50 target — 2026-09-14

> Status: open. Unexplained; one bimodal run recorded on V100 sm70.

## Context

V100 sm70, 27B NVFP4, MTP d1, `--decode-graph`, think-off single request:
**48.1 tok/s** against the ≥50 tok/s target. One run was bimodal at
**46.4 tok/s**, splitting into 39.5 vs 35.6 ms/fwd phases with no
configuration change between them.

## Root cause

Unknown. The bimodality is the diagnostic lead: the two ms/fwd levels say
the same engine spends different time per forward in two phases, but no
phase-attributed profile separates the two levels yet. Placement, card
state and first-position JIT controls have not been run.

## Fix

None. Next arm: phase-attributed per-forward timing across A-vs-A reruns in
one process, then placement controls on a second card.

## Rule

A miss against a perf target stays open until the per-forward cost is
attributed; an average tok/s number does not name a mechanism.
