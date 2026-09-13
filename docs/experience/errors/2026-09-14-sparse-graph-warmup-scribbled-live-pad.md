# Lazy sparse-graph capture warmup scribbled live frame block 0 / slot 0 — 2026-09-14

> Status: fixed on CPU/code at <pending-sha>; device confirmation pending-remote.

## Context

V100 (dd66a9c4, #580 explicit `--decode-graph`, sparse k=128),
`~/SM70_SPARSE_GRAPH_REPRO.md`:

- sparse + spec d1 + graph: deterministic degenerate long outputs
  ("Introduction / library for Python ..." loops); 1-token MMLU answers
  correct; exact on every other arm (sparse eager ±spec, sparse graph no spec).
- sparse + graph no spec under B=4 MMLU: CUDA illegal memory access after
  3 finishes, 3 slots leaked, process poisoned.
- dense d1 + graph soaks never failed.

## Root Cause (hypothesis, untested on device)

Sparse graphs are captured LAZILY per `(B,W,cmax_bucket,own_w)` key on the
first matching tick (`_run_sparse_decode_graph`, engine.py); there is no
sparse precapture (`precapture` walks only the dense `graph_keys`).

`_SparseDecodeGraph.__init__` runs two warmup forwards and the capture
forward on its static buffers BEFORE `sf.fill()` ever ran: `own_table` zero
and `_slots` zero, `page_base` zero. The sm70 write kernel
(`kernels_mma.py make_write_tokens_f32`) computes
`BlockTable[b, pos//16 - PageBase]` = physical 0, and the GDN fused decode
writes States/StepStates/conv window/parity of slot 0. On the dense path the
same warmup exists (`_DecodeGraph`), safe only because dense graphs are
precaptured at startup when block 0 / slot 0 are the pad frame; the lazy
sparse capture hits live traffic, where block 0 / slot 0 belong to the
running row. Each GDN warmup/capture forward also flips that slot's
`win_parity` 3 extra times, inverting the conv-window bank the real replay
reads — matching the spec-only, long-answer-only signature (the W=2 verify
graph is created at a later tick than W=1).

## Fix

Both graph constructors take `pad_slot`/`pad_block`; before warmup AND
capture the block table and slot buffer point at the engine's reserved pad
frame (the frame run() already uses for replay pad rows). Validated in
range. `run()` / `sf.fill()` repoint the buffers for real rows before every
replay. The dense graph gets the same steering (its precapture ordering was
the only thing protecting it).

## Rule

A capture's warmup forwards are real forwards: they write through whatever
address the zeroed static buffers name. Reserve the scratch frame at
construction and point at it explicitly — "block 0 is unused right now" is
timing, not an invariant.

## Device plan (ops-0b)

1. `scripts/probe_sparse_graph_warmup_pad.py <ckpt> --draft <mtp>` on MAIN:
   forces a lazy re-capture mid-decode; expects MAIN_BUG (live slot parity
   extra flips / non-pad frames dirty).
2. Same on this branch: BRANCH_OK.
3. `parity_ref.py` long answers + B=4 MMLU smoke on the branch: sparse+d1
   +graph token-equal to dense, no illegal access, no leaked slots.
