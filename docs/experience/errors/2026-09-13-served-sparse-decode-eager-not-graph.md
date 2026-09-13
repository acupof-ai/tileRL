# A served sparse engine ran every decode tick eager, never the sparse graph — 2026-09-13

> Status: **closed (code) — #557 (3218f0aa) landed both fixes and CPU
> forced-graph gates pin them**: the dense branch is gated on
> `self._sparse is None` (`engine.py` `_run_forward`) and `build_engine`
> resolves `sparse_device_select=None` to the decode-graph predicate. The
> card-2 served-default PATH counters remain device-pending (command below),
> pending-remote, not an open code defect.
> Owner: 52 (fix), measurements cc/5f.

## Context

#557 made the sparse steady-state decode/verify tick graph-capturable, but two
gaps meant a *served* engine never replayed it. Sparse is opt-in
(`--sparse-k N`, #558); once on, this is what happened under the serve default.

## Two gaps

1. **The dense graph shadowed the sparse graph.** `Engine._run_forward` tried
   `_run_decode_graph` before `_run_sparse_decode_graph`, with no sparse guard.
   On CUDA the decode graph auto-enables, and the dense `_DecodeGraph`
   captured a `BatchKv` with `sparse=None` over the row's own blocks, returned
   True, and won EVERY sparse decode tick. cc's card-2 diag (v557e, 32k B=1):
   `dense_graph=63 sparse_graph=0 eager=0`, `_sparse_graphs` empty,
   `buckets=0`; captured vs "eager" were identical (~1.000 ms ratio) because
   both arms replayed the same dense graph. The CPU test suite could not see
   it — `RefBackend` leaves `_decode_graph_on=False`.
2. **The served default never enabled device selection.** `serve` passes no
   `sparse_device_select`; the kwarg default was False, so even with correct
   dispatch every pure-decode tick ran the eager full-candidate re-selection —
   there was no 8-tick refresh cadence in production. 5f measured card-3,
   B=1, k=128: **88.23 ms/tick median vs the dense graph's ~13 ms**.

## Fix (#557)

- the dense graph branch is gated on `self._sparse is None`, so a sparse row
  can only reach `_run_sparse_decode_graph`;
- `build_engine` resolves `sparse_device_select=None` to the same answer as
  the decode graph: device select + the sparse graph are ON wherever the
  decode graph auto-enables (sm90), OFF on CPU/sm70 and under explicit
  `decode_graph=False` (the true sparse-eager comparison arm).

The served shape after the fix: 7 resident-only captured decode ticks, then 1
eager full-candidate refresh (`SPARSE_REFRESH_TICKS=8`), instead of eager
re-selection on every tick.

## Pending device evidence (card-2)

Pending-remote command (H20 off-limits as of 2026-09-13; run at the next card
window):

```bash
scripts/pod_run.sh diag557 <card> -- python3 scripts/probe_557_card2.py
```

The card-2 diag (`scripts/probe_557_diag.py`, `PATH default` arm builds with
the served kwargs) must show on the default: `sparse_graph` ticks with
`buckets>0` across the 8-tick refresh boundary, zero dense-graph ticks,
default-vs-captured and captured-vs-eager token equality over 64 steps, and
ms/tick for captured / eager / dense. A prior token_equal=False run is void —
the probe shared one module-level RNG across arms (fixed: re-seed per arm).

## Rule

A captured path's existence is not a serving result until the production
default is shown to REPLAY it: count per-tick paths on an engine built with
no test kwargs. Both the dispatch order (an earlier graph can shadow the new
one) and the flag's default resolution have to be pinned, and neither is
visible on a backend where graph capture is off.
