# serve --decode-graph: explicit captured-tick opt-in past the sm70 exclusion — V100, 2026-09-13

> Status: pending-remote. CPU gates green locally; the V100 number is ops-0b's
> arm on the `feat/serve-decode-graph` branch.

## Context

The V100 serve measures 7.90 tok/s single-request decode with no spec and the
decode graph auto-disabled on sm70 (`_graph_on`,
[errors/2026-09-12-failed-graph-capture-poisons-allocator-sm70.md](../errors/2026-09-12-failed-graph-capture-poisons-allocator-sm70.md));
spec alone (~1.8 tok/forward) reaches about 14, so the captured tick is the
remaining lever. The #545 exclusion is all-or-nothing: `decode_graph=None`
returns False on sm70 and also keeps the #557 sparse graph off, but the serve
CLI had no way to pass the explicit True `build_engine` already honours.

## What Worked

One tri-state flag: `serve --decode-graph` sends `decode_graph=True` through
`cmd_serve` → `_build_engine` → `build_engine`; omitted stays None (auto, as
today — sm70 excluded, every other arch unchanged). With `--sparse-k` the
sparse request routes to `_run_sparse_decode_graph`, which captures the packed
[selected; own] sparse tick, not the dense `_DecodeGraph` whose sm70 failure
the exclusion exists for. The sparse capture has never run on sm70, so this
flag is the measurement switch, not a claim it succeeds: a failed capture
poisons the process allocator with no in-process recovery.

CPU gate `test_serve_decode_graph_flag_plumbs_to_engine_on_sm70`: flag parses
as True and reaches `build_engine`; red before the plumb (unknown argument).

## Rule

A backend auto-exclusion needs an explicit opt-in switch at the CLI when the
excluded thing has more than one shape and only one shape was observed
failing — but the switch is for measurement, and the pending number says
whether the untested shape escapes the observed failure.

## Results

| date | commit | machine | target | model | flags | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---|---:|---:|---:|
| 2026-09-13 | pending | n37-002-027 V100 | sm70 | Qwen3.8-27B NVFP4 | `--sparse-k 128 --draft <head> --decode-graph` | | | |
| 2026-09-13 | pending | n37-002-027 V100 | sm70 | Qwen3.8-27B NVFP4 | same flags without `--decode-graph` (eager control) | | | |

Raw artifacts: `<server log>`.
