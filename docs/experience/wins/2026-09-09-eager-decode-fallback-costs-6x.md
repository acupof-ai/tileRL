# Eager decode fallback costs 6.0x wall — now visible on /health — H20, 2026-09-09

> Status: Shipped

## Context

`stats()` is the `/health` payload and takes the engine's step lock, so a field there
is on a live path. A decode-graph capture failure flips the engine to eager decode —
correct output, 6x wall — and was invisible: `stats()` did not report the runtime
flag. This entry prices the fallback so the visibility fix has its cost on record.

## What Worked

Same workload, same sha (f80e894), B=1, 50 GSM8K questions, card 6, 0 compiles:

| arm | wall | note |
|---|---:|---|
| decode graph on | 211.8s | the serving build |
| eager fallback (capture failed) | 1269.4s | 6.0x, correct output |

The fix adds `decode_graph` (runtime state) to `stats()`; `/health` now shows
`decode_graph: false` after a fallback. The field itself is one dict entry under the
lock `stats()` already takes — no measured cost of its own.

## Rule

A silent performance fallback needs a price on record before it can be called fixed:
here 1269.4s vs 211.8s. The post-mortem is
[errors/2026-09-09-graph-capture-fell-back-silently.md](../errors/2026-09-09-graph-capture-fell-back-silently.md).

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-09 | f80e894 | H20 card 6 | cuda/sm90 decode-graph | Qwen3.8-27B-NVFP4 | — | — | 50q in 211.8s wall |
| 2026-09-09 | f80e894 | H20 card 6 | cuda/sm90 eager fallback | Qwen3.8-27B-NVFP4 | — | — | 50q in 1269.4s wall (6.0x) |

Raw artifacts: `/work/accpf3.log` (graph) and `/work/accpf.log` (fallback) on the pod.
