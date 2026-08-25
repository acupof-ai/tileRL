# Decode graph per batch-size bucket (B>1 graph replay) — cuda/sm90, 2026-08-25

> Status: pending-remote

## Context

The captured decode graph (`_DecodeGraph`) was B=1-only: a pure-decode tick
with one request replayed ~900 baked kernels, but B>1 fell back to eager,
paying ~5ms/tick of fixed Python dispatch (900 launches × ~5µs) on top of the
kernel work. The continuous-batching scheduler (see
`2026-08-25-engine-scheduler-batch.md`) made concurrent decode batches the
common case, and the eager B=8 bench showed the tick time flat from B=2 to
B=8 — launch-bound, not bandwidth-bound — so the graph had to cover B>1
before aggregate decode throughput could approach the 80 tok/s target.

## What Worked

`_DecodeGraph` is parameterized by batch size and captured lazily per
batch-size bucket: `_run_decode_graph` takes the full decode list, looks up
the graph for `len(reqs)` (capturing on first tick of that size), and replays
it with per-request logits sliced back out. Capture failure for any bucket
flips the graph off and falls back to eager, same as B=1. Mixed ticks (decode
rows + a prefill chunk) still run eager — the shapes vary per tick there.

## Rule

A captured graph amortizes launch cost over the whole batch: eager B>1 decode
is launch-bound, not bandwidth-bound, so graph replay — not bigger kernels —
is the lever once weights are read once per tick regardless of B.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | pending | H20 pod | cuda/sm90 | Qwen3.8-27B NVFP4 slice4, B=1/2/4/8, eager vs graph | pending | pending | pending |

Raw artifacts: pending (`scripts/bench_batch_decode.py --fuse [--decode-graph]`).
