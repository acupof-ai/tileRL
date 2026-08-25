# Decode graph per batch-size bucket (B>1 graph replay) — cuda/sm90, 2026-08-25

> Status: Shipped

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

Measured on slice4 (4 layers) in a fully-idle H20 window (all 8 GPUs at 0%
outside our own run), `scripts/bench_batch_decode.py --fuse`:

| B | eager ms/tick | graph ms/tick | graph agg tok/s |
|---|---:|---:|---:|
| 1 | 5.519 | 1.811 | 552 |
| 2 | 8.067 | 4.523 | 442 |
| 4 | 8.320 | 4.709 | 849 |
| 8 | 9.840 | 7.091 | 1128 |

Graph B=1 (1.811ms) reproduces the shipped single-stream baseline (1.734ms,
`2026-08-25-projection-fusion-decode.md`) — replay is correct at every B with
no capture fallback. Graph beats eager 3.0x at B=1, 1.8x at B=2/4, 1.4x at
B=8: the Python launch tax amortizes with batch as predicted, but replay
still wins at every B. Tick time grows 3.9x from B=1 to B=8 for 8x the
tokens — weights are read once per tick regardless of B. Extrapolated to the
full 27B with the final-bench ratio (11.17x, optimistic — it counts lm_head
and fixed costs once, not per-B): B=8 aggregate ~101 tok/s vs the 80 target.

## Rule

A captured graph amortizes launch cost over the whole batch: eager B>1 decode
is launch-bound, not bandwidth-bound, so graph replay — not bigger kernels —
is the lever once weights are read once per tick regardless of B.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | 61c8e84 | H20 pod | cuda/sm90 | Qwen3.8-27B NVFP4 slice4, B=1 graph | — | 1.811 | 552 |
| 2026-08-25 | 61c8e84 | H20 pod | cuda/sm90 | Qwen3.8-27B NVFP4 slice4, B=8 graph | — | 7.091 | 1128 (aggregate) |

Raw artifacts: pod `/work/bgraph_eager.log`, `/work/bgraph_graph.log`
(`scripts/bench_batch_decode.py --fuse [--decode-graph]`,
`scripts/_pod_bgraph_bench.sh`).
