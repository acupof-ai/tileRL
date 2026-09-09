# `--deterministic`: eager decode for bitwise-reproducible A/B — H20 sm90, 2026-09-09

> Status: pending-remote (PR open; `feacccb`). The graph stays the default —
> this is an A/B switch, not a default flip.

## Context

The captured decode graph makes same-config training runs diverge across
processes: two clean runs (same sha, seed, card) differ in 16/24 rollout rows,
so every A/B comparison is unattributable. The forward kernels themselves are
bitwise deterministic (`scripts/fwd_determinism.py`: 0/8 diff, fixed layout;
attention reduces by logical position, not physical block number), and the
capture+replay is deterministic in-process (`scripts/capture_determinism.py`:
0/8). The divergence is cross-process and graph-specific; its source is
unlocated (the uninitialized-pool-memory candidate was killed by the poison
test, `scripts/poison_pool_determinism.py`: 0/8 with the pool filled 0xFF vs
0x00). See `errors/2026-09-09-decode-graph-run-to-run-nondeterminism.md`.

## What Worked

`--deterministic` sets `decode_graph=False` on the training rollout path
(`cli.py`), so rollouts run eager. Two same-config runs then differ in 0/24
rollout rows (replicated twice), vs 16/24 with the graph on. A/B runs become
bitwise reproducible.

The cost is rollout throughput: eager decode is ~6x slower than the captured
graph (14.7 vs 94.6 tok/s). The graph stays the default for throughput runs;
the flag is for attributing a trajectory difference to a config change.

## Rule

For any A/B comparison that must be attributable, run both arms with
`--deterministic`. The graph's cross-process nondeterminism is unfixed, so a
same-config pair run with the graph on cannot be read as a config effect.

## Results

| date | commit | machine | target | model | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|
| 2026-09-09 | feacccb | H20 | sm90 | Qwen3.8-27B-NVFP4 (graph, default) | 10.6 | 94.6 |
| 2026-09-09 | feacccb | H20 | sm90 | Qwen3.8-27B-NVFP4 (eager, `--deterministic`) | 68.0 | 14.7 |

Determinism (rollout rows differing across two same-config runs): graph on
16/24, `--deterministic` 0/24 (×2). Raw artifacts: `/work/reprodet.log`,
`/work/reprong.log`, `/work/fwddet.log`, `/work/capdet4.log`,
`/work/poison2.log`.
