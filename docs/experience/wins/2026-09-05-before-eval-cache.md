# Before-eval cache — CPU, 2026-09-05

> Status: pending-remote (CPU gates passed; 27B unmeasured)

## Context

Changing training inputs repeated the unchanged base-model evaluation.

## What Worked

`$TILERL_RUNS/eval-cache/<sha256>.json` stores before metrics, per-problem rows
and the GSM8K/MATH mean length. Hits restore `eval-before.jsonl` and record
`eval_before_cache.cache_hit` and `eval_before_cache.key` in the manifest.
After evaluations always run. Loaded adapters and drafts bypass the cache.

The key includes checkpoint file paths, sizes and nanosecond mtimes, model
config, target/precision, eval-file content hash, slice size, matcher, sampling
settings, thinking mode and the MMLU questions at concurrency 8. Hub sources use
the locally downloaded snapshot. Cache files publish through an atomic rename.
Changing eval length or loaded adapter also changes the run ID.

Rows identify `dataset` as `mmlu` or `gsm8k` (including MATH under the latter
CLI path). Only GSM8K/MATH lengths feed the rollout-length guard.

The tiny CPU gate changes the learning rate between two training runs: eval
sampling calls fall from 4 to 2, with identical before rows. Changing the eval
cap misses; loading an adapter bypasses. A separate check changes a checkpoint
file's mtime without changing its size and requires a different key.

## Rule

Reuse base evaluations only when the evaluated weights, questions and settings agree.

## Results

| date | base commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-05 | 4cf169b | local Mac | cpu | tiny | n/a | n/a | n/a |

Raw artifacts: `/private/tmp/codex-eval-cache-check`,
`/private/tmp/codex-eval-cache-check.log`, `/private/tmp/codex-eval-cache-red.log`.
