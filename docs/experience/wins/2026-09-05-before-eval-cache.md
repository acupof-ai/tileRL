# Before-eval cache — CPU, 2026-09-05

> Status: pending-remote (CPU gates passed; 27B unmeasured)

## Context

Changing training inputs repeated the unchanged base-model evaluation.

## What Worked

`$TILERL_RUNS/eval-cache/<sha256>.json` stores before metrics, per-problem rows
and the GSM8K/MATH mean length. Hits restore `eval-before.jsonl` and record
`eval_before_cache.cache_hit` and `eval_before_cache.key` in the manifest.
After evaluations always run. Loaded adapters and drafts bypass the cache.

The key includes a `weights` entry that is never absent by omission — the 27B's
checkpoint file paths, sizes and nanosecond mtimes; `tiny`'s explicit `None`,
because `build_random` is a pure function of `--seed` and the seed reaches the key
through the sampling params; and for any other model **no key at all**, so a base
the key cannot identify gets no cache rather than one that silently omits it.
Alongside it: model config, `--tp`, target/precision, eval-file content hash, slice
size, matcher, sampling settings, thinking mode and the MMLU questions at
concurrency 8. `--tp` reaches the key twice over, since `cfg` at that point is
already `tp_config(cfg, tp)`; the explicit field is there so a change in call order
cannot remove it. Cache files publish through an atomic rename.

## Not established

- **Existing run ids change.** `eval_max_new_tokens` and `load_adapter` join the id
  inputs, so every previously written manifest is unreachable for idempotency — a
  rerun of P1's exact flags retrains rather than returning `1fa1e58388a2`. Checked:
  that is the only manifest on the pod, so nothing in flight is affected.
- **The hub branch is unexercised.** `_QWEN38_SOURCE` is a real directory on the pod,
  so `snapshot_download(local_files_only=True)` never runs there and no test covers
  it. A re-download would move the snapshot path and miss.
- `--max-think-tokens` cannot move the key on the tiny path: `sampling()` drops it
  when `thinking` is None, so both 0 and 64 give the same params. Covered on the 27B
  by the separate `thinking` field; unexercised by any gate.

Rows identify `dataset` as `mmlu` or `gsm8k` (including MATH under the latter
CLI path). Only GSM8K/MATH lengths feed the rollout-length guard.

The tiny CPU gate changes the learning rate between two training runs: eval
sampling calls fall from 4 to 2, with identical before rows. Changing the eval
cap misses; loading an adapter bypasses. A separate check changes a checkpoint
file's mtime without changing its size and requires a different key. A third
asserts what the key must and must not cover — `--seed`, `--eval-n`,
`--eval-max-new-tokens` and `--temperature` move it, `--lora-rank`, `--steps` and
`--lr` do not, and an unknown model returns None. **Each of the three new
assertions has a control that fails**: dropping the explicit `tp` field, making the
unknown-model branch return a key, and omitting `weights` (caught by the mtime
check, not this one).

## Rule

Reuse base evaluations only when the evaluated weights, questions and settings agree.

## Results

| date | base commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-05 | 4cf169b | local Mac | cpu | tiny | n/a | n/a | n/a |

Raw artifacts: `/private/tmp/codex-eval-cache-check`,
`/private/tmp/codex-eval-cache-check.log`, `/private/tmp/codex-eval-cache-red.log`.
