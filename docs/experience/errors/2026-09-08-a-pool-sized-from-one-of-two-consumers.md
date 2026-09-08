# A pool sized from one of its two consumers

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55

## Context

`main` went red at 3ab79b1 (#318) on both runners:

```
tests/test_ledger.py::test_train_cli_writes_manifest_and_is_idempotent FAILED
RuntimeError: PagedKvPool exhausted: all 137 blocks in use
src/tilerl/kv_cache.py:149
```

#318 sized the training engine from `--group` instead of three literal `8`s, which is
correct and was measured (B=16 at 30.13 GiB peak). It is not where the bug is.

## Root cause

The training KV pool is computed from the ROLLOUT's terms only:

```python
ctx = max(max(map(len, prompts)) + args.max_new_tokens + 64, 1024)
num_blocks = -(-ctx // BLOCK_TOKENS) * rollout_batch + 8
```

The engine has a second consumer. The eval arms submit `_EVAL_CONCURRENCY` (8) rows at
`--eval-max-new-tokens` (2048 by default) — 512x the 4-token rollout cap the tests use — and
MMLU's prompts reach 515 tokens against GSM8K's 183. None of those three terms is in the
expression. `--group` was one of the literal 8s, so before #318 the rollout term happened to
be `1024/16 * 8 = 520` blocks and covered the eval by accident at the tests' sizes.

**#318 did not introduce this. It lowered the accidental cover.** Measured, one process per
arm, the same `--eval-n 8` case on the pre-#318 `cli.py`:

| arm | eval rows | pre-#318 | post-#318 |
|---|---:|---|---|
| `--group 2` | 2 | pass (520 blocks) | **exhausted, 137** |
| `--group 8` | 8 | **exhausted, 521** | **exhausted, 521** |
| `--group 16` | 8 | — | **exhausted, 1033** |

The failure exists at every group size once 8 rows are scored; `--group 2` merely brought the
boundary down to the 2 rows this test uses. A bisect that stopped at "the last green commit"
would have reverted the wrong change and left the defect at `--eval-n 100`, which is the
default and what every real run passes.

## Fix

Size the pool from the max over both consumers, and make the eval's width a named constant
read by both the sizing and the three call sites that were passing `concurrency=8`:

```python
rollout_ctx = prompt_max + args.max_new_tokens + 64
eval_ctx = max(prompt_max, 515 if args.eval_mmlu else 0) + args.eval_max_new_tokens + 64
eval_batch = _EVAL_CONCURRENCY if (eval_rows or args.eval_mmlu) else 0
blocks = max(-(-rollout_ctx // BLOCK_TOKENS) * rollout_batch,
             -(-eval_ctx  // BLOCK_TOKENS) * eval_batch) + 8
```

`test_the_pool_is_sized_for_the_eval_arm_not_only_the_rollout` asserts the recorded pool
covers the eval's demand, computed from the same flags. It uses 8 eval rows because at 2 the
eval fits under the rollout's pool at group >= 4 and the test would go green over the bug.

Its negative control needed a second attempt. Reverting `cli.py` wholesale made the test fail
on `ImportError: cannot import name '_EVAL_CONCURRENCY'` — red, but by a route that has
nothing to do with pool sizing. Mutating only the sizing line fails on
`PagedKvPool exhausted: all 25 blocks in use`, which is the reason under test.

## Rule

When one resource serves two callers, its size is a max over both, and a sizing expression
that mentions only one of them is wrong even while it passes. The passing case is not
evidence: `--group` being one of the hardcoded widths made the rollout term cover the eval by
coincidence, and the arithmetic never referred to the eval at all.

A regression's bisect point is where the symptom appeared, not necessarily where the defect
is. Before attributing it, run the failing case against the parent commit with the OTHER
parameters at their real values — here `--eval-n 8` instead of the test's 2, which fails on
both sides of the bisect and says the new commit is not the cause.
