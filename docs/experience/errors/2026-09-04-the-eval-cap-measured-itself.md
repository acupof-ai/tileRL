# The 39% baseline measured the token cap, not the model

**Date:** 2026-09-04
**Run:** `a4332cbca4fa` (grpo-gsm8k-27b, 100 steps, LoRA rank 16, lr 1e-4)

## Context

The run reported the RL stack's first downstream movement on the 27B:

```
gsm8k_before  195/500 = 39.0%
gsm8k_after   471/500 = 94.2%
mmlu_before   75.2%    mmlu_after 75.0%
tied_group_fraction 0.76   <- groups_untied gate FAILED (threshold 0.5)
```

Read as "+55 points on GSM8K, MMLU flat". That reading is wrong, and the error
is in the control, not the treatment.

## Root cause

Both arms are scored by one `SamplingParams` built once at `cli.py:209` from
`args.max_new_tokens` and handed to `gsm8k_accuracy` twice by the `evals()`
closure. The recipe's default is 256 tokens with thinking off. Measured on the
27B at that setting, n=60:

| | median len | hit cap | correct |
|---|---:|---:|---:|
| greedy (eval path) | 256 | **39/60 = 65%** | 22/60 = 36.7% |
| untruncated (rollout path) | 256 | 44/60 = 73% | 21/60 = 35.0% |

65% of completions never reached an answer. `answer_match` takes the last number
in the completion, so on a truncated derivation it scores whichever intermediate
the model happened to be mid-way through. The 39.0% is not the model's GSM8K
accuracy; it is how often a mid-derivation intermediate coincides with the gold
answer.

Uncapped, with thinking on (n=40, 8192 cap):

| | mean | median | p90 | max | hit cap | correct |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 320.8 | 283 | 527 | 926 | **0/40 = 0%** | **33/40 = 82.5%** |
| untruncated | 318.1 | 285 | 568 | 679 | 0/40 = 0% | 32/40 = 80.0% |

The model answers in 283 median tokens and scores 82.5%. The honest baseline is
~82.5% (n=40, sd ~6), not 39.0%.

The after-arm carries the same defect in the same direction. It was scored under
the same 256 cap, so `94.2% - 39.0%` conflates two effects: getting more answers
right, and learning to finish inside 256 tokens. The second is a real training
effect and not the one that was claimed.

Two further reasons the run does not support its headline:

- `groups_untied` **failed** at 0.76. Three quarters of GRPO groups had identical
  rewards, so `group_advantages` returned zeros and those steps contributed no
  gradient. The number came from a quarter of the intended signal.
- `manifest["artifacts"]` was empty. The adapter that produced 94.2% does not
  exist, so no protocol can be re-run against it by anyone.

Not the cause, but checked and cleared: `gsm8k_train.jsonl` (512 rows) and
`gsm8k_test.jsonl` (500 rows) share 0 prompts. 382 of 500 test answers also
appear as some train answer, which is the small integer range of GSM8K, not
leakage.

## Fix

- Size the rollout KV pool from `max(prompt) + max_new_tokens`. A flat
  `num_blocks=512` is 1024 tokens per row over 8 slots, so any cap above ~1024
  died on `PagedKvPool exhausted` (`kv_cache.py:80`). The uncapped protocol was
  unreachable before this.
- Save the trainable tensors into the run directory before the after-eval, so a
  reported metric always has the weights that produced it.
- Re-run before/after in one uncapped configuration (8192 cap, thinking on), so
  the control measures the model.

## Rule

**A control that saturates its own instrument is not a control.** Before
reporting a delta, check the ceiling the measurement ran into: what fraction of
samples hit the cap, the timeout, the token budget, the retry limit. Where that
fraction is not near zero, the number describes the limit, not the system, and
the delta against it is unearned in the direction that flatters the treatment.

The check is one counter next to the score. `hit-cap 39/60` and `hit-cap 0/40`
took one probe each and settled a question that a 500-question eval could not,
because the eval reports only the score.
