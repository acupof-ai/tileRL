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

## What survives the defect: MMLU

`mmlu_score` (`eval.py:71`) builds its own `SamplingParams(max_new_tokens=1, ...)`
and never touches the shared `params` object, so the answer comes off the prefill
and the cap cannot reach it. MMLU is the one metric in these runs the defect does
not corrupt:

| run | MMLU before | after | delta |
|---|---:|---:|---:|
| `a4332cbca4fa` GRPO | 75.2% | 75.0% | **-0.2** |
| `41e3301e22a5` OPD | 75.2% | 81.4% | **+6.2** |

At n=1000, treating the arms as independent (conservative -- they are paired on
the same questions and seed, so the real test is sharper), the 95% interval on a
difference is +/-3.8 points. GRPO's -0.2 is no movement; OPD's +6.2 is 3.2 sd and
is real.

Read this correctly: **MMLU is an off-task holdout for GSM8K training, so flat is
the pass condition, not a failure.** `mmlu_holds` exists to catch collateral
damage, and GRPO caused none. A flat MMLU is evidence the run did not break the
model. It is not evidence about whether GSM8K improved, in either direction.

## The same class, second instrument: `reward_rises`

`grpo_loop` takes `prompts[step % len(prompts)]` (`train.py:265`) -- positional,
not sampled. With 512 training rows and 100 steps no prompt repeats, so the
windows `cli.py:301-302` average over are **disjoint prompt sets**:

- `reward_first` = steps 1-25 = prompts 0-24
- `reward_last` = steps 76-100 = prompts 75-99

`0.705 -> 0.94` therefore compares one policy on one set of questions against a
different policy on a different set of questions. Policy improvement and prompt
difficulty are summed and cannot be separated. The `reward_rises` gate passed on
that comparison.

The code comment at `cli.py:298` shows the per-step version of this was known --
"per-step reward moves with the sampled prompt, so two single steps compare two
draws, not two policies" -- and windows were the response. Averaging reduces the
variance; it does not remove the offset, because the two windows were never the
same questions.

Ruled out first, so this is the remaining explanation for rollout reward 0.705
against a 39.0% eval at the same cap: train and test are indistinguishable in
difficulty (median prompt 228 vs 222 chars, 4 vs 3 sentences), and greedy vs
untruncated differ by 1.7 points (36.7% vs 35.0%, n=60).

**Gate status after this audit:**

| gate | valid? |
|---|---|
| `mmlu_holds` | yes -- `max_new_tokens=1`, the cap cannot reach it |
| `groups_untied` | yes -- and it FAILED at 0.76, unremarked |
| `gsm8k_improves` | was broken by the cap; valid once both arms run uncapped |
| `reward_rises` | **confounded** -- disjoint prompt sets |

Two of four were reporting on something other than what their name says.
`gsm8k_before`/`after` is unaffected by this one: it scores the same fixed 500
test questions before and after, which is the comparison `reward_rises` only
appears to make.

## Fix

- Size the rollout KV pool from `max(prompt) + max_new_tokens`. A flat
  `num_blocks=512` is 1024 tokens per row over 8 slots, so any cap above ~1024
  died on `PagedKvPool exhausted` (`kv_cache.py:80`). The uncapped protocol was
  unreachable before this.
- Save the trainable tensors into the run directory before the after-eval, so a
  reported metric always has the weights that produced it.
- Re-run before/after in one uncapped configuration (8192 cap, thinking on), so
  the control measures the model.

## The reading, fixed before the number

Registered before the re-run reports, so the interpretation is not chosen after
seeing the result. Both arms score `eval_n=500` (`recipes.py`) in one uncapped
configuration. Against an 82.5% base at 80% power, alpha 0.05 two-sided, a
two-proportion test at n=500 per arm resolves **6.2 points**; 5, 10 and 15
points need 797, 168 and 59 per arm respectively.

- gain >= 6.2 points: confirmed.
- gain below 6.2 points: **underpowered, not refuted.** "No significant
  difference" at this n does not mean training did nothing -- a real 4-point
  gain lands here, and separating that from zero needs n ~ 1250 per arm.
- The uncapped before-arm is the only valid comparison partner. The 39.0%
  figure is not a baseline and no delta may be quoted against it.

## Rule

**A control that saturates its own instrument is not a control.** Before
reporting a delta, check the ceiling the measurement ran into: what fraction of
samples hit the cap, the timeout, the token budget, the retry limit. Where that
fraction is not near zero, the number describes the limit, not the system, and
the delta against it is unearned in the direction that flatters the treatment.

The check is one counter next to the score. `hit-cap 39/60` and `hit-cap 0/40`
took one probe each and settled a question that a 500-question eval could not,
because the eval reports only the score.
