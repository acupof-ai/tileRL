# The eval file was not the level its name and its recipe claimed — 2026-09-05

**Date:** 2026-09-05
**Session:** tilerl-25
**Task:** MATH run 2, `grpo-math-27b` on the 27B

## Context

Run 2 trains on MATH level 5 because P1 showed GSM8K solved at 88.0% base. I
built four JSONL files locally and pushed them to the pod, which has neither
`datasets` nor network. `grpo-math-27b`'s comment said "LEVEL 5 ONLY" and cited
"level 5 alone 45.8% (11/24)".

The run's before-arm came back **400/500 = 80.0%**. Against a 45.8% expectation
that is a 34-point gap, and I checked the file rather than the model.

## Root Cause

`math_test500.jsonl` is **levels 3-5 mixed**. Joined against the parquet shards
by problem text:

| file | n | levels |
|---|---:|---|
| `math_tr_5.jsonl` (training) | 2304 | Level 5: 2304 |
| `math_test500.jsonl` (eval) | 500 | **Level 3: 165, Level 4: 157, Level 5: 178** |

The training data is correct. The eval file was built without the `--level 5`
filter that the training files got, so 80.0% is a 3/4/5 average and was never
comparable to the 45.8% the recipe cites — that figure came from a 24-problem
hand sample of level 5, a different set entirely.

**Neither number was wrong; the comparison was.** The recipe comment put them in
the same sentence, which is what made the gap look like a model result.

The file itself carries no level column. It was built by a throwaway script on the Mac
(the pod has neither `datasets` nor network, so `scripts/math_jsonl.py` could not run
there), and that script wrote only `prompt` and `answer` — so nothing downstream could
have caught it. The name was the only claim about the contents, and a name is not a
measurement.

The same build produced the "mean completion on level 5 is 1038 tokens" line.
That was the mixed file too; this run measures **1029** on it (514554 / 500).
The rollout-cap decision it justified is unaffected — 1029 against a 512 cap
truncates just as surely — but the number was labelled with a level it did not
come from.

## Fix

The recipe comment now states what each file actually is, and the numbers carry
the set they were measured on. The delta this run reports stays valid: both arms
score the same file, so the paired comparison holds. It answers "did level-5
training help on levels 3-5" rather than "on level 5", and says so.

A level-5-only eval is an eval-only run afterwards; the before-arm cache (#134)
makes it cheap.

## Rule

**A data file's name is a claim about its contents, and joining it back to the
source is one query.** Verify the filter actually applied before quoting an
accuracy against an expectation from a different set. When a measured number
misses expectation by 34 points, the file is a cheaper hypothesis than the model.

And a derived dataset should carry the field it was filtered on. Both builders read
`level`, filtered on it, then dropped it — so the one fact that distinguishes these
four files existed nowhere in their output. `scripts/math_jsonl.py` now writes it; the
throwaway one is gone.
