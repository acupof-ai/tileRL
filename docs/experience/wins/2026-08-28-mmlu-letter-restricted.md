# MMLU 0-shot at 76.3% once sampling is restricted to the four letters — H20/sm90, 2026-08-28

> Status: Shipped. The 763/1000 below is a slice the code no longer draws — `81e9789` replaced
> shuffle-then-take with `sample`, so today's 1000 questions share 2 with these. Same runner,
> clean main, 2026-09-03: 746/1000 = 74.6%. See
> [errors/2026-09-03-the-mmlu-slice-moved-under-the-number.md](../errors/2026-09-03-the-mmlu-slice-moved-under-the-number.md).

## Context

The 27B NVFP4 decode path was fast but never scored. A first MMLU run read
0.8%: the model answers a multiple-choice prompt with reasoning prose, and the
grader took the first character of that prose. The number measured the grader,
not the model.

## What Worked

`SamplingParams.allowed_ids` — a tuple of token ids the sampler may pick;
logits outside it go to `-inf` before temperature/top-p. MMLU passes the eight
ids for `A/B/C/D` with and without a leading space, which is the lm-eval
convention (argmax over the answer letters), and one greedy token is the answer.

| engine | model | MMLU 0-shot (n=1000) | unparsed | wall |
|---|---|---:|---:|---:|
| tileRL | Qwen3.8-27B-NVFP4 | **76.3%** (763/1000) | 0 | 221 s |

Weakest subjects: econometrics 1/3, public_relations 3/8, college_physics 3/6.

The sglang arm is not a valid comparison yet. Its grammar-constrained decode
(`regex: " ?[ABCD]"`) splits the first token and emits a broken byte before the
letter (`' �A'`), scoring 20.7% with 230 unparsed — a decoder artifact, not
the bf16 model. Re-run scores it the same way tileRL does: unconstrained, one
token, argmax over the letter ids in `top_logprobs`.

`allowed_ids` defaults to `None` and is a single `if` on the sampling path, so
serving throughput is unchanged; it is also the seam constrained-action RL will
use.

## Rule

An eval number on a chat-tuned model measures the *grader* until the answer
space is restricted. Restrict at the sampler (logit mask over the answer token
ids), never by parsing free text — and never through a grammar/FSM you have not
checked byte-by-byte.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-28 | (this) | H20 GPU7 | sm90 | Qwen3.8-27B-NVFP4 | — | — | 76.3% MMLU |

Raw artifacts: `/work/mmlu_tilerl.json`, `/work/mm.log`; runner `scripts/mmlu.py`.
