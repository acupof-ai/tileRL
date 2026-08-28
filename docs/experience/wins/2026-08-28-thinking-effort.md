# Thinking effort: closing the reasoning block on the caller's terms — H20/sm90, 2026-08-28

> Status: Shipped

## Context

A chat-tuned model decides for itself how long to reason. That is the wrong
owner for the decision in two places that matter here: an eval, where the
reasoning is not the answer (it took MMLU to 0.8% before `allowed_ids`), and an
RL rollout, where thinking length is the cost being budgeted.

## What Worked

`SamplingParams.thinking_budget` + `end_think_ids`: once the budget of generated
tokens is spent, the engine emits the end-of-think ids instead of sampling, and
stops as soon as the block is closed (by the model or by the force). The ids
come from the caller, so the engine never sees a tokenizer. The server maps
OpenAI's `reasoning_effort` — none=0, minimal=128, low=512, medium=2048,
high=8192.

Verified on the real Qwen3.8-27B-NVFP4, prompt `What is 17 * 23?`:

| effort | budget | completion |
|---|---:|---|
| `none` | 0 | `'</think>\n\nTo calculate $17 \times 23$ … 340 …'` — no reasoning at all |
| `minimal` | 32 | 32 tokens of reasoning, forced close, then `17 × 23 = **391**` |

The forced close costs one `if` per sampled token when unset.

## Rule

Cap reasoning at the sampler by emitting the end-of-think ids, not by
post-processing the text and not by prompt instructions: it is the only place
where the budget is actually enforced, and it is the same seam a rollout needs
to bound thinking cost.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-28 | 78f54ca | H20 gpu7 | sm90 | Qwen3.8-27B-NVFP4 | — | — | unchanged (one branch) |

Raw artifacts: `/work/chain2.log`; probe `scripts/think_effort.py`.
