# RL on the 27B gets a real task, a real reward, and a before/after gate — pending-remote, 2026-09-02

> Status: pending-remote — code shipped and gated on the tiny model; the 27B
> run is blocked until the pod frees up (all eight H20s hold another job's
> pretrain, 73 GB each, `/work/aupai`).

## Context

`tilerl train --rl` on the 27B had run once (`wins/2026-08-29-grpo.md`) — on
random token prompts with a demo reward (token id below a threshold). That
proves the loop moves weights; it says nothing about learning. The roadmap's P1
gate is a downstream metric, and nothing in the tree could produce one.

## What Worked

- `--data` JSONL `{prompt, answer}`: prompts go through the checkpoint
  tokenizer as one ChatML user turn (`tilerl.tokenizer.render_chat`, the same
  renderer the server uses); the reward is exact match on the last number of
  the decoded completion, the GSM8K convention. `scripts/gsm8k_jsonl.py` dumps
  the dataset (reachable from the pod via `hf-mirror.com`).
- Rollouts stop at `<|im_end|>` and skip `<think>` by default
  (`--think-budget 0`, the server's `reasoning_effort=none` mechanism);
  `grpo_loop` / `opd_loop` take one `SamplingParams` template instead of loose
  kwargs.
- `--eval-mmlu N` scores the same engine before the loop (LoRA B is zero at
  init, so that is the base) and after. `tilerl.eval` holds the MMLU slice and
  scorer; `scripts/mmlu.py` reuses it.
- `grpo_loop` returns seconds per step (rollout + update) — the number an RL
  runtime is priced on, and what `scripts/rl_compare.sh` reads.
- The tokenizer moved out of `server.py` so training does not import FastAPI.

Gates on the tiny model, CPU: `tests/test_rl.py::test_train_cli_real_task`
(JSONL → tokenizer → ChatML → reward → one step → per-step seconds) and
`test_last_number`.

## Rule

A training path without a task and a metric is a smoke test. Ship the task,
the reward and the before/after gate together, or the "trains" claim is
unbacked.

## Results

| date | commit | machine | target | model | task | steps | reward first→last | MMLU before→after | s/step |
|---|---|---|---|---|---|---|---|---|---|
| pending-remote | | H20 GPU 7 | cuda | Qwen3.8-27B-NVFP4 | GSM8K, group 8, 256 tok | 20 | | | |

Run: `scripts/pod_sync.sh run rl_compare 'bash scripts/rl_compare.sh'`.
