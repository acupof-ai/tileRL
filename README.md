# tileRL

**Serve and RL-train Qwen3.8-27B (NVFP4) on one Hopper card, in one process.**

sglang refuses this checkpoint on Hopper; vLLM falls back to weight-only Marlin.
tileRL runs fp4 weights × fp8 activations in its own TileLang kernels — **135.5 tok/s
single-stream against sglang's 54.2** — and the engine that samples is the model that trains.

| one H20, Qwen3.8-27B | B=1 decode tok/s | prefill tok/s | MMLU 0-shot |
|---|---:|---:|---:|
| **tileRL**, NVFP4 + FP8, speculation on | **135.5** | — | 74.2% |
| **tileRL**, NVFP4 + FP8 | **92.4** | **2887.6** | 74.6% |
| sglang, bf16 (cannot load NVFP4 on Hopper) | 54.2 | — | — |
| sglang, online fp8 | 39.9 | — | — |

B=1 decode is the target because that is the shape a rollout has. sglang wins prefill
and wins B=8 decode; both its arms run a dequantized bf16 checkpoint that emits
garbage, which is why their MMLU column is empty. Speculation is a B=1 lever only —
at B=8 it lands at 0.928x.

## One runtime, not two

GRPO and self-teacher distillation roll out through the same engine and the same
weights — LoRA on the frozen fp4 base, no weight sync, no second stack.
`train --recipe` → `ledger` → `merge` → `serve`, every run writing a manifest with its
inputs and its gates.

**RL has not yet moved a downstream metric, and the number that said it had was an
artifact.** A run reported GSM8K 39.0% → 94.2%; the 39.0% control was measuring the
256-token cap, not the model. Uncapped the base scores **89.6%**, leaving 10.4 points
of headroom against 4.8 resolvable at n=500. The honest re-run is the first roadmap
gate. [The audit.](docs/experience/errors/2026-09-04-the-eval-cap-measured-itself.md)

## Quickstart

```bash
uv sync
TILERL_QWEN38_SOURCE=/path/to/Qwen3.8-27B-NVFP4 uv run tilerl serve --model qwen38-27b
```

OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` — point a client, or Claude
Code, at it. No GPU: `uv run tilerl serve` runs the tiny model on CPU.

```bash
python scripts/gsm8k_jsonl.py train gsm8k.jsonl
uv run tilerl train --recipe grpo-gsm8k-27b --data gsm8k.jsonl --eval-gsm8k gsm8k_test.jsonl
uv run tilerl ledger
```

## Where things are

[`docs/roadmap.md`](docs/roadmap.md) — phases and gates ·
[`docs/design-rl-stack.md`](docs/design-rl-stack.md) — ISO, the draft head, the ledger ·
[`docs/design-engine.md`](docs/design-engine.md) ·
[`docs/design-kernels.md`](docs/design-kernels.md) ·
[`docs/support-matrix.md`](docs/support-matrix.md) — per-op status per target ·
[`docs/experience/`](docs/experience/) — every measurement, win and dead end, dated ·
[`AGENTS.md`](AGENTS.md) — the gates a change clears

Every number above sits in a dated entry under `docs/experience/`, gated at ≥ 0.97× the
committed baseline. `uv run pytest` is the suite that gates every commit.
