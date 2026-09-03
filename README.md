# tileRL

Serve and RL-train Qwen3.8-27B (NVFP4) on one Hopper card, in one process.

- **Native NVFP4 on H20.** sglang refuses the NVFP4 checkpoint on Hopper and
  vLLM falls back to weight-only Marlin; tileRL runs fp4 weights × fp8
  activations in its own TileLang kernels, and single-stream decode is the
  fastest measured on this card.
- **The engine that samples is the model that trains.** GRPO and self-teacher
  on-policy distillation roll out through the same engine and the same
  weights, LoRA on the frozen fp4 base — no weight sync, no second stack.
- **One CLI an agent drives.** `train --recipe` → `ledger` → `merge` →
  `serve`; every run writes a manifest with its inputs and gates.

| one H20, Qwen3.8-27B | B=1 decode tok/s | prefill tok/s | MMLU 0-shot |
|---|---:|---:|---:|
| tileRL, native NVFP4 + FP8 | **92.4** | **2887.6** | 74.6% |
| sglang, bf16 (cannot load NVFP4 on Hopper) | 54.2 | — | — |
| sglang, online fp8 | 39.9 | — | — |

The card, not the field: a tick reads 20.4 GB, and the H20's measured copy
bandwidth is 3312 GB/s, so unspeculated B=1 decode roofs at 162 tok/s. 92.4 is
57% of that — the remaining gap is kernel efficiency, not the memory system.
sglang is ahead at batch ≥ 8; that is not the target, rollout is single-stream
decode.

Speculative decode is the lever that passes the roofline, because one weight
pass verifies a block. Both halves are measured and neither is wired to the
other yet: a width-8 verify tick costs 2.4× a width-1 tick at B=1 and 1.8× at
B=8, and the DFlash2 block drafter lands 5.8 of 8 tokens. Their product is not
a throughput number until the drafter runs on the tick, and this table will not
carry one until it does.

RL on the 27B has not yet moved a downstream metric; that run is the first
roadmap gate. Every number above sits in a dated entry under
`docs/experience/`, gated at ≥ 0.97× the committed baseline.

## Quickstart

```bash
uv sync
TILERL_QWEN38_SOURCE=/path/to/Qwen3.8-27B-NVFP4 uv run tilerl serve --model qwen38-27b
```

OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` — point a client, or Claude Code, at it.

```bash
python scripts/gsm8k_jsonl.py train gsm8k.jsonl
TILERL_QWEN38_SOURCE=/path/to/Qwen3.8-27B-NVFP4 uv run tilerl train --recipe grpo-gsm8k-27b --data gsm8k.jsonl --eval-gsm8k gsm8k_test.jsonl
uv run tilerl ledger
```

No GPU: `uv run tilerl serve` runs the tiny model on CPU, and `uv run pytest`
is the correctness suite that gates every commit.

## Where things are

- [`docs/roadmap.md`](docs/roadmap.md) — phases and the gate each exits on
- [`docs/design-rl-stack.md`](docs/design-rl-stack.md) — ISO optimizer and merger, the draft head, the ledger
- [`docs/design-engine.md`](docs/design-engine.md), [`docs/design-kernels.md`](docs/design-kernels.md), [`docs/support-matrix.md`](docs/support-matrix.md) — how it is built, per-op status per target
- [`docs/experience/`](docs/experience/) — every measurement, win and dead end, dated
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`AGENTS.md`](AGENTS.md) — the gates a change clears
