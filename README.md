# tileRL

**Serve and RL-train Qwen3.8-27B (NVFP4) on one Hopper card, in one process.**

sglang refuses this checkpoint on Hopper; vLLM falls back to weight-only Marlin.
tileRL runs fp4 weights × fp8 activations in its own TileLang kernels — **135.5 tok/s
single-stream against sglang's 54.2** — and the engine that samples is the model that trains.

| one H20, Qwen3.8-27B | B=1 decode tok/s | prefill tok/s (512) | MMLU 0-shot |
|---|---:|---:|---:|
| **tileRL**, native NVFP4 + FP8, speculation on | **135.5** | — | 74.2% |
| **tileRL**, native NVFP4 + FP8 | **92.4** | **2887.6** | 74.6% |
| sglang, bf16 (cannot load NVFP4 on Hopper) | 54.2 | — | — |
| sglang, online fp8 | 39.9 | — | — |

B=1 decode is the target because that is the shape a rollout has. Read the rest of
the row honestly: sglang wins prefill (4022 against our 1836 in the one head-to-head
session) and wins B=8 decode (387.0 against 308.6); our 2887.6 comes from a later,
idle-box session and does not belong in a row with theirs. Both sglang arms run a
dequantized bf16 checkpoint that emits garbage, which leaves their decode rates
standing and their MMLU column empty.

**Where the speed comes from, and where it stops.** A tick streams 21.89 GB against
the H20's measured 3312 GB/s copy bandwidth — 1.99 TB/s in 11.0 ms, **60% of the
achievable roof** against sglang-bf16's 70%. The remaining gap is kernel efficiency,
not the memory system. Speculation buys its 1.73x by running **6.11x fewer trunk
forwards** at B=1; at B=8 the base tick already amortises across rows and speculation
lands at 0.928x, so it is a B=1 lever and only that.

## One runtime, not two

GRPO and self-teacher distillation roll out through the same engine and the same
weights — LoRA on the frozen fp4 base, no weight sync, no second stack. One CLI an
agent drives: `train --recipe` → `ledger` → `merge` → `serve`, every run writing a
manifest with its inputs and its gates.

**RL has not yet moved a downstream metric, and the number that said it had was an
artifact.** A run reported GSM8K 39.0% → 94.2%; the 39.0% control was measuring the
256-token cap, not the model — 65% of its completions were cut off before reaching an
answer, so the scorer read a mid-derivation intermediate. Uncapped, the base model
scores **89.6%**, which leaves 10.4 points of headroom, and at n=500 the test resolves
4.8 of them. The honest re-run is the first roadmap gate.
[The full audit.](docs/experience/errors/2026-09-04-the-eval-cap-measured-itself.md)

## Quickstart

```bash
uv sync
TILERL_QWEN38_SOURCE=/path/to/Qwen3.8-27B-NVFP4 uv run tilerl serve --model qwen38-27b
```

OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` — point a client, or Claude
Code, at it.

```bash
python scripts/gsm8k_jsonl.py train gsm8k.jsonl
uv run tilerl train --recipe grpo-gsm8k-27b --data gsm8k.jsonl --eval-gsm8k gsm8k_test.jsonl
uv run tilerl ledger
```

No GPU: `uv run tilerl serve` runs the tiny model on CPU, and `uv run pytest` is the
correctness suite that gates every commit.

## Where things are

- [`docs/roadmap.md`](docs/roadmap.md) — phases and the gate each exits on
- [`docs/design-rl-stack.md`](docs/design-rl-stack.md) — ISO optimizer and merger, the draft head, the ledger
- [`docs/design-engine.md`](docs/design-engine.md), [`docs/design-kernels.md`](docs/design-kernels.md), [`docs/support-matrix.md`](docs/support-matrix.md) — how it is built, per-op status per target
- [`docs/experience/`](docs/experience/) — every measurement, win and dead end, dated
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`AGENTS.md`](AGENTS.md) — the gates a change clears

Every number above sits in a dated entry under `docs/experience/`, gated at ≥ 0.97× the
committed baseline.
