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

**The same checkpoint also runs on a V100** — sm70, no bf16, no fp8 hardware path, two
generations before NVFP4 existed. 43.2 tok/s wall over the network, 22 GiB. It is not
the perf target; it is the evidence that the kernels are not bound to one arch.

```
curl http://10.37.2.27:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"hi"}]}'
```

## One runtime, not two

GRPO and self-teacher distillation roll out through the same engine and the same
weights — LoRA on the frozen fp4 base, no weight sync, no second stack.
`train --recipe` → `ledger` → `merge` → `serve`, every run writing a manifest with its
inputs and its gates.

**The thinking cap: training under a tight token budget buys accuracy *and* economy.**
Cap the rollout at 256 tokens, score correctness only, then measure uncapped — the
policy finds the shorter path to the same answer.

| GSM8K, uncapped, n=500 | accuracy | total tokens | tokens / correct |
|---|---:|---:|---:|
| base | 89.6% | 157,601 | 351.8 |
| after 100 GRPO steps | **94.8%** | **121,642** | **256.6** |
| | +5.2 pts, p=0.002 | **−22.8%** | **−27.1%** |

It transfers to tasks the adapter never saw — tokens fall 22.0% on MMLU, 18.9% on
ARC-Easy, 22.9% on PIQA, with no measurable accuracy change at n=100. Output tokens
are the serving bill, so this is a win in the units the product is sold in.

The control moved the claim twice. Retrained at 2048, the policy solves 96.6% of
training prompts and **92% of GRPO steps carry no gradient** — the tight cap was
holding the task hard enough to keep groups mixed. But that arm still reached
**96.4% on GSM8K, off 8 gradient steps** — so the cap is not what gets the
policy to ~95%. What it demonstrably buys is sample efficiency and the token
cut; whether it also buys accuracy this run cannot say, since ranking two arms
1.6 points apart needs ~2,600 questions each and we ran 500.

[The result](docs/experience/wins/2026-09-04-the-thinking-cap.md) ·
[the control that reinterpreted it](docs/experience/wins/2026-09-04-the-cap-was-the-gradient.md) ·
[why the first number was wrong](docs/experience/errors/2026-09-04-the-eval-cap-measured-itself.md)

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
