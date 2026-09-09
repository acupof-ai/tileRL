# tileRL

**Serve and RL-train Qwen3.8-27B (NVFP4) on one Hopper card, in one process.**

sglang refuses this checkpoint on Hopper. tileRL runs it in its own TileLang kernels —
**92.4 tok/s single-stream against sglang's 54.2 on the same shape** — and the engine
that samples is the model that trains.

| one H20, Qwen3.8-27B | workload | B=1 decode tok/s | prefill tok/s |
|---|---|---:|---:|
| **tileRL**, NVFP4 + FP8 | d512, 64 out | **92.4** | **2689.8** |
| sglang, bf16 (cannot load NVFP4 on Hopper) | d512, 64 out | 54.2 | 2512 |
| sglang, online fp8 | d512, 64 out | 39.9 | **4022** |
| **tileRL**, W=8 block speculation | 200 GSM8K, 512 out | **126.5** | — |
| **tileRL**, same arm, speculation off | 200 GSM8K, 512 out | 79.5 | — |

B=1 decode is the target because that is the shape a rollout has. sglang's fp8 arm still
wins prefill; its bf16 arm no longer does. Weights are fp4 against **bf16** activations
at B=1 — the fp8-activation path is the M > 1 kernel, so it carries prefill and batched
decode, not the single-stream number this table leads with.

Accuracy is not a decode workload and is not in that table: these weights score **74.6%
MMLU 0-shot**. Both sglang arms run a dequantized bf16 checkpoint that emits garbage, so
there is no accuracy number to compare them on — the rows above are a kernel comparison
only.

**Read the workload column before comparing rows.** Only the first three are the same
shape, and only they are comparable to sglang. The speculation pair ran 200 real GSM8K
problems, so 126.5 is read against its own 79.5 base (**1.591x**) and never against
54.2. Speculation is a B=1 lever: at B=8 it lands at 0.928x.

This table read **135.5** for the speculation row until 2026-09-09. That number was
measured 2026-09-03 on H20 gpu7; re-measured today on the current sha, same workload, on
a different H20 (card 6), the arm reads **126.5** warm and 122.4 cold. The base arm reproduces (79.5 warm, 78.2 cold,
against 78.4 recorded) and so does the drafter (6.19 of 8 blocks accepted, against
6.14) — what did not reproduce is throughput, with the algorithm unchanged. We publish
what we can reproduce today; 135.5 stands in its dated entry, not here.
[The 2026-09-03 arm](docs/experience/wins/2026-09-03-batched-selector-walk.md)

**The same checkpoint also runs on a V100** — sm70, no bf16, no fp8 hardware path, two
generations before NVFP4 existed. 50.0 tok/s decode-only, 46.3 wall measured from the
pod with RTT outside the window; 19 GB of weights off disk. It is not the perf target;
it is the evidence that the kernels are not bound to one arch.

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

The cap is doing more work in that result than a reward would. On MATH level 5,
where the task does not exhaust, a correctness-only reward lengthens rollouts
until they hit the cap and every group ties at the floor — run 2 was killed at
step 45 of 100. So the token cut above is a property of training under a tight
cap on a task the model largely solves, not a general result about the reward.
[The run](docs/experience/errors/2026-09-06-the-rollouts-grew-into-the-cap.md)

## Quickstart

```bash
uv sync
TILERL_QWEN38_SOURCE=/path/to/Qwen3.8-27B-NVFP4 uv run tilerl serve --model qwen38-27b
```

OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` — point a client, or Claude
Code, at it. No GPU: `uv run tilerl serve` runs the tiny model on CPU.

```bash
uv add datasets   # only for the GSM8K dump below; not a runtime dependency
python scripts/gsm8k_jsonl.py train gsm8k.jsonl
uv run tilerl train --recipe grpo-gsm8k-27b --data gsm8k.jsonl --eval-gsm8k gsm8k_test.jsonl
uv run tilerl ledger
```

## Where things are

[`docs/roadmap.md`](docs/roadmap.md) — phases and gates ·
[`docs/design-rl-stack.md`](docs/design-rl-stack.md) — ISO, the draft head, the ledger ·
[`docs/design-engine.md`](docs/design-engine.md) ·
[`docs/design-kernels.md`](docs/design-kernels.md) ·
[`docs/design-ssd-read-path.md`](docs/design-ssd-read-path.md) — the prefix tier's break-even and async fetch ·
[`docs/support-matrix.md`](docs/support-matrix.md) — per-op status per target ·
[`docs/experience/`](docs/experience/) — every measurement, win and dead end, dated ·
[`AGENTS.md`](AGENTS.md) — the gates a change clears

Every number above sits in a dated entry under `docs/experience/`. The decode, prefill,
KV-reuse and training rows are additionally held by `bench` against
[`bench-baseline.json`](docs/experience/wins/bench-baseline.json) at ≥ 0.97×; the sglang
comparison, the V100 arm and the GSM8K results have no baseline key and rest on their
entries alone. `uv run pytest` is the suite that gates every commit.
