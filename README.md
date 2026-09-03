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

| one H20, Qwen3.8-27B | B=1 decode tok/s | prefill tok/s (512) | MMLU 0-shot |
|---|---:|---:|---:|
| tileRL, native NVFP4 + FP8 | **92.4** | **2887.6** | 74.6% |
| sglang, bf16 (cannot load NVFP4 on Hopper) | 54.2 | — | — |
| sglang, online fp8 | 39.9 | — | — |

The sglang cells are decode only, from one head-to-head session; our 2887.6
prefill comes from a later one and does not belong in a row with them. In that
head-to-head — same card, same session, our prefill reading 1836 — sglang's
online-fp8 arm won prefill at 4022 and its bf16 arm won B=8 decode at 387.0
against our 308.6. B=1 decode is the target, because that is what a rollout is.
Both sglang arms run a dequantized bf16 checkpoint that emits garbage, which
leaves their decode rates standing and their MMLU column empty. Our prefill
number was read on an idle box; the same code reads 2694.2 with neighbours on
the other cards.

**The roofline percentage is under revision and should not be quoted.** The
22.8 GB/tick it rests on is recorded in
`docs/experience/2026-08-28-vs-sglang-h20.md` with no derivation, and two
independent attempts to rebuild it from the checkpoint land elsewhere. What a
tick streams is 12.81 GB of fp4 nibbles plus its block scales; the checkpoint
holds one fp8 scale per 16 elements (1.60 GB) but `model.py:141` widens them to
f32 at load and `kernels_linear.py:546` is the kernel signature that requires
it, which is 6.41 GB — so 19.22 GB, and 19.22 GB in 11.0 ms is 1.75 TB/s, 53%
of the 3312 GB/s the card measures. The residual 3.6 GB is not accounted for.
A DRAM-read counter on one steady-state tick settles it; until then the honest
statement is that decode is bandwidth-bound, the achieved fraction is between
53% and 63%, and the gap is kernel efficiency rather than the memory system.

That scale widening is itself the largest lever on record: the f32 scales are
33% of what a tick streams and four times what the checkpoint holds, and
`renorm_fp4_scale` divides by a per-row power of two, which moves the exponent
and leaves the mantissa alone — so storing them back as e4m3 is lossless if no
row underflows. 19.22 → 14.41 GB would be 1.33× fewer bytes on a
bandwidth-bound kernel.

RL on the 27B has not yet moved a downstream metric; that run is the first
roadmap gate. Every number above sits in a dated entry under
`docs/experience/`, gated at ≥ 0.97× the committed baseline.

## Speculative decode: wired, measured, rejected at this wiring

One weight pass verifying a block is the only lever that passes a single-stream
roofline, so the DFlash2 block head now runs on the engine tick instead of in a
probe. Acceptance is high and the wall clock is worse.

| 200 GSM8K, B=8, greedy, graph on | wall | tok/s | tok / decode fwd | block accepted | GSM8K |
|---|---:|---:|---:|---:|---:|
| base | 278.6s | **232.3** | 7.80 | — | 170/200 = 85.0% |
| spec, width 8 | 466.5s | 139.4 | 42.99 | **6.18 of 8** | 167/200 = 83.5% |

**5.5× fewer trunk forwards, and 1.67× slower.** The drafter is 68.4% of the
tick: it walks rows one at a time in Python, outside the captured graph. A spec
tick costs about 9.2× a base tick, where acceptance pays for at most 6.18.

Batching the drafter is worth **3.64×** at the ceiling — and that is a division,
a measured 6.20× forward reduction over a measured 1.70× tick cost, not a third
result beside them. A width-8 verify tick costs 1.70–1.81× a width-1 tick at
B=8, across three runs on two cards, and 2.41× at B=1.

**The width-8 tick is not lossless**, which W>1 never promised to be: 152 of 200
completions differ from the base arm's. The 1.5-point GSM8K gap is 3 questions
against a 2.5-point binomial sd at n=200 and supports no regression claim in
either direction.

MMLU measures none of this. `mmlu_score` runs at `max_new_tokens=1`, so the
answer comes off the prefill and speculation never fires — `decode fwd 0`,
`drafted 0`, both arms. So it carries no throughput information here. What it
does carry is a correctness invariant: **742/1000 = 74.2% on both arms, 0 of
1000 completions differing.** A head wired into the tick, tapping five trunk layers
and re-serving its params fp8, has to leave the unspeculated path untouched, and
0 of 1000 says so without slack. (74.2% here against the 74.6% in the table
above is 4 questions and a different eval concurrency, not a speculation
effect.)

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
