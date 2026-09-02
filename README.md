# tileRL

Serve and RL-train `Qwen3.8-27B` (NVFP4) on one Hopper card, in one process.

Two things nobody else ships together:

- **Native NVFP4 + FP8 decode on H20.** On Hopper, sglang and vLLM fall back
  to weight-only Marlin and recommend the FP8 checkpoint instead; sglang
  refuses the NVFP4 checkpoint outright. tileRL's TileLang kernels run it as
  fp4 weights × fp8 activations, and single-stream decode is the fastest
  measured on this card.
- **The engine that samples is the model that trains.** GRPO and self-teacher
  on-policy distillation roll out through the same engine and the same weights,
  LoRA on the frozen fp4 base. No weight sync between trainer and rollout —
  not a protocol, a memory-format fact.

## Where it stands (2026-08-28, one H20, Qwen3.8-27B-NVFP4)

Decode is `tilerl bench --suite decode-kv` (steady-state median of 3×20 ticks);
every row is gated at ≥0.97× the committed snapshot in
`docs/experience/wins/bench-baseline.json`.

| | B=1 tok/s @512 / 2k / 8k / 32k | B=8 agg tok/s @512 | prefill tok/s @512 |
|---|---|---:|---:|
| **tileRL** (native NVFP4 + FP8) | **92.4 / 87.3 / 88.9 / 80.1** | 308.0 | 1836 |
| Arle (agent-infer, same card) | 84.5 | — | — |
| sglang bf16 (same card; no NVFP4 path on Hopper) | 54.2 | **387** | 2512 |
| sglang online-fp8 | 39.9 | 266.6 | **4908** |

Long context, B=1: **61.1 tok/s at 128K, 48.1 at 256K** (the KV alone is
17 / 34 GB). Accuracy: **MMLU 0-shot 76.3%** (763/1000, `scripts/mmlu.py`).
The sglang rows are throughput only — the bf16 checkpoint it has to run emits
garbage ([why](docs/experience/errors/2026-08-28-sglang-bf16-checkpoint-garbage.md)).

Training on the same card: LoRA on the frozen fp4 base at **26.5 / 32.6 / 38.2
tok/s** (1×64/128/256, peak 47–58 GB); Adafactor full fine-tuning of the 27B
fits in 73 GB. GRPO on GSM8K with an MMLU before/after gate is wired
(`tilerl train --rl --data`) and **pending-remote** — the 27B numbers land in
[`docs/experience/wins/2026-09-02-rl-real-task.md`](docs/experience/wins/2026-09-02-rl-real-task.md).

sglang is ahead at B≥8 and on prefill. Those are not targets: an RL runtime is
priced by seconds per step, and rollout is decode at B≥32 — the next kernel
([`docs/roadmap.md`](docs/roadmap.md)). How decode got from 52 to 92, with the
dead ends: [`docs/experience/2026-08-28-decode-52-to-84.md`](docs/experience/2026-08-28-decode-52-to-84.md);
the sglang comparison and its caveats:
[`docs/experience/2026-08-28-vs-sglang-h20.md`](docs/experience/2026-08-28-vs-sglang-h20.md).

## What is in the box

- **Engine**: continuous batching, `submit`/`poll` + `StepLimits`, one forward
  per tick; paged KV for the full-attention layers, recurrent state for the
  gated-delta layers, hash prefix cache; OpenAI-compatible server with SSE.
- **Trainer**: hand-written reverse-mode tape (no `torch.autograd`, no
  `torch.optim`); GRPO, self-OPD, SFT; LoRA or Adafactor full-parameter.
- **Kernels**: one TileLang file tree behind both. `cpu` is the CI and parity
  harness — every kernel has a CPU-executable twin — `metal` runs locally, and
  CUDA sm90 holds the fp4/fp8 tensor-core schedules (71% of the 1,969 kernel
  lines). Per-op status: [`docs/support-matrix.md`](docs/support-matrix.md).

## Relationship to agent-infer

[`agent-infer`](https://github.com/cklxx/agent-infer) is the Rust ancestor of
this design. tileRL ports its ideas to Python + TileLang:

| | agent-infer | tileRL |
|---|---|---|
| Language | Rust | Python (uv package `tilerl`) |
| Kernels | per-backend native (CUDA C / Metal C++) | one TileLang tree; 71% of it is sm90 schedules |
| Engine seam | `BackendExecutor`: submit/poll + `StepLimits` | same seam, same cost contract |
| KV | paged full-attn + recurrent state + prefix cache | same |
| Training | OPD, shared engine/weights | OPD, shared engine/weights |
| Autograd | hand-written `autograd` crate | hand-written reverse-mode tape mirroring it |

The difference is the backend strategy: agent-infer writes and maintains a
native kernel tree per target; tileRL writes each kernel in TileLang and lets
it lower. Where a target needs a different schedule the registry swaps that one
kernel — Metal takes naive FMA gemms, sm90 takes WGMMA schedules — so "one
source" means one file tree and one op contract, not one schedule. torch is
used only as the tensor container TileLang requires — no `torch.autograd`, no
`torch.optim`.

## Install

Two distributions, split along the only boundary the codebase already
enforces: `tilerl_kernels` is the sole place that imports TileLang or calls
torch beyond the tensor container, and everything above it is backend-neutral.

| | install | what for |
|---|---|---|
| `tilerl` | `pip install tilerl` | engine, model loading, OPD training, CLI |
| `tilerl[server]` | `pip install "tilerl[server]"` | + the OpenAI-compatible HTTP server |
| `tilerl-kernels` | `pip install tilerl-kernels` | kernels and the backend seam alone |

A kernel change rebuilds one of them; a serving change rebuilds the other.
Requires Python 3.11+. Without a GPU, `TILERL_TARGET` resolves to `cpu`.

## Quickstart

```bash
pip install "tilerl[server]"
tilerl serve                                     # OpenAI-compatible, :8000
tilerl serve --devices 0,1,2,3                   # one engine replica per GPU
```

```python
from tilerl.engine import SamplingParams, build_engine

rid = engine.submit(tokens, SamplingParams(max_new_tokens=64, logprobs=True))
while rid not in (done := engine.poll()):
    engine.step()                                # one forward per tick
tokens, scores = done[rid], engine.logprobs(rid)
```

From a checkout:

```bash
uv sync                                          # never pip install
uv run pytest                                    # correctness suite (CPU)
TILERL_TARGET=cpu uv run tilerl serve            # OpenAI-compatible server
TILERL_TARGET=cpu uv run tilerl bench            # benchmark → docs/experience/
uv run tilerl train --opd                        # OPD self-teacher training
python scripts/gsm8k_jsonl.py train gsm8k.jsonl  # then, on a card:
tilerl train --recipe grpo-gsm8k-27b --data gsm8k.jsonl --eval-gsm8k gsm8k_test.jsonl
tilerl ledger                                    # every run: inputs, gates, lineage
```

## Development

The checkout is a uv workspace: `uv sync` installs both packages editable plus
everything the gates touch, so one command covers serving, training and kernel
work.

```bash
uv sync                                         # both packages, editable
uv run ruff check                               # lint (rule set: pyproject.toml [tool.ruff])
uv run pytest                                   # same deterministic suite as CI
uv run python -m tilerl_kernels.reference       # ops self-checks, no GPU
```

Contributing, and the gates a change has to clear: [CONTRIBUTING.md](CONTRIBUTING.md).

CI (`.github/workflows/ci.yml`) runs on `ubuntu-latest` + `macos-14`:
`uv sync --dev`, ruff lint, and the hermetic CPU suite (`TILERL_TARGET=cpu`).
Only deterministic checks gate the build — GPU/Metal tests auto-skip, so
plain `uv run pytest` on CI is exactly the deterministic set.

## Architecture

```
                 ┌──────────────────────────────────────────┐
                 │  cli (tilerl serve | train | bench)      │
                 └───────┬──────────────────────┬───────────┘
                         │                      │
                  ┌──────▼──────┐        ┌──────▼──────┐
                  │   server    │        │    train    │
                  │ (OpenAI API)│        │    (OPD)    │
                  └──────┬──────┘        └──────┬──────┘
                         │                      │ reverse-mode tape
                         │              ┌───────▼───────┐
                         │              │  autograd bwd │
                         │              └───────┬───────┘
                  ┌──────▼──────────────────────▼──────┐
                  │             engine                  │
                  │  submit/poll + StepLimits           │
                  │  continuous batching, one fwd/tick  │
                  └──────┬──────────────────┬──────────┘
                         │                  │
                  ┌──────▼──────┐    ┌──────▼──────────────┐
                  │    model    │    │       kv_cache       │
                  │  Qwen3.8    │    │  paged (full-attn)   │
                  │  NVFP4      │    │  recurrent (gdn)     │
                  └──────┬──────┘    │  hash prefix cache   │
                         │           └─────────────────────┘
                  ┌──────▼──────────────────────────────┐
                  │           ops (seam)                 │
                  │  TileLang kernels, one file tree:    │
                  │  cpu ✓ | metal ✓ | sm90 ✓ (pod)      │
                  └──────────────────────────────────────┘
```

Everything above `packages/tilerl-kernels/` is backend-neutral; that package
is the only layer that touches TileLang or torch beyond the tensor container.

## Status

| Component | Status |
|---|---|
| CPU target | ✓ CI + local, every commit (97 passed, 4 skipped) |
| Metal target | ✓ local (97 passed, 4 skipped) |
| CUDA sm90 target | ✓ H20 pod, 27B decodes correctly (verify checks 1–3), perf gated by the snapshot harness |
| sm100 / sm120 | registered empty; `NotImplementedError` on use |
| Tiny model end-to-end | ✓ |
| Qwen3.8-27B NVFP4 weights | ✓ served natively (twiddled fp4 + fp8), 92.4 tok/s B=1 on H20, MMLU 76.3% |
| Paged KV + prefix cache | ✓ (tiny) |
| Autograd tape + GRPO + OPD | ✓ (tiny); 27B GSM8K + MMLU gate pending-remote |
| OpenAI-compatible server | ✓ |

Perf work is recorded under `docs/experience/wins/` (and `errors/` for
regressions); see `AGENTS.md` for the working contract.
