# tileRL

Cross-platform train + inference runtime for `Qwen3.8-27B` (NVFP4), in Python,
on **one kernel source that targets CPU, CUDA, ROCm, and Metal** via
[TileLang](https://github.com/tile-ai/tilelang). The CPU target is the portable
default and the CI path; GPU targets compile from the same source.

tileRL pairs an inference engine (continuous batching, paged KV, prefix cache)
with **On-Policy Distillation (OPD)** training that shares the engine and
weights — one runtime, no second stack.

## Relationship to agent-infer

[`agent-infer`](https://github.com/cklxx/agent-infer) is the Rust ancestor of
this design. tileRL ports its ideas to Python + TileLang:

| | agent-infer | tileRL |
|---|---|---|
| Language | Rust | Python (uv package `tilerl`) |
| Kernels | per-backend native (CUDA C / Metal C++) | **one TileLang source → cpu/cuda/rocm/metal** |
| Engine seam | `BackendExecutor`: submit/poll + `StepLimits` | same seam, same cost contract |
| KV | paged full-attn + recurrent state + prefix cache | same |
| Training | OPD, shared engine/weights | OPD, shared engine/weights |
| Autograd | hand-written `autograd` crate | hand-written reverse-mode tape mirroring it |

The difference is the backend strategy: agent-infer writes and maintains a
native kernel tree per target; tileRL writes each kernel once and lets
TileLang lower it. torch is used only as the tensor container TileLang
requires — no `torch.autograd`, no `torch.optim`.

## Quickstart

```bash
uv sync                                          # never pip install
uv run pytest                                    # correctness suite (CPU)
TILERL_TARGET=cpu uv run tilerl serve            # OpenAI-compatible server
TILERL_TARGET=cpu uv run tilerl bench            # benchmark → docs/experience/
uv run tilerl train                              # OPD training
```

Requires Python 3.11+ and uv 0.9+. On machines without a GPU, `TILERL_TARGET`
defaults to `cpu`.

## Development

```bash
uv run ruff check                               # lint (rule set: pyproject.toml [tool.ruff])
uv run ruff format --check                      # format check
uv run pytest                                   # same deterministic suite as CI
```

CI (`.github/workflows/ci.yml`) runs on `ubuntu-latest` + `macos-14`:
`uv sync --dev`, ruff lint, and the hermetic CPU suite (`TILERL_TARGET=cpu`).
Only deterministic checks gate the build — the real-weight test is
`TILERL_TEST_REAL=1`-gated and GPU/Metal tests auto-skip, so plain `uv run
pytest` on CI is exactly the deterministic set.

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
                  │  TileLang kernels, one source:       │
                  │  cpu ✓ | cuda | rocm | metal         │
                  └──────────────────────────────────────┘
```

Everything above `src/tilerl/ops/` is backend-neutral; `ops/` is the only
layer that touches TileLang or torch beyond the tensor container.

## Status

| Component | Status |
|---|---|
| CPU target | ✓ working (CI/dev path) |
| CUDA / ROCm / Metal targets | pending-host (same kernel source) |
| Tiny model end-to-end | ✓ |
| Qwen3.8-27B NVFP4 weights | pending HF integration |
| Paged KV + prefix cache | ✓ (tiny) |
| Autograd tape + OPD | ✓ (tiny) |
| OpenAI-compatible server | ✓ |

Perf work is recorded under `docs/experience/wins/` (and `errors/` for
regressions); see `AGENTS.md` for the working contract.
