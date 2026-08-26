# tileRL

Cross-platform train + inference runtime for `Qwen3.8-27B` (NVFP4), in Python,
on one [TileLang](https://github.com/tile-ai/tilelang) kernel source. Three
targets have executed it: CPU (the CI and dev path), Metal, and CUDA sm90 (on
an H20 pod). ROCm shares the CPU kernel set and has never run.

The source is not evenly shared. Of the 1,969 lines under
`src/tilerl/ops/kernels*.py`, 1,406 (71%) are sm90-only schedules and 175 (9%)
are the kernels every target executes; the rest is per-target gemm schedules
and a shared header. Per-op status:
[`docs/support-matrix.md`](docs/support-matrix.md).

tileRL pairs an inference engine (continuous batching, paged KV, prefix cache)
with **On-Policy Distillation (OPD)** training that shares the engine and
weights — one runtime, no second stack.

## Relationship to agent-infer

[`agent-infer`](https://github.com/cklxx/agent-infer) is the Rust ancestor of
this design. tileRL ports its ideas to Python + TileLang:

| | agent-infer | tileRL |
|---|---|---|
| Language | Rust | Python (uv package `tilerl`) |
| Kernels | per-backend native (CUDA C / Metal C++) | one TileLang source; 71% of it is sm90-only |
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
                  │  rocm = the cpu set, never run       │
                  └──────────────────────────────────────┘
```

Everything above `src/tilerl/ops/` is backend-neutral; `ops/` is the only
layer that touches TileLang or torch beyond the tensor container.

## Status

| Component | Status |
|---|---|
| CPU target | ✓ CI + local, every commit (97 passed, 4 skipped) |
| Metal target | ✓ local (97 passed, 4 skipped) |
| CUDA sm90 target | ✓ H20 pod 2026-08-24 (60 passed); not re-run since |
| ROCm target | never executed — resolves to the CPU kernel set |
| sm100 / sm120 | registered empty; `NotImplementedError` on use |
| Tiny model end-to-end | ✓ |
| Qwen3.8-27B NVFP4 weights | pending HF integration |
| Paged KV + prefix cache | ✓ (tiny) |
| Autograd tape + OPD | ✓ (tiny) |
| OpenAI-compatible server | ✓ |

Perf work is recorded under `docs/experience/wins/` (and `errors/` for
regressions); see `AGENTS.md` for the working contract.
