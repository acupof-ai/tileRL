# tileRL — Agent Contract

Assisting **ckl**. Project caveats and hard gates only — generic Python /
TileLang / git knowledge is intentionally absent, and so is anything you can
read off the file tree. Match the surrounding code's idiom, naming, and
comment density rather than a style rulebook.

**Load on demand, not upfront:**

| When | Read |
|------|------|
| What tileRL is, architecture, status | [`README.md`](README.md) |
| Any bench or perf claim | [`docs/experience/wins/`](docs/experience/wins/), [`docs/experience/errors/`](docs/experience/errors/) |
| Bench entry skeleton | [`docs/experience/wins/TEMPLATE-bench.md`](docs/experience/wins/TEMPLATE-bench.md) |
| The seam / tape / KV design being mirrored | `/Users/bytedance/code/agent-infer` (read-only reference) |
| Kernel idioms and SOTA examples to copy | `/Users/bytedance/code/tilelang` (read-only reference) |

`AGENTS.md` is canonical; `CLAUDE.md` is a symlink to it.

---

## Project shape

`tilerl` is a uv-managed Python package for cross-platform **training and
inference** of `Qwen3.8-27B` (NVFP4). One kernel source, four targets.

- **ONE backend: TileLang.** One kernel source compiles for `cpu` / `cuda` /
  `rocm` / `metal`. The **CPU target is the portable default and the CI/dev
  path** — this machine has no GPU, so everything verified here runs on CPU.
  GPU targets are pending-host bring-up.
- **torch is the tensor container only** (TileLang hard-depends on it).
  NO `torch.autograd`, NO `torch.optim` in framework code — training runs on
  our own reverse-mode tape.
- **Hand-written reverse-mode autograd tape**, mirroring
  `agent-infer/crates/autograd`. Backward kernels: TileLang where SOTA exists
  (gated-delta); torch-eager reference for the rest on day-1, behind the same
  op interface, each marked
  `# ponytail: torch-eager backward, tilelang kernel when perf demands`.
- **Engine seam = the cost contract**: `submit` / `poll` + `StepLimits`,
  continuous batching, one forward per tick. Same idea as agent-infer's
  `BackendExecutor` — a new target implements the loop, it does not bend the
  seam.
- **State**: paged KV cache (full attention) + recurrent state
  (gated-delta) + hash-based prefix cache.
- **OPD training shares the engine and weights with serving** — one runtime,
  no second product line.
- **SOTA kernels are copied, not reinvented**: gated-delta-net examples,
  TileOPs, TileRT from the TileLang ecosystem.

Non-obvious ownership:
- `src/tilerl/ops/` is the ONLY layer that touches TileLang or torch beyond
  the container type. Everything above it is backend-neutral.
- Reference repos are read-only: never modify
  `/Users/bytedance/code/agent-infer` or `/Users/bytedance/code/tilelang`.

---

## Hard gates

**Backend isolation (CRITICAL).** Modules above `src/tilerl/ops/` never call
TileLang or torch directly — they call backend ops. `torch` may appear only as
the tensor container (`torch.Tensor` type). No `torch.autograd`, no
`torch.optim`, anywhere in framework code.

**Target-neutral kernels.** Block-parallel schedules only — no warp /
warp-memory specifics — so every kernel compiles on CPU AND GPU from one
source. GPU-tuned schedules are day-2 work; do not specialize early.

**Every runtime change produces a bench entry.** A dated entry under
`docs/experience/wins/` (or `errors/` on regression) — no entry, not shipped.
In scope: anything under `src/tilerl/` on the hot path, bench parameter
changes, default flips, hot-path dep bumps. Exempt: docs / agent files /
dev-only tooling — say so in the commit body. Can't run a target locally (GPU
on a Mac) → stub `pending-remote`; no silent skips.

**Correctness parity = the correct-inference gate.** TileLang (CPU target) vs
the torch-eager reference, `allclose(rtol=1e-2)`, on the tiny model. Every new
op lands with this parity check.

**Tape gradient check for any new backward.** Numerical gradcheck on the tiny
model — the tape is hand-written, so silent gradient errors are the default
failure mode, not an exotic one.

**No half-states.** Finish a refactor unit or revert it; never leave parallel
old+new paths in the tree.

**Extreme minimality.** Smallest line count that keeps function and
performance — both are non-negotiable, LOC is the thing to cut. Delete before
adding; merge before growing; a shorter diff that passes the same gates wins.
Parallel agents write defensively; a consolidation pass that deletes their
excess is part of finishing the work.

**Plain language.** Say the finding in plain words first, numbers second. No
dense jargon, no hedging, no restating the question. If a sentence needs a
glossary, rewrite it.

**Approach-first for >3 files or architectural decisions** — outline, then
execute. Wait for the user ONLY when there is a real tradeoff to adjudicate
(two viable paths with different costs). No tradeoff → nothing to decide →
don't ask.

**Ponytail.** Laziest correct implementation; no speculative abstractions. A
shortcut with a known ceiling gets one marker:
`# ponytail: <ceiling>, <upgrade path>`.

**One runnable check per non-trivial logic.** An `assert` in `__main__` or one
small test — whichever is lighter.

---

## Working rules

**Phases** (non-trivial tasks): Explore until you can name every file you will
touch → Plan (accepted in writing; >5 files or irreversible → stop and flag) →
Implement (runs, simplify pass on the diff) → Verify (`uv run pytest`,
parity/gradcheck where applicable, bench entry) → Reflect (bug that took >1
attempt → `docs/experience/errors/`; user correction → feedback memory).
Trivial → Implement + Verify.

**Tests: minimal and end-to-end.** Default is no new test. Add one only when
the change carries logic that can silently break — KV management, attention
indexing, quantization, tape backward, sampling — and then the smallest
end-to-end gate that fails when it breaks, not a per-function suite.

**Delegation.** The orchestrating agent does direction, docs, planning, and
integration; `general-purpose` subagents execute; `Explore` maps; `Plan`
handles >5-file plans. Independent tasks go out in one message, in parallel.
Two failed subagent attempts → hand-write the diff or re-brief a fresh agent
with what was tried.

**Git.** Commitizen `<type>(<scope>): <subject>`, scopes `kv` `engine` `ops`
`autograd` `train` `server` `docs`. Commit directly to `main` — no feature
branches. Small tranches, each self-contained, simplify pass first. Commit
only your own files by explicit path. Never touch the reference repos.
**No AI attribution in commits** — no Claude/cc/co-authored-by mentions in
messages or trailers; write them as if a human wrote them.

**CHANGELOG is the central progress record.** Three event classes land a line
the same day, linking the wins/errors entry: **phase exit · default flip ·
accept-or-reject verdict**.

**Code layout.** Flat modules — no deep package trees. Comments carry the
non-obvious *why* in ≤1 English line, never the *what* and never which task
added it. If the code already reads clearly, leave it bare — no comment.

**Memory.** Skeletons: `errors/YYYY-MM-DD-slug.md` = Context / Root Cause /
Fix / Rule; `wins/…` = Context / What Worked / Rule. Bench snapshots use
[TEMPLATE-bench.md](docs/experience/wins/TEMPLATE-bench.md), never overwritten.

---

## Build & run

```bash
uv sync                          # install deps into .venv (never pip install)
uv run tilerl serve              # OpenAI-compatible server
uv run tilerl train              # OPD training
uv run tilerl bench              # benchmark (writes a bench entry)
uv run pytest                    # test suite
uv run ruff check                # lint
uv run ruff format --check       # format check
```

Target selection: `TILERL_TARGET=cpu|cuda|rocm|metal|auto` (default `auto`;
on this machine that resolves to `cpu`).

Dependencies: `uv add <pkg>` / `uv add --dev <pkg>` — never `pip install`.

**Ruff rule set.** `select = ["E", "F", "I", "UP", "SIM"]`, line-length 100,
target py311 (see `pyproject.toml [tool.ruff]`). Day-1 baseline: 11 categories
(E501/E702/E731/E741/F401/F541/I001/SIM108/UP006/UP035/UP037) are `ignore`d
because in-flight `src/`/`tests/` code violates them — re-enable one by one as
the tree is cleaned, never grow the list. F821 stays on globally, suppressed
per-file only for `autograd.py` and `ops/reference.py` (known in-flight spots).

**CI.** `.github/workflows/ci.yml` gates on `ubuntu-latest` + `macos-14`:
`uv sync --dev` → `ruff check` → `TILERL_TARGET=cpu uv run pytest -v`.
Determinism policy: only lint + hermetic CPU tests block; GPU/Metal tests
auto-skip, and no bench/perf steps exist. `ruff format --check` runs non-blocking until the tree
is reformatted.
