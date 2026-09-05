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
| Kernel file layout, registry rules, SOTA iteration loop | [`docs/design-kernels.md`](docs/design-kernels.md) |
| Engine layering and seams (frontend/schedule/storage) | [`docs/design-engine.md`](docs/design-engine.md) |

`AGENTS.md` is canonical; `CLAUDE.md` is a symlink to it.

---

## Project shape

`tilerl` is a uv-managed Python package that **serves and RL-trains**
`Qwen3.8-27B` (NVFP4) on one Hopper card in one process.

- **ONE backend: TileLang.** One kernel file tree; `cpu`, `metal` and CUDA
  sm90 have executed it. The CPU target is the test harness and the CI/dev
  path — every kernel has a CPU twin so the parity gate runs on this GPU-less
  machine. sm90 is the perf target and holds its own cells.
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

Reference repos are read-only: never modify `/Users/bytedance/code/agent-infer`
or `/Users/bytedance/code/tilelang`.

---

## Hard gates

**Backend isolation.** `packages/tilerl-kernels/src/tilerl_kernels/` is the
only layer that touches TileLang or torch beyond the container type; modules
above it call backend ops. No `torch.autograd`, no `torch.optim`, anywhere in
framework code.

**Every kernel has a CPU twin.** A new op lands in the CPU cell first, so the
parity gate runs here; an sm90 schedule is a per-arch cell that overrides it,
never the only implementation.

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
failure mode.

**No half-states.** Finish a refactor unit or revert it; never leave parallel
old+new paths in the tree.

**Extreme minimality.** Smallest line count that keeps function and
performance. Delete before adding; merge before growing; a shorter diff that
passes the same gates wins. Parallel agents write defensively; a consolidation
pass that deletes their excess is part of finishing the work.

**Plain language.** Say the finding in plain words first, numbers second. No
dense jargon, no hedging, no restating the question. If a sentence needs a
glossary, rewrite it.

**Approach-first for >3 files or architectural decisions** — outline, then
execute. Wait for the user only when two viable paths have different costs;
no tradeoff, don't ask.

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

**Shared checkout.** Other Claude sessions work in this directory at the
same time. Never stash, checkout, or commit files you did not change; do
integration in a scratchpad worktree and fast-forward `main`. Before a push
or a fast-forward pull, tell the peers `ListAgents` shows for this repo
(`SendMessage`) which files you touch, and wait a few minutes for objections.

**Addressing.** Send to `name [ref]` from `runs/roster.json`, never a bare
name: several projects run sessions whose names share a prefix. And **a relay
channel is not the owner of the work it relays** — a session named in a grant,
a merge or a hand-off is the channel, not the runner, and a review sent to it
goes to the wrong desk. Name the runner. (2026-09-05: a MoE-bench review
reached `tilerl-27` because a grant commit read "via tilerl-27".)

**Pod jobs go through `scripts/pod_run.sh <name> <card> -- <cmd>`.** Never
hand-type the launcher: it reaps the job (a `setsid nohup` from an exiting
shell orphans to a PID 1 that is `sleep infinity` and never `wait()`s), logs
to `/work`, the tree that survives a container restart, claims with the python
pid and retries only the device-fd refusal, releases from a trap, and refuses
a card holding >64 MiB with no claim. After any kill, read `ps -o stat=`:
`kill -0`, `/proc/<pid>` and `pgrep -f` all call a zombie alive.

**Writing `bench-baseline.json`.** Append-mostly and shared, so serialize it
with `json.dumps(d, indent=2, sort_keys=True)`. Any other form reindents the
file and a one-row append becomes 196 insertions that every peer's edit then
conflicts against; `merge=union` covers CHANGELOG, not this. Since #112 a beat
is a candidate written to `runs/<id>/baseline-candidate.json`, so a write here
is a deliberate promotion.

**On-policy rollouts.** `grpo_loop` and self-OPD refuse an engine whose caches
would outlive an update, per cache: `recapture_graph=` waives the decode graph,
`clear_prefix=` the prefix store, and a waiver obliges the loop to call
`invalidate_weights()` after every step. The RL path now runs with
`decode_graph=True, recapture_graph=True` — worth 2.16x on the 27B step
(73.62 → 34.09 s) — and `prefix_store=NoPrefixStore()` until the block-granular
store lands, since today's store publishes and never serves.

**Git.** Commitizen `<type>(<scope>): <subject>`, scopes `kv` `engine` `ops`
`autograd` `train` `server` `docs`. Work on a named branch from a scratchpad
worktree, push it, open a PR, and merge it once CI is green and review
comments are answered. Small tranches, each self-contained,
simplify pass first. Commit only your own files by explicit path. Never touch the reference repos.
**No AI attribution in commits** — no Claude/cc/co-authored-by mentions in
messages or trailers; write them as if a human wrote them.

**CHANGELOG is the central progress record.** Three event classes land a line
the same day, linking the wins/errors entry: **phase exit · default flip ·
accept-or-reject verdict**.

**Code layout.** Flat modules — no deep package trees. Comments carry the
non-obvious *why* in ≤1 English line, never the *what* and never which task
added it. If the code already reads clearly, leave it bare — no comment.

**Before a number leaves your session** (2026-09-05: twenty errors entries in one day,
most of them one of these five):
- A mechanism claim ships with the probe that tested it. Arithmetic over config dims and
  a reading of the code are hypotheses; two of each were wrong in both directions today.
- An arm's number needs a placement control: the same arm in another position, another
  process, or another card. First-position JIT, an idle-card microbench (11.6 vs 161.9 ms
  in-path) and a two-conversation sweep each produced a clean, false table.
- A gate is green only after its negative control is red. Four vacuous gates today: a
  capture path that silently fell back on CPU, a `callable(getattr(...))` true for every
  engine, a monkeypatch that never reached the importlib'd copy, a one-step test that a
  loop-entry clear satisfied.
- Volume is not time. A pad histogram in elements says where copies are, not where
  seconds go; only a phase-attributed profile does.
- A negative verdict on a condition with two variables needs a sweep of each. "DRAM tier
  is structurally useless" came from a sweep of the HBM budget at two sessions; the
  condition was sessions > budget, and the other axis reversed it (0/63 → 24/0 hits).
  A one-axis sweep finds a threshold, and a threshold reads like a law. A veto gets the
  scrutiny a claim gets; an instrument that errs toward your conclusion produces no
  surprise and so never gets checked.
- Before building, `git log -S<symbol> --all`: KvTier was four commits and a review pass
  on a branch when "no such tier exists" was written on main.

**An errors entry that names a fix is not a fixed bug.** The eval cap was written up on
09-04 with its fix and scored a live run on 09-05. An entry whose fix has not landed says
`Status: open` and is listed in [`docs/experience/OPEN.md`](docs/experience/OPEN.md); the
PR that closes it removes the line. Review a training or serving change against that list.

**Memory.** Skeletons: `errors/YYYY-MM-DD-slug.md` = Context / Root Cause /
Fix / Rule; `wins/…` = Context / What Worked / Rule. Bench snapshots use
[TEMPLATE-bench.md](docs/experience/wins/TEMPLATE-bench.md), never overwritten.

---

## Build & run

```bash
uv sync                          # install deps into .venv (never pip install)
uv run tilerl serve              # OpenAI-compatible server
uv run tilerl train --recipe X   # SFT / --rl / --opd from a gated flag set; runs/<id>/manifest.json
uv run tilerl merge --base B --specialists S1,S2 --out D   # ISO merge, one tensor at a time
uv run tilerl ledger             # runs, gates, lineage (--json for agents)
uv run tilerl bench              # benchmark (writes a bench entry)
uv run pytest                    # test suite
uv run ruff check                # lint
uv run ruff format --check       # format check
```

Target selection: `TILERL_TARGET=cpu|cuda|metal|auto` (default `auto`;
on this machine that resolves to `cpu`).

Dependencies: `uv add <pkg>` / `uv add --dev <pkg>` — never `pip install`.

**Ruff rule set.** `select = ["E", "F", "I", "UP", "SIM"]`, line-length 100,
target py311 (see `pyproject.toml [tool.ruff]`). Day-1 baseline: 11 categories
(E501/E702/E731/E741/F401/F541/I001/SIM108/UP006/UP035/UP037) are `ignore`d
because in-flight `src/`/`tests/` code violates them — re-enable one by one as
the tree is cleaned, never grow the list. F821 stays on globally, suppressed
per-file only for `autograd.py` and `ops/reference.py` (known in-flight spots).

**CI.** `.github/workflows/ci.yml` gates on `ubuntu-latest` + `macos-14`:
`uv sync --dev` → `ruff check` → `TILERL_TARGET=cpu uv run pytest -v`. Only
lint + hermetic CPU tests block; GPU/Metal tests auto-skip; no bench steps.
`ruff format --check` is non-blocking until the tree is reformatted.
