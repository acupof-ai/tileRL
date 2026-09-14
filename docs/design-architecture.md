# Target architecture (wrap-up refactor)

Baseline: origin/main `0e247634`, 2026-09-14. `src/tilerl` is 20,185 lines in 41 flat modules.
`packages/tilerl-kernels` is 8,368 lines. `scripts/` holds 248 files, 78 of them `probe_*`.
`tests/` holds 79 files.

This document fixes the module layout and the dependency direction that the cleanup PRs
converge to. It adds no feature and changes no behaviour. Every step below is a move or a
deletion that the existing gates can check.

## What is wrong today (measured)

| Problem | Evidence |
|---|---|
| God class | `Engine` is 2,568 lines: core loop 1,236, sparse runtime 640, memory ledger and stats 326, decode graphs 167, spec and sampling 121 |
| God CLI | `cli.py` is 2,820 lines. It imports 20 tilerl modules. `_train_adapters` alone is 608 lines, `_build_parser` 381, `cmd_bench_kernels` 173 |
| Two builders | `engine.build_engine` (324 lines) and `cli._build_engine` / `cli._build_model` both assemble an engine; tests and 89 scripts call one or the other |
| Import cycles | `autograd`↔`sparse_index`, `calibration`↔`cli`, `cli`↔`memory`, `dflash2`↔`spec`, `model`↔`tensor_parallel` |
| Upward imports | `memory` and `calibration` import `cli`; `prompt` imports `engine` |
| Storage is one file | `kv_cache.py` (1,919 lines) mixes device pools, host and SSD tiers, the boot store and the prefix store |
| Docs describe a removed engine | 13 stale sections in design-engine, design-kernels, design-sparse-kv and design-cost-model: no hybrid serve, `NoPrefixStore` for sparse, the old `tilerl.ops` path |
| Measurement sprawl | 248 scripts; the 09-09 review measured probe creation at ~11 per day |

## Layers and the one rule

Imports point downward only. A module may import from its own layer or from any layer below.
Nothing imports `cli`.

```
L6  cli.py                      argparse + thin cmd_* dispatch
L5  apps       server messages responses prompt ui_assets   (serving front end)
               generate eval judge math_answer bench        (offline use)
               train rollout iso merge calibration recipes ledger kernel_cost
L4  build.py                    config + checkpoint -> Model, Engine (the only assembler)
L3  schedule   engine.py        submit/poll/StepLimits, admit, plan, commit, release, loop
               decode_graph.py  captured dense and sparse decode graphs, buckets, precapture
               sparse_engine.py SparseTracker, SparseForward, SparsePrefixCache, sparse runtime
               spec.py          draft + verify (dflash2 merged in)
               memory.py        byte plan + measured ledger rows (engine stats call it)
L2  storage    kv_cache.py      PagedKvPool, LinearStatePool, PrefixStore, BatchKv
               kv_tiers.py      HostKvPages, ColdSsdFile, DramSnapshots, KvBootStore
               sparse_index.py  page bounds and selection math
L1  model      model.py tensor_parallel.py autograd.py
L0  base       precision.py config.py tokenizer.py
    kernels    packages/tilerl-kernels (backend, registry, kernels_*, reference)
```

What this keeps from design-engine.md: the `submit`/`poll` + `StepLimits` seam is the cost
contract, storage owns KV memory, the model calls backend ops only, and training shares the
same model and backend.

What changes is placement only:
- `build.py` becomes the single assembler.
- The sparse runtime leaves `Engine` for `sparse_engine.py`.
- Graphs leave for `decode_graph.py`.
- Ledger rows move to `memory.py`.
- Host and SSD tiers move to `kv_tiers.py`.
- Training orchestration leaves `cli.py` for `train.py`, and benchmark commands move to
  `bench.py`.

### The hybrid seam

After #586 the engine serves two regimes: a dense captured graph below `--sparse-min-tokens`
and eager sparse above it. The target makes that a named object instead of `sparse_*`
methods and `if r.sparse_on` branches spread across `Engine`. `Engine` holds an optional
`SparseRuntime` (in `sparse_engine.py`). The runtime owns the 17 `_sparse_*` methods and
`_hybrid_charge`, and it exposes the few calls the loop needs:

| Call | Replaces | When the loop calls it |
|---|---|---|
| `admit_headroom()` | `_sparse_hot_headroom` | dense admit |
| `rows(plan)` | `_sparse_rows`, `_sparse_decode_rows` | building a tick |
| `run(plan)` | `_run_sparse_decode_graph` and the eager path | forward |
| `finalize(req)` | `_sparse_finalize` + offers | release |
| `stats()` | `_sparse_live_stats` | stats |

A dense-only engine holds `None`, and the dense path contains no sparse branch. This is the
largest move and runs last, behind a device gate.

## Target sizes (estimates, to be replaced by the PR diffs)

| Module | Now | Target |
|---|---|---|
| engine.py | 3,633 | ~1,500 (core loop + request types) |
| decode_graph.py | – | ~450 |
| sparse_engine.py | 1,189 | ~1,800 (absorbs the sparse runtime) |
| build.py | – | ~350 (replaces `build_engine` + `cli._build_*`) |
| cli.py | 2,820 | ≤ 900 |
| kv_cache.py / kv_tiers.py | 1,919 | ~950 / ~950 |

The net reduction comes from deleting dead code, the duplicate builder and removed-feature
residue, not from the moves. A move PR is judged by "no behaviour change". A deletion PR is
judged by lines removed.

## Guardrail first

Before any move, add one structural test, `tests/test_layering.py`. It parses every module
under `src/tilerl` with `ast` and fails on an import from a higher layer, reading the layer
table above as data. Today's violations go into an explicit allowlist in that test, and the
allowlist may only shrink. It holds the direction while five people cut code in parallel,
because a wrong import is rejected when written, not caught in review. Its negative control
is an added `from .cli import x` in `memory.py`, which must fail.

## Sequence

Each row is one PR. Each is behaviour-preserving and passes CI (full suite). Rows marked
*device* also need a V100 smoke on the PR head before merge: short think-on tok/s within 2%
of 52, MMLU n=50 equal to 0.70, one unique 32k request answered.

| # | PR | Owner | Gate |
|---|---|---|---|
| 0 | This doc | coordinator | review |
| 1 | `tests/test_layering.py` + allowlist, red on an injected upward import | fixkv | CI |
| 2 | Stale design docs: rev-87's 13 items; archive `arch-review-2026-09-09`, `design-ssd-read-path` (KvTier removed) and completed ownership tables to `docs/history/` | fixmisc | CI |
| 3 | Dead code in the serving core, grep-proven (removed-feature residue, unused helpers) | fixkv | CI + device |
| 4 | Dead code in cli, server and API routes, grep-proven; `rollout.py` if its only consumer is its own test | fixmisc | CI |
| 5 | scripts/: delete dead one-off probes, keeping anything a doc, test, CI job or `test_main_selfchecks` glob reaches | ops | CI |
| 6 | Break the 5 cycles and 3 upward imports (small moves); shrink the allowlist | fixkv | CI |
| 7 | `build.py`: one assembler; callers of `cli._build_*` and `engine.build_engine` in src, tests and the 89 scripts are rewritten in the same PR (no re-export shim) | fixmisc | CI + device |
| 8 | `cli.py` split: training orchestration → `train.py`, bench commands → `bench.py`; the CLI surface is unchanged (`_EXPECTED_CLI_FLAGS`) | fixmisc | CI |
| 9 | `kv_tiers.py` split out of `kv_cache.py` | fixkv | CI |
| 10 | `decode_graph.py` + ledger rows → `memory.py` | fixkv | CI + device |
| 11 | `SparseRuntime` seam in `sparse_engine.py` | fixkv | CI + device (+ one unique 128k request) |

Order reason: deletions (3–5) shrink what the moves have to carry. Cycle breaks (6) make the
moves mechanical. The engine seam (11) goes last because it touches the hot path of both
regimes.

## Rules for every cleanup PR

- **Proof of death.** Name the consumer's own filter, not only a name search: `git grep`
  over src, tests, scripts, packages, docs and .github, plus `test_main_selfchecks`'s glob,
  `ci.yml`'s `*_world[0-9].py` pattern, `getattr` and string keys. Include one positive
  control that matches a live symbol.
- **No shims.** A moved symbol has no alias left behind; every caller is updated in the same PR (AGENTS.md: no half-states).
- **A move changes no code.** A move PR shows `git diff -M` as renames plus import edits.
  Any logic edit goes in a separate PR.
- **Frozen surfaces stay frozen:** OpenAI, Anthropic and Responses JSON shapes, CLI flags,
  the `submit`/`poll` seam, the bench-baseline schema.
- **Size.** At most ~600 changed lines per PR, one owner, reviewed by rev-87, merged by the
  coordinator.
- **The Mac runs one targeted test file at a time.** The full suite runs in CI; the Mac
  exhausted swap twice on 2026-09-14.

## Out of scope

- `packages/tilerl-kernels`: parity-gated, left as is this round.
- Test-suite restructuring. Tests move only when the module they import moves.
- Performance work; OPEN.md rows.
- `web/`.
