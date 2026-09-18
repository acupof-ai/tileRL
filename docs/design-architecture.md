# Target architecture (wrap-up refactor)

Baseline: origin/main `0e247634`, 2026-09-14. `src/tilerl` is 20,185 lines in 33 files (32 modules plus `__init__`).
`packages/tilerl-kernels` is 8,368 lines. `scripts/` holds 248 files, 78 of them `probe_*`.
`tests/` holds 79 files.

This document fixes the module layout and the dependency direction that the cleanup PRs
converge to. It adds no feature and changes no behaviour. Every step below is a move or a
deletion that the existing gates can check.

## What is wrong today (measured)

| Problem | Evidence |
|---|---|
| God class | `Engine` is 2,568 lines: core loop 1,236, sparse runtime 640, memory ledger and stats 326, decode graphs 167, spec and sampling 121 |
| God CLI | `cli.py` is 2,820 lines and four products (parser 381, train/RL loop with eval statistics ~1,100, bench renderers ~280, ledger rendering ~100). It imports 19 tilerl modules at top level, 23 including lazy in-function imports. `_train_adapters` alone is 608 lines, `_build_parser` 381, `cmd_bench_kernels` 173 |
| Two builders | `engine.build_engine` (324 lines) and `cli._build_engine` / `cli._build_model` both assemble an engine; tests and 89 scripts call one or the other |
| Import cycles | `autograd`↔`sparse_index`, `calibration`↔`cli`, `cli`↔`memory`, `dflash2`↔`spec`, `model`↔`tensor_parallel` |
| Upward imports | `memory` and `calibration` import `cli`; `autograd` imports `sparse_index` |
| Three API routes, three request paths | OpenAI, Anthropic and Responses routes each copy the completion wait loop, thinking resolution and the non-stream post-processing |
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
               train iso merge calibration recipes ledger kernel_cost
L4  build.py                    config + checkpoint -> Model, Engine (the only assembler)
L3  schedule   engine.py        submit/poll/StepLimits, admit, plan, commit, release, loop
               decode_graph.py  captured dense and sparse decode graphs, buckets, precapture
               sparse_engine.py SparseTracker, SparseForward, SparsePrefixCache
               sparse_runtime.py SparseRuntime: residency, promotion, sparse decode graph
               spec.py          draft + verify (dflash2.py merged in at step 6)
               memory.py        byte plan + measured ledger rows (engine stats call it)
L2  storage    kv_cache.py      PagedKvPool, LinearStatePool, PrefixStore, BatchKv
               kv_tiers.py      HostKvPages, ColdSsdFile, DramSnapshots, KvBootStore
               sparse_index.py  page bounds and selection math
L1  model      model.py tensor_parallel.py autograd.py
L0  base       precision.py config.py tokenizer.py testing.py

`rollout.py` is deleted in step 4 and is absent from the table.
    kernels    packages/tilerl-kernels (backend, registry, kernels_*, reference)
```

What this keeps from design-engine.md: the `submit`/`poll` + `StepLimits` seam is the cost
contract, storage owns KV memory, the model calls backend ops only, and training shares the
same model and backend.

What changes is placement only:
- `build.py` becomes the single assembler.
- The sparse runtime leaves `Engine` for `sparse_runtime.py`.
- Graphs leave for `decode_graph.py`.
- Ledger rows move to `memory.py`.
- Host and SSD tiers move to `kv_tiers.py`.
- Training orchestration leaves `cli.py` for `train.py`, and benchmark commands move to
  `bench.py`.

### The hybrid seam

After #586 the engine serves two regimes: a dense captured graph below `--sparse-min-tokens`
and eager sparse above it. The realized seam (step 11) is `SparseRuntime` in
`sparse_runtime.py`: Engine holds it as `engine._sparse` (None on a dense engine), and it
owns the sparse-tick methods — selection geometry, residency resolve/evict,
promotion/demotion, prefix offers, warm-draft restore, and the sparse capture/replay.
Engine keeps scheduling (`route`/sparse_on predicates, `_hybrid_charge`, the
`_sparse_prefill_cap` decision), the #500 `sparse_retier` page walk, and
`_sparse_live_stats`/`_sparse_hot_headroom`. Pools, model, backend and the
verify/sample/draft-step callbacks cross the seam on a frozen `SparseCtx`; the runtime
never imports Engine. The tracker attributes the tree reads are proxied through the same
`engine._sparse` facade, so callers below kept their spellings. The planned calls:

| Call | Replaces | When the loop calls it |
|---|---|---|
| `attach(req)` | the prefix adoption block in `_admit`: `SparseTracker.attach`, `SparsePrefixCache.lookup`/`set_bounds`, `_sparse_warm_draft` | admit |
| `recall(req, target_mass)` | `sparse_selection_recall` | admit |
| `build_rows(rows, seq_q, decodes)` | `_sparse_rows`, `_sparse_decode_rows`, the `do_refresh` / `_sparse_device_select` / `_sparse_prefill_cap` decision | building a tick |
| `decode_rows(decodes, q_dec)` | the eager sparse decode rows | forward |
| `run_decode_graph(reqs, chains)` | `_run_sparse_decode_graph` and the eager fallback | forward |
| `process_offers(dropped_offers)` | `_sparse_process_offers` | mid-loop |
| `finalize(sf, rows, hidden)` | `_sparse_finalize`, offers, `SparsePrefixCache.close_request` | release |
| `transfer_to_shared(...)` | `_sparse_transfer_to_shared` and its freeze/share refs | release |
| `warm_draft(r, entry, matched)` | `_sparse_warm_draft` | admit |
| `offer_drop(r, page, draft_pages)` | the sparse offer/drop bookkeeping | mid-loop |
| `sparse_retier(keep)` | `Engine.sparse_retier` | public #500 manual seam; thin delegation, kept |

`route`/`sparse_on`, the `--sparse-min-tokens` regime split, `_sparse_hot_headroom`
and `_sparse_live_stats` stayed on `Engine`.

A dense-only engine holds `None`, and the dense path contains no sparse branch. This is the
largest move and runs last, behind a device gate.

## Target sizes (estimates, to be replaced by the PR diffs)

| Module | Now | Target |
|---|---|---|
| engine.py | 2,305 (step 11) | ~1,500 (core loop + request types) |
| decode_graph.py | 382 | ~450 |
| sparse_engine.py | 1,174 | tracker/forward/index only |
| sparse_runtime.py | 688 (step 11) | the absorbed sparse runtime |
| build.py | – | ~350 (replaces `build_engine` + `cli._build_*`) |
| cli.py | 2,820 | ≤ 900 |
| kv_cache.py / kv_tiers.py | 962 / 991 (step 9) | ~950 / ~950 |

The net reduction comes from deleting dead code, the duplicate builder and removed-feature
residue, not from the moves. A move PR is judged by "no behaviour change". A deletion PR is
judged by lines removed.

## Guardrail first

Before any move, add one structural test, `tests/test_layering.py`. It parses every module
under `src/tilerl` with `ast` and fails on an import from a higher layer, with the layer table above as a
literal in the test; the test is the source of truth once it lands, and a module missing from
the table fails, so every new file is placed on purpose. Today's violations go into an explicit allowlist in that test, and the
allowlist may only shrink. It holds the direction while five people cut code in parallel,
because a wrong import is rejected when written, not caught in review. Its negative control
is an added `from .cli import x` in `memory.py`, which must fail.

## Sequence

Each row is one PR. Each is behaviour-preserving and passes CI (full suite). Rows marked
*device* also need a V100 smoke on the PR head before merge: short think-on tok/s within 2%
of 52 on the same question set; MMLU n=200 against the same gold-question list (pair by question hash, not seed; at n=50 the binomial half-width is ~0.13 and cannot gate equality), paired accuracy within 0.05; one unique 32k request answered.

| # | PR | Owner | Gate |
|---|---|---|---|
| 0 | This doc | coordinator | review |
| 1 | `tests/test_layering.py` + allowlist, red on an injected upward import | fixkv | CI |
| 2 | Stale design docs: rev-87's 13 items; archive `arch-review-2026-09-09`, `design-ssd-read-path` (KvTier removed) and completed ownership tables to `docs/history/` | fixmisc | CI |
| 4 | Dead code in cli, server and API routes, grep-proven: the `tilerl pretrain` subcommand (no invoker; `train.pretrain` stays and is gated by test_pretrain), delete `rollout.py`/`run_rollout` (only tests/test_rollout.py consumes it; that test is deleted with it), inline single-use cli helpers | fixmisc | CI |
| 5 | scripts/: delete dead one-off probes, keeping anything a doc, test, CI job or `test_main_selfchecks` glob reaches — **closed 2026-09-15**: the seven-set `scripts/audit_scripts_entrypoints.py` enumeration (incl. the selfcheck glob) reports **0 DEAD of 210**; the 10 hand-run card/deploy/ops/eval tools a file never names are registered in its `MANUAL_KEEP` with a per-tool reason, and `tests/test_scripts_closure.py` fails on any new unreachable, unregistered script | fixmisc | CI |
| 6 | Break the 5 cycles and 3 upward imports; merge `dflash2.py` into `spec.py` (this is the dflash2↔spec cycle fix); move `_rolling_hash` (sparse content hash) out of `kv_cache.py` and `group_map` into `sparse_index.py`; shrink the allowlist | fixkv | CI |
| 7a | `build.py`: one assembler; src callers rewritten in the same PR (no re-export shim) | fixmisc | CI + device |
| 7b | Migrate the 89 script and 21 test-file call sites in numbered batches; each batch is one PR. The ~600-line cap counts authored lines and excludes mechanical call-site rewrites | fixmisc | CI |
| 8 | `cli.py` split: training orchestration → `train.py`, bench commands → `bench.py`; the CLI surface is unchanged (`_EXPECTED_CLI_FLAGS`) | fixmisc | CI |
| 8b | One request path for the THREE NON-STREAM routes only (SSE and /ws stay separate): one completion-wait helper carrying the #590 disconnect poll, one thinking resolver with three thin input adapters (raw dict / MessagesRequest / ResponsesRequest), `_flatten_tools` (2 copies) and `_parse_tool_calls` (move from messages.py) in `prompt.py`; the three frozen non-stream JSON shapes are the gate; streaming/ws excluded by name | fixmisc | CI |
| 9 | `kv_tiers.py`: HostKvPages, ColdSsdFile, DramSnapshots, KvBootStore move together with `_shared_ssd_path`/`_blob_spec`/`assert_spill_writable`/`_nbytes`/`_to_device` and SpillWriteError (engine import edited, no shim); ~12 import edits across engine, 4 test files, trace_256k_spill_time.py | fixkv | CI + device (mmap spill is device-only) |
| 10a | `decode_graph.py`: _DecodeGraph/_SparseDecodeGraph/_CpuSparseGraph + bucket/for/precapture. Functions take the ~12 Engine values they read as explicit arguments — signature construction, not a literal move. Boundary vs 11: decode_graph owns capture/replay keyed by (B,W); SparseRuntime owns when to call. ~1,300 mechanical test rewrites excluded from the cap | fixkv | CI + device |
| 10b | Ledger rows `_memory_rows`/`_measured_peak_bytes` → `memory.py`, taking primitives; `memory.py` must not import `engine` (same-layer legal but entrenching the coupling the split ends). `_sparse_live_stats` stays for 11 | fixkv | CI + device |
| 11 | `SparseRuntime` seam in `sparse_runtime.py` (12 sparse-tick methods; the 7 config/counter fields; frozen SparseCtx of pools/model/backend + named verify/sample/draft-step callbacks; zero Engine references). Tracker attributes stay reachable through the preserved `engine._sparse` facade (proxies + `__getattr__`); Engine retains scheduling predicates, `_hybrid_charge`, `sparse_retier`, live stats. Boundary prep shared the graph pad/pool holder (`GraphCapture`) with the dense path | fixkv | CI + device (+ one unique 128k request) |

The serving core has no dead code: a grep inventory of engine, sparse_engine and kv_cache found a caller for every symbol, so its gains come only from the moves. Order reason: deletions (4–5) shrink what the moves have to carry. Cycle breaks (6) make the
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
