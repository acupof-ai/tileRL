# Bench inventory — 2026-09-09

What every bench-shaped script in `scripts/` measures, whether it still runs,
and whether the ruler table keeps it. The ruler itself is
`scripts/bench_harness.py` (`tilerl bench`): seven suites, a `Gate`, and a
baseline JSON keyed `(suite, shape, target)`. This inventory is the
convergence step — no 33rd bench script.

## Scope and method

On `origin/main` (90f308a): **32 `bench_*.py`** (including `bench_harness.py`
itself), **12 `ab_*.py`**, `baseline.py`, 5 `_sweep_*`/`_matrix_gemv.py`
drivers, and 5 `_pod_*.sh`/`_ab_*.sh` launchers — 55 files. All 44 Python
bench scripts compile; their `tilerl`/`tilerl_kernels` imports were
spot-checked against the tree and resolve.

The shared checkout `/Users/bytedance/code/tileRL` holds a *different*, older
set (17 `bench_*.py`, including `bench_bf16_gemv.py`, `bench_decode_dispatch.py`,
`bench_sgl_kernel.py` that do not exist on `origin/main`). Everything below is
about `origin/main`.

"Runs on" is read from each script's own usage line; GPU scripts were not
executed here (no card on this machine).

## Infrastructure — the ruler itself (keep)

| File | What it is |
|---|---|
| `bench_harness.py` | `tilerl bench`: suites decode-kv / prefill / kv-reuse / spec / train / train-full / accuracy, `Gate` against `docs/experience/wins/bench-baseline.json` |
| `benchkit.py` | A/B plumbing: `timeit`, `relerr`, `ab`, engine settle/drive helpers shared by the family scripts |
| `baseline.py` | Merges the pod's baseline JSON into the repo's (`pull`/`show`/`selfcheck`) |

## Live measurements that feed the ruler table (keep)

| File | Quantity | Target | Table row |
|---|---|---|---|
| `bench_prefill.py` | prefill wall vs context, cold cache (the n² path) | cuda sm70, 27B | prefill tok/s @ length |
| `bench_ctx_decode.py` | steady decode rate vs context length, no prefill in window | cuda sm70 | decode vs depth |
| `bench_decode_steady.py` | steady ms/tick, first-3 skipped (the 548 s JIT trap) | cuda sm70 | decode tok/s |
| `bench_decode_b8.py` | B=8 aggregate tok/s, continuous batching | cuda sm70 | decode B=8 |
| `bench_b1_decode.py` | B=1 decode via the server, two-point slope (prefill cancels) | server | decode B=1 (low weight) |
| `bench_workloads.py` | decode tok/s + spec tokens per trunk forward across coding/dialogue/thinking/long-context | cuda | decode + spec |
| `bench_batch_decode.py` | decode vs B on slice4, `--draft`/`--depth` — the runner behind the H20 B=1 spec decode table (#349) | cuda | decode + spec |
| `bench_chat_reuse.py` | multi-turn prefix reuse vs a live server, store hit counters | server | prefix reuse (the 19x) |
| `bench_chat_cold_warm.py` | one prompt, cold vs warm arm, one restart per arm | server | prefix reuse |
| `bench_ssd_restart.py` | restart faults the prefix off disk vs second-run warmth | server + SSD | SSD restart (the 2.041x) |
| `bench_tier_wall_clock.py` | per-turn wall, tier off vs on, across session counts | server + SSD | SSD tier |
| `bench_chat_interleaved.py` | N conversations interleaved — the workload a snapshot tier must win | server + SSD | SSD tier workload |
| `bench_c4_ppl.py` | wikitext-103 perplexity, teacher-forced | cuda, 27B | quality (second point after MMLU) |
| `bench_api_routes.py` | per-request HTTP+render+parse cost over a scripted engine | cpu (real number pending-remote) | serving overhead |

Emitter coverage (2026-09-09): every metric with weight > 0 has a collector.
`mmlu_pct` and `kernel_ms` have none **by design** — both are weight 0.0, so an
unmeasured weight-0 metric is correct prioritization, not debt; do not build
emitters for them before a weight > 0 metric needs one. The debt line is
"weight > 0 and no emitter"; there is none as of this date.

## SSD/tier support measurements (keep while the tier is shipped)

| File | Quantity |
|---|---|
| `bench_ssd_bandwidth.py` | spill read bandwidth — settles the 198.9 MiB/s vs 548.9 MiB/s contradiction |
| `bench_write_through.py` | write-through cost inside the publishing prefill (the 45.3% → 8.96% fix) |
| `bench_h2d.py` | pinned H2D/D2H bandwidth at the real GDN snapshot size (149.6 MiB) |
| `bench_pin_cost.py` | where a DRAM demotion's time goes: pin vs copy |
| `bench_prefix_state.py` | prefix-boundary snapshot cost and survival count, both store kinds |

## Conditional verdicts — keep until the named change lands

A verdict that hangs on a condition we plan to change is not settled: the
moment the condition moves, these scripts rerun first.

| File | Conditional verdict |
|---|---|
| `bench_tp.py` | TP loses to DP **until a capturable all-reduce exists** — a named, planned change |
| `ab_draft_depth.py` + `ab_w1_baseline.py` | spec depth verdict has flipped three times (V100 depth 1 wins, H20 every depth loses, eager build every depth wins 1.8x) — a live question, not a verdict |
| `_sweep_fp8_prefill.py` | sole source for `_FP4_BLOCK_N = 64` in `kernels_linear.py:146` — keep until the prefill tile size is retuned |

## One-shot, verdict in the tree (deletion candidates)

| File | Verdict already recorded |
|---|---|
| `bench_fp4_gemv.py` | fp4 GEMV vs padded WGMMA — shipped, 1.3–5.9x (CHANGELOG 2026-08-24) |
| `bench_fp8_gdn.py` | native fp8 GDN projections — shipped, 1.48x (`wins/2026-08-25-native-fp8-weights.md`) |
| `bench_fp8_prefill.py` | fp8 prefill path — shipped, 1.5x (CHANGELOG 2026-08-24) |
| `bench_fp8_qkvz.py` | qkv+z concat fusion — `wins/2026-08-26-fp8-qkvz-fusion-prefill.md` |
| `bench_fp8_split2.py` | k_split 1 vs 2 — verdict in CHANGELOG (2026-08-26) |
| `bench_gdn_prefill.py` | fused GDN chunk vs eager — shipped, 49.8x (CHANGELOG 2026-08-24) |
| `bench_gemv_gap.py` | direct-vs-backend roof gap — `errors/2026-08-27-decode-latency-bound-not-bandwidth.md` |
| `bench_paged_attn.py` | naive vs FlashAttention — shipped, 83x (CHANGELOG 2026-08-24) |
| `bench_qwen38_baseline.py` | 27B serving baseline — superseded by harness decode-kv/prefill suites |
| `bench_smoke.py` | benchkit's own smoke check — CUDA-only, cannot move to CI (CI forbids perf steps); deleted, parity gated by the CPU twin tests |
| `ab_fp8_gemv.py` | flat vs grouped prefetch — `wins/2026-08-25-native-fp8-weights.md` |
| `ab_smallm_gemv.py` | small-M GEMV — `wins/2026-08-26-batch-decode-h2.md` |
| `ab_prefill_ncols.py` | in-process ncols A/B — `wins/2026-09-03-ncols2-is-1.5x-on-the-verify-path.md` |
| `ab_scale_f16.py` | f16 scale plane — `wins/2026-09-02-f16-block-scales.md` |
| `_sweep_gemv.py`, `_sweep_gemv3.py`, `_matrix_gemv.py` | parameter sweeps whose settings shipped |

## Investigation series — keep until the sm70 prefill work closes

The sm70 prefill n² kernel is the current main line (49.2% of TTFT recoverable
at 16k). These are its tools, not one-shots:

| File | Question |
|---|---|
| `ab_gemv_variant.py` | candidate GEMV variants at M=1/8/32 — five candidates died here, the arena stays |
| `ab_gemv_npartition.py` | n_partition vs X re-reads — died at 1.01x, arena for the next candidate |
| `ab_gemv_ablate.py` | M=32 gap by ablation (no perf counters on the pod) |
| `ab_gemv_l1_knee.py` | L1 capacity hypothesis at the M-path |
| `ab_gemv_xh_m32.py` | rung-8 cliff: hardware vs missing xh flag |
| `bench_gemv_micro.py` + `_sweep_gemv_micro.py` | the (micro_size_k, GROUP) grid sweep, no model load — the active tuning tool |

## Launchers

| File | Serves |
|---|---|
| `_pod_bench.sh` | generic pod runner — keep |
| `_pod_baseline_launch.sh`, `_pod_bgraph_bench.sh`, `_pod_sweep_fp8.sh`, `_ab_batch_launch.sh` | one deleted script each — delete with their script |

## Tally

- Keep: 3 infra + 14 table feeders + 5 SSD support + 7 sm70 series + 4 conditional verdicts + 1 launcher = **34**
- Delete candidates: **21** (10 bench, 4 ab, 3 sweep/matrix, 4 launchers)

Deletions executed in the step-5 PR: the 21 files below are gone. Each verdict
above stays checkable against the CHANGELOG line cited — the scripts were the
runners, the entries are the record. `bench_smoke.py` was listed as "moves to
CI" and does not: it is CUDA-only and CI policy forbids perf steps
(`.github/workflows/ci.yml` header); its kernel's parity has been gated by the
CPU twin tests since the kernel landed. `_sweep_fp8_prefill.py` was listed for deletion and stays: it is the sole source for `_FP4_BLOCK_N = 64` in `kernels_linear.py:146` — a published kernel constant explaining itself by a deleted file is a comment nobody can recompute.

One correction to the first draft, worth its own line: `bench_batch_decode.py`
was listed as "superseded by harness decode-kv" and is not — the harness suite
has no `--draft`, and #349's H20 B=1 spec decode table ran this script on the
same day. "Script X is covered by Y" is a claim that needs a run, not a
reading of two docstrings.

## Deferred (not this round)

- `prereg.py` ledger sharding: at three rows on day one and a rebase per
  concurrent append, per-day files (`prereg/2026-09-09.jsonl`) or one file per
  run id would remove the conflict. Recorded for after the ruler lands.
