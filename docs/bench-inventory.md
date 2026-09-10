# Bench inventory — 2026-09-09 (SUPERSEDED in part)

**SUPERSEDED:** the metric→collector map and the Keep/Delete triage this doc once
held are consumed. The collector map lives in `docs/bench-metrics.json`
(`tilerl bench --collectors`); the 21 deletion candidates were deleted in #378 —
the mapping is exact, verified 2026-09-10 by name-level comparison of the two
lists. #379 deleted 4 `probe_` scripts from the D2 family, which were never on
this list. What remains is the residue the registry cannot hold: why the
non-metric scripts must not be deleted.

## Infrastructure — the ruler itself

| File | What it is |
|---|---|
| `bench_harness.py` | `tilerl bench`: suites decode-kv / prefill / kv-reuse / spec / train / train-full / accuracy, `Gate` against `docs/experience/wins/bench-baseline.json` |
| `benchkit.py` | A/B plumbing: `timeit`, `relerr`, `ab`, engine settle/drive helpers shared by the family scripts |
| `baseline.py` | Merges the pod's baseline JSON into the repo's (`pull`/`show`/`selfcheck`) |
| `_pod_bench.sh` | generic pod runner |

## SSD/tier support measurements (keep while the tier is shipped)

| File | Quantity |
|---|---|
| `bench_ssd_bandwidth.py` | spill read bandwidth — settles the 198.9 MiB/s vs 548.9 MiB/s contradiction |
| `bench_write_through.py` | write-through cost inside the publishing prefill (the 45.3% → 8.96% fix) |
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

## sm70 investigation series — keep until the sm70 prefill work closes

The sm70 prefill **cell** was rejected (#232, TTFT at 16,384 tokens ≥ 5.97x
baseline); the sm70 **target** was not — it is a live execution target on the
roadmap and the contrast point for the speculation line. These are its tuning
tools, not one-shots:

| File | Question |
|---|---|
| `ab_gemv_variant.py` | candidate GEMV variants at M=1/8/32 — five candidates died here, the arena stays |
| `ab_gemv_npartition.py` | n_partition vs X re-reads — died at 1.01x, arena for the next candidate |
| `ab_gemv_ablate.py` | M=32 gap by ablation (no perf counters on the pod) |
| `ab_gemv_l1_knee.py` | L1 capacity hypothesis at the M-path |
| `ab_gemv_xh_m32.py` | rung-8 cliff: hardware vs missing xh flag |
| `bench_gemv_micro.py` + `_sweep_gemv_micro.py` | the (micro_size_k, GROUP) grid sweep, no model load — the active tuning tool |

## Rule: "covered by" needs a run, not a docstring

`bench_batch_decode.py` was once listed as "superseded by harness decode-kv"
and was not — the harness suite has no `--draft`, and #349's H20 B=1 spec
decode table ran this script on the same day. "Script X is covered by Y" is a
claim that needs a run, not a reading of two docstrings.

## Deferred (not this round)

- `prereg.py` ledger sharding: at three rows on day one and a rebase per
  concurrent append, per-day files (`prereg/2026-09-09.jsonl`) or one file per
  run id would remove the conflict. Recorded for after the ruler lands.
