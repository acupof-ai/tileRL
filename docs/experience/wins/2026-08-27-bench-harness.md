# tilerl bench harness — the perf baseline gate, 2026-08-27

> Status: Shipped (CPU + cuda)

## Context

Perf coverage had holes: decode timed at ONE shallow KV depth, prefix cache
never perf-tested, training zero perf coverage, 15 scattered bench scripts
with no snapshot. `scripts/bench_harness.py` (via `tilerl bench --suite`)
runs decode-vs-KV-depth (512/2k/8k/32k × B=1,8), prefill curve, kv-reuse
hit-rate + warm/cold speedup, train_step tok/s, gated against
`docs/experience/wins/bench-baseline.json`: ≥0.97× passes, beating the
snapshot auto-raises it, first run seeds. CPU rows are report-only (±4%
noise); GPU rows hard-fail.

## What Worked

- Decode DOES drop with depth: B=1 **51.9 → 48.2 → 39.7 → 23.2 tok/s** at
  512/2k/8k/32k (baseline seed: 54.2/51.2/41.5/23.9 with the fusions) — paged-attention KV read is the second lever after GEMVs.
- Prefix cache confirmed real on GPU: only after sizing the pool to hold the
  pinned prefix plus two live requests (the serving 256-block pool evicted the
  entry before the warm request → hits 0; that was a harness bug, not engine).
- Stability: run-to-run 0.4% on a quiet host. Host contention (another
  tenant's nvcc) moved B=8 rows 60% with no GPU sharing — loadavg is stamped.
- Six sizing bugs found by running it: prefix store pins finished prompts
  (2× pool headroom), max_total_tokens vs gen length, settle lifetime at
  B=8 (row 0 finished before row 7 prefilled), prefill 8k vs pool, 27B train
  fp32 masters = 108 GB > one H20 (train row runs tiny; 27B pending-remote).

## Rule

One timing job per host. Rebuild the engine per row. A row that can't reach
pure decode is printed as skipped, never silently dropped.

## Results

See `bench-baseline.json` (cuda rows seeded this date) and `/work/bench_gpu.log`.
