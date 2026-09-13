# Memoize the dense memory ledger off the per-step stats path — V100, 2026-09-13

> Status: shipped. V100 A-B-A confirmed 2026-09-13.

## Context

The #460 memory ledger added `"memory": _memory_rows()` to `_build_stats`,
which runs twice per step under the loop lock (before and after every
forward). The dense engine's ledger is static after build, but each call
re-ran `memory.plan` and walked every materialized param tensor plus the KV
pools on the decode thread. The V100 bisect named #460 (8a985917) first bad:
46.5 → 44.5 tok/s think-off, d1 + captured graph, ±0.1 repeats.

## What Worked

Dense `_memory_rows` results memoize after the first call; the sparse engine
keeps a live ledger because `kv_hot`/`kv_cold` track residency each tick.
CPU gate `test_dense_memory_rows_computed_once_and_stable`: one plan call
across N stats builds, identity-served rows, recomputation after
invalidation still equal. Red before the patch (plan called 6 times).

## Rule

A stats derivation of static allocations belongs to build-time caching, not
the per-tick snapshot — the snapshot runs on the forward thread.

## Results

V100 n37-002-027, Qwen3.8-27B NVFP4, 4/4/8192, draft d1, `--decode-graph`,
one card A-B-A (the patched arm needs its own process):

| arm | decode ms/fwd think-off | tok/s think-off | tok/s think-on |
|---|---:|---:|---:|
| A main 1137f7b0 | 37.8 / 37.0 | 45.3 / 46.4 | 49.0 / 50.3 |
| B eb581ed0 | 34.8 | **49.3** | **53.2** |

`_memory_rows()` ×1000 (warm): A median 0.804 ms/call (p10 0.751, p90
1.034), B 0.001 ms/call. At two stats builds per step that is the measured
~2.9 ms/fwd delta. B smoke: MMLU 0.70, determinism identical, 400 clean
requests.
