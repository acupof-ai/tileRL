# Memoize the dense memory ledger off the per-step stats path — V100, 2026-09-13

> Status: pending-remote. CPU gate green; the V100 A-B-A decode number is
> ops-0b's arm.

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

| date | commit | machine | target | model | config | A ms/tick | B ms/tick | A tok/s | B tok/s |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 2026-09-13 | pending | n37-002-027 V100 | sm70 | Qwen3.8-27B NVFP4 | 4/4/8192, draft d1, graph, think-off | | | | |

Timeit `e._memory_rows()` ×1000 in the A process: `<pending>` ms/call — closes
the PR if far below 0.4 ms/call (the regression is ~1 ms/tick, 2 calls/step).
Raw artifacts: `<server log>`.
