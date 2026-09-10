# GIL yield in step loop fixes SSD prefetch abandonment on CUDA — sm90, 2026-09-10

> Status: Shipped

## Context

`test_a_prefetched_hit_reads_nothing_on_the_calling_thread` failed on CUDA
with `ssd_recovered=1, ssd_hits=0`. The prefetch was triggered but never
completed: the 75 ms deadline (192 tokens ÷ 2558.6 tokens/s seed rate)
expired before the fetch finished, because GIL contention with the main
thread's `step()` loop made `torch.load` take 71.7 ms instead of 0.4 ms.

See [errors/2026-09-10-prefetch-deadline-gil-contention.md](../errors/2026-09-10-prefetch-deadline-gil-contention.md)
for the full root cause.

## What Worked

Yield the GIL once per tick in `Engine.step()`, conditional on the SSD tier
being active (`PrefixStore.has_ssd`). Two yield points, each at most once
per tick:

1. Before the early return (idle path — the row is held waiting for fetch)
2. After the forward (work path — lets the reader thread run between ticks)

The yield is `time.sleep(0)`: releases the GIL, does not sleep. When the SSD
tier is off (the default, and all RL/serve paths without `--ssd-path`), the
condition is False and the yield never executes — zero overhead.

Measured on H20 card 3, 2026-09-10, tree this branch:

| Metric | Before fix | After fix |
|---|---|---|
| `ssd_prefetches` | 1 | 1 |
| `ssd_fetches_ready` | 0 | **1** |
| `ssd_hits` | 0 | **1** |
| `ssd_fetch_drops` | 1 | **0** |
| `fetch_ms` | 70 | **1** |

Decode throughput (slice 4 layers, fused+graph, SSD off — yield never executes):

| B | Without yield (tok/s) | With conditional yield (tok/s) | Delta |
|---|---|---|---|
| 1 | 860.4 | 840.2 | −2.3% (noise; 1.2 ms ticks) |
| 8 | 3176.4 | 3183.6 | +0.2% (noise) |

For reference, an unconditional yield (before the `has_ssd` guard) cost
−6.2% at B=1 and −2.3% at B=8 on the same slice — the guard eliminates this.

## Rule

A deadline computed from device speed assumes the fetch runs at device
speed. When the fetch runs in a different thread, GIL contention can make
it 175x slower. Yield the GIL on the tick that holds the row, and gate the
yield on the tier being active so the no-SSD path pays nothing.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-10 | this PR | H20 card 3 | sm90 | 27B-nvfp4-slice4 | — | 1.19 (B=1) | 840.2 (B=1) |
| 2026-09-10 | this PR | H20 card 3 | sm90 | 27B-nvfp4-slice4 | — | 2.51 (B=8) | 3183.6 (B=8) |

Raw artifacts: probe output `/work/probe-cond.log`, bench output
`/work/bench-cond.log` on the pod.
