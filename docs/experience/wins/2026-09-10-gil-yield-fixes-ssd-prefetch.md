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
being active (`PrefixStore.has_ssd`). The yield is **outside the engine lock**
on both the idle and work paths, so a reader thread that ever takes `_lock`
during a load will not block on it. Two yield points, each at most once
per tick:

1. After the idle early-return (the row is held waiting for fetch)
2. After the forward (lets the reader thread run between ticks)

When the SSD tier is off (the default, and all RL/serve paths without
`--ssd-path`), the condition is False and the yield never executes — zero
overhead.

Measured on H20 card 3, 2026-09-10, tree this branch:

| Metric | Before fix | After fix |
|---|---|---|
| `ssd_prefetches` | 1 | 1 |
| `ssd_fetches_ready` | 0 | **1** |
| `ssd_hits` | 0 | **1** |
| `ssd_fetch_drops` | 1 | **0** |
| `fetch_ms` | 70 | **1** |

Decode throughput (slice 4 layers, fused+graph, **SSD off** — yield never executes):

| B | Without yield (tok/s) | With conditional yield (tok/s) | Delta |
|---|---|---|---|
| 1 | 860.4 | 840.2 | −2.3% (noise; the yield line does not run) |
| 8 | 3176.4 | 3183.6 | +0.2% (noise) |

The SSD-off delta is noise by construction: `has_ssd` is False, so the yield
never executes. The −2.3% at B=1 is system jitter on a 1.2 ms tick, not the
fix.

Decode throughput (same slice, **SSD on** — yield executes every tick):

| B | SSD off (tok/s) | SSD on (tok/s) | Delta |
|---|---|---|---|
| 1 | 860.4 | 799.9 | −7.0% |
| 8 | 3176.4 | 3082.4 | −3.0% |

The yield costs ~80 µs/tick. On the full 27B model (10–24 ms ticks) this
extrapolates to ~0.3–0.8%. The SSD-on arm also pays spill writes and tier
bookkeeping, so the delta is the tier's total cost, not the yield alone.

## Rule

A deadline computed from device speed assumes the fetch runs at device
speed. When the fetch runs in a different thread, GIL contention can make
it 175x slower. Yield the GIL on the tick that holds the row, and gate the
yield on the tier being active so the no-SSD path pays nothing.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-10 | this PR | H20 card 3 | sm90 | 27B-nvfp4-slice4 | — | 1.19 (B=1, SSD off) | 840.2 |
| 2026-09-10 | this PR | H20 card 3 | sm90 | 27B-nvfp4-slice4 | — | 2.51 (B=8, SSD off) | 3183.6 |
| 2026-09-10 | this PR | H20 card 3 | sm90 | 27B-nvfp4-slice4 | — | 1.25 (B=1, SSD on) | 799.9 |
| 2026-09-10 | this PR | H20 card 3 | sm90 | 27B-nvfp4-slice4 | — | 2.60 (B=8, SSD on) | 3082.4 |

Raw artifacts: probe output `/work/probe-cond.log`, bench output
`/work/bench-cond.log` and `/work/bench-ssd-on.log` on the pod.
