# Spin-until-ready on the fetch path fixes the CPU prefetch deadline flake — sm90/cpu, 2026-09-10

> Status: Shipped

## Context

The GIL yield from [the prefetch fix](2026-09-10-gil-yield-fixes-ssd-prefetch.md)
left a CPU flake: one `sleep(0)` per tick gives the reader thread a GIL window
only ~once per 14 ticks (it is mid-I/O, not GIL-blocked, when most yields fire),
so a fetch takes 1.2–2.4 s on CPU against a 2.56 s deadline at 192 tokens. The
e2e test `test_a_prefetched_hit_reads_nothing_on_the_calling_thread` flaked
3/16 on a loaded dev machine (clean main, pre-existing).

See [errors/2026-09-10-prefetch-deadline-gil-contention.md](../errors/2026-09-10-prefetch-deadline-gil-contention.md)
for the full root cause and the sweep data.

## What Worked

While a fetch is in flight, spin `sleep(0)` until the fetch parks or a 50 ms
wall-clock bound expires. The unconditional once-per-tick yield stays (it feeds
the writer thread). The spin is strictly additive on the fetch path.

The bound is a safety valve for a stuck reader, not a tuned parameter: 50 ms is
10x the healthy case (~5 ms, one switch interval) and 2/3 of the CUDA deadline
(75 ms). Worst-case tick inflation is 50 ms (40x on a 1.25 ms CUDA B=1 tick),
and it fires only when the reader is stuck.

## Measurements

Flake rate, 192 tokens, the two prefetch e2e tests, 20 runs each:

| Build | Failures |
|---|---|
| Before spin (clean main) | 3/16 |
| After spin | **0/20** |

Decode throughput (slice 4 layers, fused+graph, SSD on — the spin executes only
while a fetch is in flight, which is no ticks in this bench; the numbers confirm
the spin exits immediately and costs nothing on CUDA):

| B | Before spin (tok/s) | After spin (tok/s) | Delta |
|---|---|---|---|
| 1 | 799.9 | 799.1 | −0.1% (noise) |
| 8 | 3082.4 | 3076.8 | −0.2% (noise) |

The spin's cost is zero on CUDA by construction: the fetch completes in one GIL
window (~1 ms), the loop sees `any_fetching()` go false, and exits. The N-sweep
that sized this (yields per tick vs `fetch_ms`): N=1 → 1.5–1.7 s, N=10 → 2–96 ms,
N=20 → 2–189 ms, N=50 → 2 ms. The spin is N=∞ with an early exit, so it takes
the fast path of the N≥10 cells without the per-tick cost of a fixed N.

## Rule

A single yield per tick feeds a background thread only if the thread is
GIL-blocked at the yield instant. A thread doing I/O misses most windows. When
the background work is on the critical path (a fetch a row is waiting for),
spin until it completes, with a wall-clock bound as a safety valve — not a fixed
yield count tuned from a distribution's tail.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-10 | this PR | H20 card 3 | sm90 | 27B-nvfp4-slice4 | — | 1.25 (B=1, SSD on) | 799.1 |
| 2026-09-10 | this PR | H20 card 3 | sm90 | 27B-nvfp4-slice4 | — | 2.60 (B=8, SSD on) | 3076.8 |

Raw artifacts: bench output `/work/bench-spin-ssd.log` on the pod.
