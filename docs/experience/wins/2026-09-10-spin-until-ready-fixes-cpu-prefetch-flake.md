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
10x the healthy case for a small fetch (~5 ms, one switch interval) and 2/3 of
the CUDA deadline (75 ms). Worst-case tick inflation is 50 ms (40x on a 1.25 ms
CUDA B=1 tick). Whether the bound fires on the slice is unmeasured (the bench
below had no fetch in flight during decode); on 27B it will fire regularly —
see Behavioral changes.

## Measurements

Flake rate, 192 tokens, the two prefetch e2e tests, 20 runs each:

| Build | Failures |
|---|---|
| Before spin (clean main) | 3/16 |
| After spin | **0/20** |

Decode throughput (slice 4 layers, fused+graph, SSD on, H20 card 3):

| B | Before spin (tok/s) | After spin (tok/s) | Delta |
|---|---|---|---|
| 1 | 799.9 | 799.1 | −0.1% (noise) |
| 8 | 3082.4 | 3076.8 | −0.2% (noise) |

**These numbers are tautological — the spin never ran in that bench.** The
decode arm had no fetch in flight, so `any_fetching()` was false at every tick
and the spin body never executed. −0.1%/−0.2% prices the `any_fetching()`
check itself, not the spin. The slice's `.st` is 9.8 MB (the 18 KB was the
`.kv`), so a fetch on the slice would not complete in one GIL window — but no
fetch was running, so the bench cannot answer whether the spin exits promptly
on the slice either.

**On the full 27B model the spin will actually run.** The snapshot is 155.2
MiB (states 144.0 + conv_window 11.25), and a 155.2 MiB `torch.load` needs many
GIL windows, not one. Each tick will spin for a meaningful fraction of the 50
ms bound instead of exiting immediately. The slice bench cannot price this;
it needs a 27B measurement with SSD on.

The N-sweep that sized the spin (yields per tick vs `fetch_ms`, 4-layer slice,
`.st` 9.8 MB):
N=1 → 1.5–1.7 s, N=10 → 2–96 ms, N=20 → 2–189 ms, N=50 → 2 ms. The spin is
N=∞ with an early exit, so it takes the fast path of the N≥10 cells without
the per-tick cost of a fixed N.

## Behavioral changes (not cost)

**The spin's time counts against the fetch's own deadline.** The CUDA deadline
is 75 ms and the spin bound is 50 ms, so the first tick's spin eats 2/3 of it.
A fetch that does not complete in the first tick's spin is abandoned at the
second tick. This is fast-fail — the row prefills from scratch instead of
waiting — but it changes the deadline's meaning from "time to start a fetch"
to "time to finish one, spin included".

**`any_fetching()` is global, not per-request.** It is true when ANY fetch is
in flight, not just the one this tick's held row is waiting for. Under
concurrency, an unrelated fetch makes every tick spin. The B=8 −0.2% above
cannot answer this — it was measured with no fetch in flight during decode.

## Known edges

The spin calls `any_fetching()` once per iteration, each taking the KvTier
`_lock` — thousands of acquisitions in a 50 ms spin. The reader's
`_fetching.discard(key)` at fetch completion needs the same lock. The 0/20
flake run is empirical evidence this does not deadlock or stall, but if the
spin ever fails to exit on the real model, lock contention between the spin
loop and the reader's completion path is the first thing to check.

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
