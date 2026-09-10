# The faster the card, the less likely prefetch completes: GIL contention starves the reader thread

## Context

`test_a_prefetched_hit_reads_nothing_on_the_calling_thread` failed on CUDA
(H20, card 3, 2026-09-10) with `ssd_recovered=1, ssd_hits=0`. The SSD entry
was loaded from disk at startup but no request used it. CPU passed.

First observed ≤ 2026-09-10.

**Scope caveat**: the SSD tier is off by default (`--ssd-path` empty) and has
a recorded REJECT on the serve path (1.65x worse per turn at 12 sessions,
0 hits, [errors/2026-09-06](2026-09-06-the-ssd-tier-is-165x-worse-at-12-sessions.md)).
This fix repairs a feature that is disabled by default and was judged worse
in its only measured serving configuration. The defect is real — the prefetch
mechanism never worked on fast cards — but the fix does not make serving
faster in any default configuration.

## Root Cause

The prefetch was triggered (`ssd_prefetches=1`) but never completed
(`ssd_fetches_ready=0, ssd_fetch_drops=1`). The deadline
(`len(tokens) / seed_rate = 192 / 2558.6 = 75 ms`) expired before the fetch
finished, so the engine abandoned it and the row prefilled from scratch.

The fetch itself should have been fast: `torch.load` on the 19 KB of spill
files takes **0.4 ms** in the main thread. But the fetch loop runs in a
daemon thread, and the main thread was in a tight `step()` loop holding the
GIL. Measured on the pod (H20, torch 2.11.0+cu129):

| Scene | Time |
|---|---|
| Main thread `torch.load` | 0.4 ms |
| Background thread, main idle | 0.5 ms |
| Background thread, main busy loop | **71.7 ms** |
| Background thread, main yields (`sleep(0)`) | 0.5 ms |

The reader thread needs the GIL for unpickling; CPython's switch interval
(5 ms) means it acquires the GIL every ~5 ms, and `torch.load` needs several
acquisitions. The result is 70 ms for what should be 0.4 ms of work — and
the 75 ms deadline expires first.

**The mechanism only works where it is useless.** On CPU the seed rate is
75 tokens/s, so the deadline is 2.56 s and the fetch easily completes. On
CUDA the seed rate (2558.6) makes the deadline 75 ms, and the same GIL
contention that the tight step loop creates starves the fetch. Faster cards
get shorter deadlines and the same contention — the prefetch is guaranteed
to lose.

## Fix

Yield the GIL once per tick in `Engine.step()`, conditional on the SSD tier
being active (`PrefixStore.has_ssd`). This brings the fetch from 71.7 ms to
1 ms, well inside the 75 ms deadline. See
[wins/2026-09-10-gil-yield-fixes-ssd-prefetch.md](../wins/2026-09-10-gil-yield-fixes-ssd-prefetch.md)
for the bench data.

An alternative safety net (not the root fix): do not abandon an in-flight
fetch when the deadline expires — the 70 ms is already paid, and the result
is useful for the next request with the same prefix. The deadline should
govern whether to *start* a fetch, not whether to *discard finished work*.

A `fetch_ms` upper bound was measured and rejected as a guard: on CPU the
fetch takes 1185 ms even with the yield (the slow CPU forward gives the
reader one 5 ms window per tick), so it cannot discriminate yield from
no-yield in the only environment CI runs. A guard that is always green in
the only environment that runs it is worse than none.

Reproducible apparatus: `scripts/probe_prefetch_hit.py` (the three counters
in the test scenario) and `scripts/probe_torch_load_overhead.py` (the
four-scenario GIL table). One permanent guard came out of them:
`test_a_yielded_gil_runs_a_background_load_promptly` in `tests/test_kv.py`
(the sleep(0) premise itself, red on a CPython/torch GIL behavior change).

## General finding

Any Python background thread in this engine is silently starved by the
`step()` loop's GIL hold. The SSD tier's `_writer` (flush) and `_reader`
(prefetch) both suffer; the `_reader` was visible because its deadline
fires. The `_writer` has no deadline, so its starvation was invisible.
The GIL yield in `step()` covers both, and comments at both `Thread(...)`
creation points in `kv_cache.py` point here. The next background thread
added to the engine needs the same yield.

## Rule

A deadline computed from device speed assumes the fetch runs at device
speed. When the fetch runs in a different thread, GIL contention can make
it 175x slower. Measure the fetch time in the actual threading environment,
not in isolation.
