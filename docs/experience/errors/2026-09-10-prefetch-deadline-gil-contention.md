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

A GIL-contention test's background workload must produce GIL handoff points.
One big tensor's `torch.load` is a single `read()` with the GIL released,
which a busy main thread cannot slow — the control arm sees no contention
(CI macos-14, 2026-09-10: busy 1.3 ms vs yielded 0.4 ms, the negative control
correctly refused to sign). Many small storages churn the GIL once per file;
that is also the real reader's workload shape — the 27B snapshot is many
per-window files, not one blob.

## CPU deadline margin too thin (second instance of the same shape)

Status: fixed 2026-09-10 by the spin-until-ready loop (72d83303,
[wins entry](../wins/2026-09-10-spin-until-ready-fixes-cpu-prefetch-flake.md)):
the e2e flake went 3/16 → 0/20. The fix spins `sleep(0)` while a fetch is in
flight instead of yielding a fixed N times per tick — the sweep below shows N
is a distribution tail, and the spin is N=∞ with an early exit plus a 50 ms
wall-clock bound as a safety valve. Open remainder (carried in OPEN.md): the
spin's cost on 27B is unmeasured (the 144 MiB snapshot needs many windows, so
the spin actually runs there), and `any_fetching()` is global, not per-request.

The fix above landed the GIL yield, but on CPU the deadline was still thin:
`tokens / seed_rate` = 2.56 s at 192 tokens, while the GIL-starved fetch took
1.2–2.4 s on a loaded box. The e2e test
`test_a_prefetched_hit_reads_nothing_on_the_calling_thread` flaked (3/16 on a
dev machine, clean main — pre-existing, not introduced by the fix). The test's
clock was hardened (`_drain_clock`) but the engine's deadline was not — the
same shape as the original bug, second instance.

Length sweep (192/384/768, 4 runs each): deadline 2.56/5.12/10.24 s, fetch
1.2–2.4 / 0.8–3.3 / 2.0–6.6 s. Margin grows 1.05x → 1.55x then plateaus,
because the fetch is partially byte-bound (768's fetch is ~4x 192's).
Lengthening the prompt alone does not fix it (384 still flaked at run 7).

N-sweep (yields per tick vs `fetch_ms`, 192 tokens, 2 runs each): N=1 →
1529/1697 ms; N=5 → 507/588; N=10 → 96/2; N=20 → 2/189; N=50 → 2/2. The
fetch reaches the uncontended floor (2 ms) at N≥10. The reader misses most
`sleep(0)` windows (it is mid-I/O, not GIL-blocked, when the yield fires),
so N=1 catches a window only ~once per 14 ticks; N=20 catches one within
~1 tick. The sweep could not price a fixed N (2 runs per cell, and the tail
decides the flake), so the shipped fix spins until the fetch parks with a
50 ms wall-clock bound — N=∞ with an early exit, no tuned parameter.
