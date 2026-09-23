# Stats snapshot builds once per steady tick, not twice — V100 sm70, 2026-09-23

> Status: pending-remote

## Context

`Engine.step()` built the published stats snapshot twice per tick: once
between `_build_plan()` and the forward, and again in the forward's `finally`.
The phase window (`2026-09-23-sparse-w2-phase-timing.md`) measured
`step − graph` = 3.3–3.9 ms per 32.9k steady tick on the V100 sm70 27B path,
and the per-tick `stats=3–4 ms` line accounted for essentially all of it.

## What Worked

The two builds read different moments of the tick: `_run_forward()` moves
rows to finished (`running`, `finished`, `slots_used`, `blocks_used`,
`pool_used_blocks`), bumps `decode_forwards`/`prefill_forwards`/
`tokens_generated`, and changes the prefix store. Only the end build carries
that state, so the end build is the authoritative one.

The pre-forward build existed for one reason: the first forward (and the
first tick of a long multi-chunk prefill) otherwise has no published
snapshot, and `stats()` falls back to taking `_lock` while the forward holds
it — `/health` blocked on the engine lock for seconds during a 21.7k-token
prefill (`test_health_does_not_wait_on_the_engine_lock`).

The pre-forward build is now conditional: it fires only when the tick
admitted rows (`_slots_used` grew in `_build_plan`) or `submit()` parked a new
waiting row since the last end snapshot. On a steady decode tick neither
holds — nothing is admitted between ticks and the previous end snapshot is
current — so one `_build_stats()` runs per tick instead of two. Cancel's
own build is unchanged.

Expected saving on the measured workload: one `_build_stats()` call removed
per steady tick, ~1.5–2 ms of the observed 3–4 ms `stats` line (3–6% of the
~47 ms dw2048 step). No stats field's value or semantics change: the
snapshot during a steady forward is the previous tick's end state, which
equals the pre-forward state under the old code.

One accepted visibility change: on a no-admit tick the snapshot served
DURING the forward is the previous tick's end, so `blocks_used`, `finished`,
`prefix_hits` and the other forward-moved counters read one tick stale on
`/health` while that tick's forward runs. That is what a snapshot already is;
only a reader reconciling health numbers against a per-tick log sees an
offset of one. The invariant that is preserved: a snapshot is always
published before a forward runs, so `stats()` never falls back to its
locking path while the forward holds `_lock`.

## Rule

**A lock-free stats snapshot does not need rebuilding inside a tick unless
the tick changed engine state before the forward; the end build covers every
steady tick.** The pre-forward build is admission insurance for `/health`,
not a per-tick necessity.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-23 | pending | V100 sm70 | cuda | qwen38-27b, 32.9k ctx | pending-remote | pending-remote | pending-remote |

CPU regression gate: `tests/test_server.py::
test_stats_snapshot_is_built_once_per_tick_and_carries_tick_end_state` —
tiny engine, one row through completion plus a parked waiting row; asserts
build count == ticks + 2 (one end build per tick, one demand build on the
admit tick, one when the waiter appeared) and the final snapshot equals a
fresh build field-for-field (running/finished/tokens_generated/waiting).

Negative control (run, then reverted): replace the conditional pre-forward
build with the old unconditional one and delete the end build. The gate
failed both ways it must: builds == ticks (no end builds), and the final
snapshot read running=1, finished=0, tokens_generated=5 while a fresh build
read 0, 1, 6 — the snapshot was the tick's PRE-forward state.

Remote measurement (V100, 32.9k steady graph ticks, the phase-window probe)
runs after fixmisc.
