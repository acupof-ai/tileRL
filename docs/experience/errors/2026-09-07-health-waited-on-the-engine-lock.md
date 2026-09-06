# `/health` waited on the engine lock: 87.66 s on a liveness endpoint — V100 sm70, 2026-09-07

> Status: fixed — **verified live on the V100 (`7d5700b`, child pid 2895338) during a real
> 30,000-character prefill: `/health` median 1.24 ms, min 1.16, max 1.36, against 8.12 s median
> and 87.66 s max before the fix (~6,500x on the median) and indistinguishable from the 1.45 ms
> idle baseline.** Engine state at the time: `running=1`, `blocks_used=939`,
> `pool_used_blocks=940`, `slots_used=1` — so the samples were taken against a genuinely busy
> engine, which is the whole point: an idle reading is what had me call this work non-urgent
> earlier the same day. 12 samples, each from a separate ssh so the poll could not be serialized
> behind the request by a shared connection. The request finished normally afterwards
> (`finished=1`, `blocks_used` back to 0). Deployment verified by clock, not by sha: process
> start epoch 1788734914.534 against `engine.py` mtime 1788734911.942, so the child began 2.59 s
> after the post-fix source was written, with `__pycache__` cleared and `find src -name '*.pyc'`
> returning 0.

## Context

Measuring question 3's agent turns, the pool sampler recorded its own `/health`
latency per row. During a 21,727-token prefill on the live V100 (`09e1e84`):

    /health seconds:  min 0.002   median 8.12   max 87.66

Against **0.002 s on the same endpoint idle** — four orders of magnitude, on
identical code. A liveness endpoint that takes 87 s is not reporting liveness.

This is a **second** defect on the same symptom as #208 and it survived that
fix. #208 was the event loop: `/v1/messages` was `async def` and polled a
blocking wait, so every route starved. That is fixed, and `/health` was still
slow — because the remaining blocker is not the loop, it is the lock.

## Root Cause

`stats()` took `self._lock` (`engine.py:731` before the fix), and `step()`
holds that same lock across the **whole forward**:

```python
def step(self) -> None:
    with self._lock:
        decodes, prefills, chunks = self._build_plan()
        ...
        self._run_forward(decodes, prefills, chunks)
```

So a `/health` call arriving mid-tick waits for a forward to finish. On a 43-chunk
prefill it waits for whichever chunk is in flight, and the 87.66 s max is a
queue of those waits.

`peek()` (`engine.py:623`) had already solved this shape and documented why —
"deliberately lock-free: `step()` holds `_lock` across the whole forward, so any
reader that took the lock would block for the entire generation (measured: one
blocked call covered 325 ms of a 335 ms run)". `stats()` is the same kind of
reader and did not get the same treatment.

## Fix

The loop publishes a snapshot; `stats()` returns it without the lock.

- `_build_stats()` is the old locking body, unchanged.
- `step()` assigns `self._stats_snapshot = self._build_stats()` **before** the
  forward and again in a `finally` after it.
- `stats()` returns the published dict. The dict is only ever *replaced*, never
  mutated, so a reader sees one consistent generation; assignment is a single
  bytecode under the GIL, which is `peek`'s own argument.

A stale read is the deliberate trade on the served path: the snapshot is at most
one tick old, and `/health`'s job is liveness, not a transaction.

Two details that are load-bearing, both found by a red gate rather than by
reading:

**The publish must happen before the forward, not only after.** With only the
trailing publish, the *first* forward has no snapshot, `stats()` falls back to
the locking path, and waits. The gate measured **106 s** against a 2 s sleep in
that state. The trailing publish still earns its place: `_loop` stops calling
`step` once nothing is running, so it is the only publish that can carry what
the last tick left — including what a *failed* forward left, which is why it is
in `finally` and not after the `try`.

**A direct-drive caller must get live numbers, not the snapshot.** With no loop
thread there is no background forward to wait on, so the snapshot buys nothing
and costs correctness. `stats()` therefore returns `_build_stats()` when
`self._thread is None`. This was not foreseen: the full suite went red on
`test_a_rejected_submit_does_not_release_the_prefix_stores_blocks`, which
submits and reads between its own `step()` calls and got the previous tick's
counters — `no prefix hit; the test is inert`, on a hit that had happened.

`DataParallelEngine.stats()` needs no change: it calls each replica's `stats()`,
so it inherits the lock-free path.

## Gate

`test_health_does_not_wait_on_the_engine_lock` — a **real** `Engine`, because
the property under test is which lock `stats()` takes and a double that
reimplements `stats()` would assert its own behaviour. `_run_forward` is
replaced by a 2.0 s sleep, 20x the 100 ms assertion; that is the only
substitution and it sits a layer below the one being measured. The gate waits
on an event set *inside* the fake forward, so it cannot pass against an engine
that was never busy.

Three controls, each run separately, each red on the elapsed assertion:

| mutation | result |
|---|---|
| `stats()` reads through the lock again | **2.01 s** — exactly the injected sleep |
| the pre-forward publish removed | **4.02 s** — waits for two forwards |
| the served path forced onto the direct-drive branch | red — so the green is not coming from the fallback |

The third control is the one worth keeping: after adding
`if self._thread is None`, a green gate could have meant the served path was
quietly taking the fresh-build branch and the snapshot was never exercised.
Forcing that branch reds the gate, so the 4.15 s green is the snapshot read.

`441 passed, 14 skipped`. `ruff check` clean.

## Rule

**A reader that takes a lock a forward holds is not a reader, it is a second
writer's queue.** `peek` had the argument and the measurement written down;
`stats` was the same shape and did not get it. When one accessor on a hot object
is deliberately lock-free, the next accessor added needs a reason it is not.

And: **an idle measurement of a contended path is the unrepresentative one.**
Earlier the same day I read `/health` at 9 ms on a freshly restarted endpoint
and called this PR non-urgent on that basis. The 9 ms was real and told me
nothing — the defect only exists while a forward is running.
