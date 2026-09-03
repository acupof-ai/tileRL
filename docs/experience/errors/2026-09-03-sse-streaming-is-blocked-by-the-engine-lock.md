# Incremental SSE is blocked by the engine lock, not by a missing accessor, 2026-09-03

> **SUPERSEDED the same day by
> [`wins/2026-09-03-incremental-sse-was-a-starved-poll-loop.md`](../wins/2026-09-03-incremental-sse-was-a-starved-poll-loop.md).**
> The conclusion below — that incremental streaming needs `step()`'s lock narrowed — is **wrong**.
> The lock was never touched. `peek()` does not need the lock (GIL-atomic append/copy), and the
> reason it saw nothing is that the poll loop called `take()` every iteration, which *does* take
> the lock. Shipped at 5-6 deltas, 32.4 tok/s streamed vs 32.4 non-streamed. The lock timings
> recorded here are accurate; only the attribution is not. Kept for the rule at the bottom, which
> still holds, and as the record of a wrong attribution.

> Status: **attempted, reverted, gap recorded as a strict xfail.** The demo needs a viewer to
> see tokens arrive; `server.py:245` emits the whole completion as **one** SSE delta. I wrote
> the obvious fix — an `engine.peek()` that copies the live output — and it cannot work:
> **`step()` holds `_lock` across the entire forward**, so no reader ever observes an in-flight
> request. Measured: `take()` blocked **325 ms of a 335 ms generation**, and a 20 ms poll loop
> got **3 polls for 24 tokens**. Both files restored to HEAD.

## What the code said, and what was wrong about it

`server.py:245` carries a ponytail marker: *"engine.poll reports COMPLETED sequences, so the
completion is emitted as one content delta + finish. Incremental token streaming needs an
engine event stream (day-2)."*

I read that as an overestimate, and the first half of my reasoning was right:

- `_Req.output` (`engine.py:134`) is a live list that grows every tick.
- `_await_completion` (`server.py:146`) already loops at 20 ms.

So no event system is needed — an accessor plus a text-delta loop would do. That part held. A
`peek(request_id)` returning `list(req.output)` under `_lock`, plus a `_stream` that decodes the
whole prefix each tick and yields the text difference, was ~35 lines.

**The part I did not check first was whether a reader can run at all while a tick is in
flight.**

## The measurement

Driving `step()` by hand, `peek` works exactly as designed — 1, 2, 3, … 11, then `None` when
the request leaves `_running`. That is the version that made the change look finished.

With `engine.run()` driving the loop in its own thread, as the server does, `peek` returned
`None` **every time** across a 289 ms generation with 14 poll windows. Timing each call:

| t (ms) | `take()` (ms) | `peek()` (ms) | peek len |
|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.0 | None |
| 24.3 | **325.0** | 25.3 | None |
| 398.6 | 0.0 | 0.0 | None |

**One `take()` call blocked for 325 ms** — the whole generation. 24 tokens, **3 polls**.

`engine.py:557`:

```python
def step(self) -> None:
    with self._lock:
        decodes, prefills, chunks = self._build_plan()
        ...
        self._run_forward(decodes, prefills, chunks)
```

The lock spans plan **and** forward. The daemon loop calls `step()` back to back, so the lock is
held essentially continuously while a request is generating; a reader only gets in during the
gaps between ticks, and on this workload it does not get in at all.

So the accessor is unobservable **by construction**, not by a race that better polling would
win. The ponytail marker was right that this is not a small change, and wrong about why —
it is the lock, not a missing event stream.

## Reverted rather than shipped

`src/tilerl/engine.py` and `src/tilerl/server.py` are back at HEAD. A `peek` that always
returns `None` plus a `_stream` written to consume it is a parallel path that does nothing,
which the no-half-states rule forbids, and it would have read as working code.

What narrowing the lock actually requires, so the next attempt starts from the real problem:
release `_lock` around `_run_forward` while keeping `_build_plan` atomic, then establish that a
concurrent `submit` cannot mutate `_running`/`_waiting` under a forward that is reading the rows
it planned, and that `take`/`_finish` cannot pop a request mid-tick. That is a correctness
change to the engine's core with the tape and the KV pools downstream of it — not a demo tweak,
and it needs its own gradcheck-adjacent reasoning about what a tick may observe.

## The test is the artifact

The existing `test_completion_stream` passes with **one** delta: it asserts only that some
content arrived. So it cannot see this defect, and would not see the fix either.

The new test asserts the two properties that pull against each other:

- **more than one** content delta (it is actually incremental), and
- the joined deltas **equal** the non-streamed text for the same deterministic request, with no
  U+FFFD (a per-token decode would split multi-byte UTF-8 — `_ByteTokenizer` is one id per
  *byte*, so any multi-byte character is guaranteed to span tokens).

It is marked `xfail(strict=True)` with the lock measurement in its reason. Strict matters: the
moment streaming starts working the test **fails** for passing unexpectedly, so the marker
cannot outlive the bug silently.

## Rule

**Driving a loop by hand is not the concurrency the code runs in.** `peek` passed a
hand-stepped probe perfectly and returns `None` 100% of the time under the daemon loop the
server uses. The probe was not wrong about the accessor; it was answering a different question
than "can a reader see this while the engine is running". When a change depends on observing
shared state, the first probe has to use the real driver, not the convenient one.

Second: **a marker that names the wrong obstacle costs more than no marker.** "Needs an engine
event stream" sent me at the observability API, which was genuinely easy and genuinely
irrelevant. The obstacle was six lines away in `step()`. A ponytail marker should name the
constraint that actually binds, or say it has not been established.

## Gate

192 tests pass, 1 xfailed (this gap), ruff clean. The lock behaviour was measured under the
daemon loop, not inferred: three probes, the last one timing individual `take`/`peek` calls.
Both source files verified back at HEAD. No GPU used — the whole finding reproduces on the CPU
target with the tiny model.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 73e11fa | Mac | cpu | tiny | `peek` under hand-driven `step()` | works: 1,2,…,11 then None |
| 2026-09-03 | 73e11fa | Mac | cpu | tiny | **`peek` under the daemon loop** | **None, 100% of calls** |
| 2026-09-03 | 73e11fa | Mac | cpu | tiny | **longest single `take()` block** | **325 ms of a 335 ms generation** |
| 2026-09-03 | 73e11fa | Mac | cpu | tiny | polls achieved at a 20 ms interval | **3, for 24 tokens** |
| 2026-09-03 | 73e11fa | Mac | cpu | tiny | cause | **`step()` holds `_lock` across the forward (`engine.py:557`)** |
| 2026-09-03 | 73e11fa | Mac | cpu | tiny | verdict | incremental SSE needs the lock narrowed; accessor reverted |
| 2026-09-03 | 73e11fa | Mac | cpu | tiny | test state | strict xfail, fails loudly when streaming works |
