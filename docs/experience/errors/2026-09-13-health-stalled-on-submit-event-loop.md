# /health stalled under burst again — this time the chat route froze the event loop on submit

> Status: fixed in code, **after-number pending-remote** — cc redeploys and repeats the
> 4x7.4k burst, counting non-200/empty `/health` responses exactly as the before arm.
> Before: **4 no-response probes in ~50 min** of V100 soak load (cc, 2026-09-13).

## Context

After the 2026-09-07 fix ([2026-09-07-health-waited-on-the-engine-lock.md](2026-09-07-health-waited-on-the-engine-lock.md))
`/health` is lock-free: `step()` publishes `_stats_snapshot` before and after every
forward, `stats()` returns it. That fix was verified live on the V100 at 1.24 ms
mid-prefill. In cc's next soak, `/health` nevertheless returned no response 4 times in
~50 minutes under a 4-conversation, 7.4k-token burst.

## Root Cause

A different blocker on the same symptom. `/health` is a synchronous route, so Starlette
runs it in its worker threadpool and the lock fix genuinely unblocked it there. But the
OpenAI chat route and `/ws/chat` are `async def`, and they called `_submit(req)` **on the
event loop thread**:

- `engine.submit` takes `step()`'s `RLock` (`engine.py`) to enqueue the request.
- During a long prefill `step()` holds that lock across the whole forward.
- A chat/WS request arriving in that window blocks the **single** event loop on lock
  acquisition; while the loop is parked, no route handler — lock-free `/health` included —
  even starts.

`/v1/messages` and `/v1/responses` already ran their whole blocking handler through
`asyncio.to_thread` (gate `test_a_request_in_flight_does_not_freeze_the_server`, which
names a 10-minute V100 freeze of this exact class). The OpenAI chat route only wrapped the
*wait* (`_await_completion`); the admission call and, on the non-streaming tail,
`engine.stop_text`/`engine.logprobs` (both take the same lock) still ran on the loop.

The code read alone could not prove this was the soak's cause rather than another path, so
the gate was required to be red on unchanged main first: with a double whose `submit`
sleeps 1.4 s (a step holding the lock), both the chat and ws arms measured
**1.41 s to a `/health` answer** — the loop parked for the full hold, failing the 1 s
assertion. A green-on-main result would have sent the investigation elsewhere.

## Fix

Every engine call that takes `step()`'s lock, reached from an `async def` in
`server.py`, goes through `asyncio.to_thread`:

- chat route: `_submit` (admission), and the non-streaming tail `stop_text`, `logprobs`;
- `/ws/chat`: `_submit`.

The synchronous SSE generator (`_deltas`/`_stream`) is unchanged — it already runs in a
worker thread via `StreamingResponse`/`to_thread(next, ...)`, so its `stop_text` is off
the loop. `/health` is unchanged.

## Gate

CPU, `tests/test_health_submit_lock.py`: a request in flight whose `engine.submit` holds
for 1.4 s; `/health` must answer 200 within 1 s, on both `/v1/chat/completions` and
`/ws/chat`. Red on main at 1.41 s for the named reason (loop blocked on lock
acquisition), green after the fix. The sibling wallclock gates keep their CI skip/local
bound for the same machine-load reason recorded in
`errors/2026-09-11-flaky-wallclock-test-inventory.md`; these two use a 1.4 s hold against
a 1 s bound — 1.4x, not a GIL-ratio band — and run in CI.

## Rule

Making a route lock-free does not make it reachable: any blocking call on the async event
loop stalls every route, regardless of that route's own locking. When an endpoint is
slow/empty under load, check which **thread** every sibling async handler blocks in before
re-fixing the endpoint itself. A route fixed once for one blocker (the lock) can show the
same symptom from the next blocker (the loop).
