# A mid-stream SSE socket close never cancelled the engine row — 2026-09-15

**Date:** 2026-09-15
**Arch:** target-independent HTTP/SSE; live uvicorn (httptools, ASGI spec 2.3)
**Task:** audit findings 4/10 behavioral cancel gates (F4 was the production defect)

## Context

`POST /v1/chat/completions` with `stream=true` runs a sync generator (`server._stream`)
through Starlette's `StreamingResponse` via `iterate_in_threadpool`. The row
freed its slot/blocks on a socket hang-up only through the generator's
`except GeneratorExit: engine.cancel(request_id)` branch — audit finding 4
noted that was source-read only, no real mid-stream disconnect test existed
(the `/ws/chat` side, finding 10, had the same gap).

A behavioral gate with a real uvicorn socket closed after the first SSE content
frame (engine blocked in its poll loop, still holding 4 blocks + 1 state slot)
proved the branch is **unreachable on a live hang-up**: 5 s after the close,
`engine.cancelled` stayed empty and the worker thread kept polling. The
finding-10 websocket gate over the same close mechanism was already green
(the receive side raises `WebSocketDisconnect` synchronously).

## Root cause

uvicorn's httptools transport reports ASGI `spec_version` 2.3, so
Starlette's `StreamingResponse` takes the task-group branch
(`starlette/responses.py`): the stream iterator runs in a task group alongside
`listen_for_disconnect`. A real `http.disconnect` **does** cancel the stream
task — but `_deltas` blocks inside

```python
await anyio.to_thread.run_sync(_next, sync_generator)
```

and anyio cannot interrupt a sync call already running in the worker thread:
the cancellation is delivered only when that call returns. Separately,
`iterate_in_threadpool` (`starlette/concurrency.py`) fetches the sync
generator through `to_thread` and **never acloses it**, so throwing
GeneratorExit into the sync generator mid-flight depended on GC or process
teardown — nondeterministic and absent while the client was actually gone. The
orphaned row ran to `max_new_tokens`, holding its blocks and state slot for
nobody.

This is the exact distinction the #598 gate
(`test_a_real_http_disconnect_event_cancels_without_task_cancellation`)
documented for the non-stream routes: cancelling the ASGI task and receiving
`http.disconnect` are different events. Those routes poll a fresh
`request.is_disconnected()` through `await_or_cancel`; the SSE body had no
equivalent watcher.

## Fix

`server.stream_or_cancel(request, engine, request_id, body)`: fetch one
`next(sync_gen)` at a time through `asyncio.to_thread`, with the same 0.05 s
poll as `await_or_cancel`, and poll a FRESH `request.is_disconnected()` while
each fetch blocks. On disconnect it calls `engine.cancel(request_id)` (idempotent:
a row that finishes the race wins, cancel on it is a harmless no-op) and simply
returns — the SSE 200 response headers are already committed, so the stream
ends on a closed socket rather than a 499 (499 is the NON-stream route only:
its disconnect happens before `await_or_cancel` returns any response).
Normal completion returns before any disconnect tick can cancel a finished
row. The old `except GeneratorExit`
branch in `_stream` is retained and re-documented as GC/process-teardown
defense-in-depth — the live path is the watcher, and the comment no longer
claims a client hang-up reaches it. The ws route is unchanged: its
`WebSocketDisconnect` path already calls `gen.close()` + `engine.cancel`.

Gates (both in `tests/test_server.py`):

- `test_a_mid_stream_sse_close_cancels_and_releases_the_row` — real uvicorn
  server, real socket, close only after `"content"` reached the wire; asserts
  `engine.cancelled == [rid]` within 5 s and the held blocks/slot return. Red on
  the old code (proved before writing the fix).
- `test_a_mid_stream_ws_close_cancels_and_releases_the_row` — finding 10, hand-ASGI
  ws transport raising at the post-accept send (`WebSocketDisconnect(1006)`).

## Rule

A blocking generator run through a thread pool cannot be cancelled by async
task cancellation: an in-flight `to_thread.run_sync` is uninterruptible, and a
sync iterator fetched one `next` at a time is never aclosed mid-flight. A
disconnect that must stop work needs its OWN event-loop-side
`is_disconnected()` poll running concurrently with each blocking fetch —
do not assume task cancellation or generator finalization will reach it.

## Device verification and the follow-up fix (2026-09-15)

ops verified the fix on the V100 sm70 endpoint (36 repetitions): every
mid-stream close cancelled the row with zero slot/block leaks. The first SSE
content frame reached the client in 0.001–0.22 s; ~12% of disconnects landed
during a slow late step tick 1–2.94 s long.

That late-disconnect population exposed a SECOND defect, route-independent:
every async handler called the lock-taking `engine.cancel(rid)` synchronously
ON the event loop, not just the SSE watcher. `Engine.cancel` takes
`engine._lock`, which a step tick holds for the whole forward — independently
measured on this box at 1–5.6 s on dense+d1 when free VRAM is near ~350 MiB.
A disconnect arriving in that window parked the event loop thread behind the
tick, and `/health` for EVERY other connection stalled up to a measured 5.57 s.
Every on-loop cancel now runs `await asyncio.to_thread(engine.cancel, rid)`
(the same off-loop placement as `_submit`): both `stream_or_cancel` branches,
`await_or_cancel`'s live-disconnect branch, the chat non-stream
Cancelled/timeout/RuntimeError handlers, the messages and responses async
handlers, and the `/ws/chat` WebSocketDisconnect handler. Cancel takes its own
lock and is thread-safe. The two cancel calls INSIDE the sync SSE generator
(error frame, GeneratorExit teardown) already execute in a `to_thread` worker
and stay plain. GeneratorExit and 499 semantics are unchanged.

Gates now cover both transports under a held lock: the SSE gate and a
parametrized non-stream gate (chat/messages/responses) that disconnect
mid-completion and require `/health` on the same loop under 0.5 s; both were
red when all their cancels reverted to synchronous.

Two separate indicators, and the gate reflects the split:

1. **Event-loop responsiveness**: `/health` on the same loop answers <0.5 s
   while cancel is parked behind a held lock — SSE
   (`test_a_late_sse_disconnect_does_not_freeze_the_event_loop`) and, on the
   non-stream paths, `test_a_nonstream_disconnect_does_not_freeze_the_event_loop`
   across chat/messages/responses. Both were red against synchronous cancel.
   This is fully under the server's control.
2. **Row release**: blocks/state slot return only after the tick that owns the
   lock ends — the engine cannot release faster than its next tick, so the gate
   asserts "cancel recorded + loop stays responsive + allocation zero once the
   parked critical section ends", never a wall-clock release bound.

The long tick itself is a separate, unowned defect — see OPEN.md.

## Device verification after #637 — V100 sm70, e751b27e, 2026-09-15

ops re-ran the real-curl gate `scripts/probe_sse_overload.py` against the
merged fix (e751b27e; warmup green, no liveness restart). The probe holds
inflight at the cap with a self-healing pool and times every disconnect
against a ~20 Hz `/health` sampler. Hard gates: in-flight `/health` < 0.5 s,
and running/slots/blocks end at zero (no wall-clock release bound).

| arm | reps | close→zero release (s) | worst in-flight /health (s) | zero leak |
|---|---:|---|---:|:--:|
| SSE (1 first-frame + 5 at frame ~45) | 6 | 0.052 – 0.574 | **0.273** | yes |
| non-stream in-flight disconnect | 3 | 0.154 – 0.206 | **0.002** | yes |
| `/ws/chat` (manual, third-party `websockets`) | 4 | 0.155 – 1.540 | **0.005** | yes |

Pre-fix the same disconnects left `/health` unresponsive up to **5.57 s**
(measured from an independent curl process); post-fix worst across all arms is
**0.273 s**, and no release exceeded 2 s in this run (the slow-tick OPEN row
stays open — a parked cancel can still wait for a long tick, but the loop no
longer freezes).

**Over-capacity backpressure.** The separate #630/#633/#634 chain gave
`submit` an `max_inflight` ceiling (serve derives `2 * usable_slots` = 8),
refused pre-enqueue as `EngineOverloaded` → HTTP 503 `overloaded_error` on all
three routes, stream included. On an exact `/health` reading of running=4
waiting=4, six ninth submits — chat/messages/responses × stream and non-stream,
fired through one barrier — all returned **503** with integer `inflight=8
cap=8`, no 429 and no `retry_after`; streamed submits got the same pre-submit
JSON 503. The pool then drained to zero and a normal short chat returned 200
before and after. Waiting rows hold no KV, so inflight 8 costs no more device
memory than 4 running rows.

**Follow-up closed.** The non-stream arm logged two `Task exception was never
retrieved` (`RequestFailed: ... cancelled: the reader disconnected`): in
`await_or_cancel` the orphaned completion worker's retrieving callback was
attached only `if not worker.done()` in `finally`, and the
`await asyncio.to_thread(engine.cancel)` window let the worker finish with
`RequestFailed` first, skipping the callback. Fixed by attaching the callback
unconditionally before the worker can finish (a callback on an already-done
task is scheduled immediately and still retrieves). Gate: a scripted engine
whose `cancel()` signals entry then sleeps while `take()` raises inside that
window; a custom loop exception handler plus GC asserts no
`Task exception was never retrieved` — red on the conditional attachment with
exactly that message, green after. Pure CPU, no device run. `stream_or_cancel`
already attached both callbacks unconditionally (#637).
