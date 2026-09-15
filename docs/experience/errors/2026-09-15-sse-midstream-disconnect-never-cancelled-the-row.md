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

That late-disconnect population exposed a SECOND defect in the same watcher:
`stream_or_cancel` called `engine.cancel(rid)` synchronously ON the event loop.
`Engine.cancel` takes `engine._lock`, which a step tick holds for the whole
forward — independently measured on this box at 1–5.6 s on dense+d1 when free
VRAM is near ~350 MiB. A disconnect arriving in that window parked the event
loop thread behind the tick, and `/health` for EVERY other connection stalled
up to a measured 5.57 s. Both watcher branches now run
`await asyncio.to_thread(engine.cancel, rid)` (the same off-loop placement as
`_submit`); cancel takes its own lock and is thread-safe. GeneratorExit and
499 semantics are unchanged.

Two separate indicators, and the gate reflects the split:

1. **Event-loop responsiveness**: `/health` on the same loop answers <0.5 s
   while cancel is parked behind a held lock (`test_a_late_sse_disconnect_does_not_freeze_the_event_loop`,
   red against the synchronous call). This is fully under the server's control.
2. **Row release**: blocks/state slot return only after the tick that owns the
   lock ends — the engine cannot release faster than its next tick, so the gate
   asserts "cancel recorded + loop stays responsive + allocation zero once the
   parked critical section ends", never a wall-clock release bound.

The long tick itself is a separate, unowned defect — see OPEN.md.
