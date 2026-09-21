# A disconnected non-stream client kept its sparse row generating and publishing — 2026-09-14

## Context

The stream and `/ws/chat` paths cancel an abandoned request: SSE gets
`GeneratorExit` and the websocket branch gets `WebSocketDisconnect`, and both
call `engine.cancel`. The three non-stream routes did not. `/v1/chat/completions`,
`/v1/messages` and `/v1/responses` each handed a blocking `take()` poll loop to
`asyncio.to_thread` and awaited it; when the client left, starlette cancelled
the awaiting task, but nothing propagated that to the worker thread or the
engine. The row kept decoding to `max_new_tokens`, holding its slot and blocks.

A second defect sat behind the first: `Engine.cancel` released the row but left
`req.failed = False`. In `_release` a sparse, non-failed row runs
`close_request` and transfers its pages into shared cold blobs — the prompt-end
frontier closure intended for a reader that finishes. A cancelled reader got
its prefix published and, on a spill-capable build, could trigger an SSD write
for nobody. The #587 cold-spill failure rows set `failed` specifically to skip
that closure; cancel did not.

> **Note 2026-09-21:** the routing bug this entry fixed is unaffected, but the *motive* for the second half no longer exists: the refactor deleted the forced prompt-end closure entirely (Epic #779 M3, #787/#789), so a cancelled row can no longer publish a prefix for a reader that does not exist. Setting `failed` remains correct; it is no longer load-bearing for this reason. See [errors/2026-09-21-optimizing-at-the-wrong-layer-close-scheduling.md](2026-09-21-optimizing-at-the-wrong-layer-close-scheduling.md).

`/health` understated the damage: `stats()` serves a snapshot refreshed only
after a `step()`. A cancel leaves zero rows, so the loop idles and never
refreshes, and `/health` kept reporting the dead row and occupied slot.

## Root cause

- Route layer: no `except asyncio.CancelledError` around the non-stream awaits.
- Engine: cancel did not classify the row as failed, so the sparse release path
  could not distinguish "reader gone" from "finished, publish the prefix".
- Stats: the cancel path did not refresh the lock-protected snapshot.

## Fix

- All three non-stream routes catch `asyncio.CancelledError`, call
  `engine.cancel(rid)`, and re-raise. `messages`/`responses` submit inside the
  worker thread, so a one-element rid box carries the id back to the route.
- `cancel` sets `req.failed = True` before `_release`: a cancelled sparse row
  now releases hot pages, host cold blobs and the slot exactly like a #587
  failure row, publishing nothing.
- `cancel` refreshes `_stats_snapshot` under the lock when the loop thread owns
  the engine, so `/health` reflects the freed slot without waiting for a tick
  that never comes.

Two CPU gates, each with a red control:

- `test_a_nonstream_client_disconnect_cancels_its_request`: an httpx ASGI
  request to each of the three routes is cancelled after the row is running;
  the row leaves and the slot serves a follow-up. Red on main (row stays
  running), green after.
- `test_a_cancelled_sparse_row_publishes_no_prefix_and_returns_every_page`:
  cancelling a 24-page sparse row publishes zero prefix entries and returns
  every hot/cold page and the slot. Red without the `failed` flag (the
  prompt-end closure published), green with it.

## Rule

A disconnect is a terminal row state, not a transport detail: the route must
map it onto the same release path a failed row uses. A cancellation that only
stops the response bytes while the engine keeps generating is a leak wearing a
cancel's name. Every state-mutating event on a row also invalidates any cached
view of that row.
