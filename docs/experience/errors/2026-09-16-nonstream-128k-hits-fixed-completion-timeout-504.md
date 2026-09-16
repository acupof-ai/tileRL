# A non-stream 128k request 504s against the fixed 30-min completion timeout — 2026-09-16

**Status:** open — configurable-cap fix landed (see Fix); the default stays 1800 s, so an
unconfigured server still 504s a >30-min cold fill. Closes when the long-context
serving config raises/zeros the cap (or callers use `stream=true`), confirmed on device.

## Context

The #658 device gate (0cc82a36 env-off, boot 0) included a ~128k-token cold
sparse request. The **streaming** arm completed normally: HTTP 200,
`finish=stop`, 1037 s (~118 prefill tok/s) with a 333 MiB SSD spill. The
**non-stream** arm hit the fixed `_COMPLETION_TIMEOUT_S = 1800.0` (30 min;
defined in `src/tilerl/messages.py`, imported by `responses.py` and
`server.py`) awaited through `src/tilerl/prompt.py::await_completion`
(`server.await_or_cancel`), and returned a 504, even though the cold sparse prefill legitimately takes
~17.3 min on sm70 — under the limit today, close enough that ordinary variance
or a colder/sparser fill crosses it, and a longer context does so by design.

## Root cause

`_COMPLETION_TIMEOUT_S` is a single module constant shared by every non-stream
route (chat, messages, responses). It bounds how long `engine.take` may block,
intended to fail a row that never finishes; it also caps legitimate
long-context cold prefill, which the hybrid path runs deliberately. Streaming
has no such client-side await (frames keep the connection live), so only
non-stream callers see it.

This is unrelated to the GIL-spin wedge (#658): the loop is healthy and
scheduling; the request is progressing; only the fixed await deadline fires.

## Fix (landed): configurable non-stream completion timeout

The non-stream cap is now configurable while the default is unchanged:

- `--completion-timeout-s SECONDS` on `tilerl serve`, or the
  `TILERL_COMPLETION_TIMEOUT_S` environment variable (CLI default reads it).
- Default **1800** — identical behaviour for an existing server.
- **0 = no deadline** (long-context server mode): `prompt.await_completion`
  accepts `timeout_s <= 0` as "wait until `take` returns"; the ASGI disconnect
  watcher in `server.await_or_cancel` still drops a hung client, so 0 does not
  wait forever.
- Applies to the three non-stream routes only (chat via `_await_completion`,
  `/v1/messages`, `/v1/responses`, all resolved once in `create_app`).
  Streamed SSE and `/ws/chat` keep the fixed `_COMPLETION_TIMEOUT_S` frame
  guard — they never had the bug (frames hold the connection open).

A long-context server sets a value above its cold-fill time (e.g. 3600) or 0;
callers that can stream should keep using `stream=true`, which has no await
deadline at all.

CPU gate (`tests/test_server.py`): the resolver's three states (default 1800,
env override, 0) and a `take` stub that is unfinished past a short positive
deadline (504 + row cancelled) but completes under `timeout_s=0` (200) — the
override-only shape the plan called for. CLI frozen-surface set updated.

## Remaining to close

- flip the deployed long-context V100 serve to the raised/zero cap and confirm
  the non-stream 128k fill returns 200 on device; or
- leave the server default and direct long-context callers to `stream=true`
  (no deadline), documented on the non-stream surface.
