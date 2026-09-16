# A non-stream 128k request 504s against the fixed 30-min completion timeout — 2026-09-16

**Status:** open. Real serving limit on the V100, not a wedge or a regression.

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

## Needed / planned

- make the non-stream completion timeout configurable (per-request or a
  long-context server default above the ~17.3 min cold-sparse prefill), or
- direct long-context callers to `stream=true` (no deadline; tokens arrive as
  produced), documented on the non-stream surface.

The first needs a frozen-surfaced parameter and a CPU gate (a stub
`take` that returns past the default but inside an override must not raise
TimeoutError); the second is docs only. Choice is a product call.
