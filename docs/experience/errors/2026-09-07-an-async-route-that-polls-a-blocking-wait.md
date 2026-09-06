# An async route that polls a blocking wait freezes every other route

`Status: fixed` — `/v1/messages` awaits its wait through `asyncio.to_thread`.

## Context

The agent trial's first successful run. Claude Code completed the task end to end —
`calc.py` went from `a - b` to `a + b`, `check.py` prints `ok`, verified on disk — but
while it ran, the server answered nothing else. `/health` timed out, 17 CLOSE-WAIT
sockets piled up, and the freeze lasted the length of every reply: **over 10 minutes**
per turn on this endpoint.

Found by `py-spy` on the live serve child (2833708) rather than by reading: MainThread
blocked in `engine.take` (`engine.py:643`) called from `messages._run`
(`messages.py:215`) inside the `messages` route (`messages.py:287`, `async def`).

## Root cause

`_run` waits for the completion by polling:

```python
while time.monotonic() < deadline:
    out = engine.take(rid)
    if out is not None:
        break
    time.sleep(0.02)
```

`time.sleep` in an `async def` blocks the event loop, so for the whole generation no
other request was served. `take()` also contends for the engine lock that `step()`
holds across each forward, so the polling competes with the work it waits for.

`server.py`'s chat route has always done this correctly:

```python
output_ids = await asyncio.to_thread(_await_completion, request_id)
```

One route over, the same wait ran on the loop. Every blocking part of the request —
encode, `submit`, the poll, the decode — is inside `_run`, and `_run` is called before
the stream/non-stream split, so a single `to_thread` covers both paths. `sse()` only
formats the already-built body: no engine calls, no sleeps.

## Two routes, not one

The brief named `messages.py`. Enumerating every `async def` across the three route
modules found `/v1/responses` with the **identical** defect: `async def responses`
(`responses.py:259`) called `_run` directly, and its own poll loop sleeps at `:186`.
Fixed in the same way, in the same PR — an entry documenting a defect class while a
known instance of it stays in the tree is the half-state the contract forbids.

`/v1/chat/completions` was the only one already correct. So of the three routes that
generate, two were broken, and the one that was right is the one whose pattern the other
two were supposed to copy.

## Why the endpoint made it unmissable

Claude Code's system prompt plus tools is about **30k tokens** on this endpoint
(`prompt_len 30485`, `budget 2282`, `n_out 60` — from the server recorder at
`$ROOT/messages_requests.jsonl`, read by tilerl-27 at 03:18, one reading). So every
turn is a 30k-token prefill on a V100, and `--max-ctx 32768` leaves 2.3k tokens of
room for the reply. A 10-minute turn is a 10-minute freeze.

That is also the same prompt that produced the 400 fixed in #207: 30k of prompt against
a 32768-token context is exactly the shape that landed on the pool's edge.

## Measured

The gate is a second request answering while the first is in flight, which is the
property the defect broke — not a timing of the fix's internals.

| arm | `/health` while a reply generates |
|---|---|
| both routes awaiting through `to_thread` | under 1.0 s, 200 |
| `messages._run(...)` on the loop | **1.86 s**, assertion red |
| `responses._run(...)` on the loop | **1.80 s**, assertion red |

Controls are per route, run separately: removing only `responses`' `to_thread` reds the
`/v1/responses` arm and leaves `/v1/messages` green, and the reverse holds. A single-arm
gate would have passed with one route still broken, which is exactly how this shipped.

The control fails on the elapsed assertion by name, and the in-flight assertion before
it passes — so the red is the freeze and not a server that was never busy. A slow
engine, not a slow model: `take` returning `None` for 70 polls is the same shape as a
long generation at `_run`'s 0.02 s interval, and costs the suite ~1 s.

## Rule

**An `async def` route may not call a function that sleeps.** The blocking wait is not
visible at the route — it is three frames down, inside `_run`, behind a `while` loop —
so the defect is invisible to a reader of the handler and to every test that issues one
request at a time. The whole suite passed throughout: 436 arms, none of them concurrent.

**A sibling route doing it right is not protection.** `server.py:297` had the
`to_thread` from the start; the pattern did not propagate when `/v1/messages` was
written. Fourth defect in this shape on this seam — `limits`, `room_for`, the
hand-rolled clamp, now the missing `to_thread` — each one route over, each found in
production rather than by a gate.

**A single-request test suite cannot see a concurrency defect.** The arm that catches
this class has to hold one request open and issue another; nothing in the suite did
that before, which is why a route that froze the server for minutes shipped green.
