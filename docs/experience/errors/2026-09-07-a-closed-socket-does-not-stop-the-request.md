# A closed socket does not stop the request — V100 sm70, 2026-09-07

> Status: Fixed and verified live on the merge sha. 1891 tokens and 104 KV blocks
> were generated for a reader who had already left; after the fix, 6 tokens and
> 0 blocks.

## Context

#200 shipped a stop button on the chat page. `web/src/transport.ts` said stopping
means "the tokens on screen are the reply, not a failure" — true for the page,
and the sentence quietly implied the server had also stopped. Nobody had checked.

## Root Cause

`server.py`'s `WebSocketDisconnect` branch called `gen.close()`. That ends
`_deltas`' poll loop and nothing else: `_deltas` is a plain generator over
`engine.peek()`, and closing it never reaches the engine. **There was no cancel
path on the engine at all** — the only removal is `_finish` (`engine.py:1250`),
reached from normal completion or an error inside `step()`. The SSE route did not
even close the generator; a hung-up `curl` left the request running the same way.

So an abandoned request ran to its cap, which since #195 is the context
remainder.

Measured on the live V100 (`aacc936`), idle endpoint, one request, socket closed
at **t=2.81 s** after 12 delta frames — the client had 23 characters, "1\n2\n…\n11":

```
    t  run wait    tok  fin  blk
  0.0    0    0   4236    8    0    baseline
  2.0    0    0   4236    8    0
 4.01    1    0   4323    8    8    <- socket already CLOSED at 2.81
12.24    1    0   4807    8   38
25.38    1    0   5450    8   78
33.06    1    0   5860    8  104    <- still running, blocks still climbing
37.38    0    0   6127    9    0    <- ended on its own
```

**1891 tokens** generated after the close, **104 KV blocks** held for ~34 s,
`running: 1` throughout. It stopped by finishing its answer, not by being
dropped.

## Fix

`Engine.cancel(request_id)` under `_lock`: release the request's blocks and state
slot, file it under `_failed`, and remove it from whichever queue holds it.
Called from the `WebSocketDisconnect` branch and from the SSE generator's
`GeneratorExit`.

**`_finish` could not be reused as-is**, and this is the part that would have
shipped broken. It ends with `self._running.remove(req)` and **raises
`ValueError: list.remove(x): x not in list`** on a request still in `_waiting`.
A waiting request is not free to drop either: `submit` allocates its blocks and
state slot up front (`engine.py:545`), not at admission, so both queues have to
give them back. The teardown is now `_release`, shared by both paths, and
`cancel` removes from the queue that holds the id.

The case that would have crashed is a browser tab closed while requests are
queued — not an exotic path.

A cancelled request goes to `_failed` so a late `take()` raises instead of
returning None, which is what "not finished yet" already means. `_failed` grows
one entry per abandoned request; a TTL sweep is the named upgrade, not fixed
here.

`DataParallelEngine` gets `cancel` too. Not a choice:
`test_every_engine_the_routes_accept_implements_what_they_call` derives its
required set from the route modules' own source, so adding `engine.cancel(...)`
to `server.py` extends the requirement automatically.

## Gates

Four controls, each run separately:

| control | fails with |
|---|---|
| `cancel` calls `_finish` verbatim | `ValueError: list.remove(x): x not in list` (waiting arm) |
| the WS branch stops cancelling | `expected … to cancel; found 1` |
| the SSE `GeneratorExit` removed | `the SSE route needs GeneratorExit` |
| `cancel` frees nothing | `blocks not returned: 3 of 3` |

Both queues are exercised, because they fail differently. The waiting arm builds
an engine with `max_batch=1` — `StepLimits` is frozen, so that is the only way to
force a second request to sit in `_waiting` — and asserts the second request is
actually there before cancelling it, or the arm is vacuous.

A gate on `Engine.cancel` alone would pass with both call sites missing, which is
exactly the state this shipped in, so a second test asserts the routes call it.

**440 passed, 14 skipped** on cpu, rebased onto #208 (438 before it; #208 brought two arms).

## Verified on the live V100 after the merge

Same endpoint, same shape, on the merge sha `09e1e84` — child 2855749, confirmed
running that code rather than a stale import (both `.pyc` files written *after*
process start, and `engine.py`/`server.py` hash-match the `09e1e84` blobs).

```
     t  run wait     tok  fin  blk
  6.45    1    0     200    1    9
  6.79    1    0     220    1   10   <- socket CLOSED at t=7.00
  7.16    0    0     226    2    0
```

Running and blocks back **0.16 s** after the close, against 104 blocks held for
~34 s before the fix. Tokens 220 → 226: six, against ~47 tok/s over the ten rows
before the close, so under one poll's worth — the cancel landed inside the step
already in flight.

**The first repeat could not have shown the token half.** It sampled every 2 s,
so the close at t=7.00 fell between the t=6.01 and t=8.10 rows and the 55-token
gap across it was unattributable — every one of those tokens could have been
generated before the close. The block result survived that poll and the token
result did not, and the difference is what each one is: a **level** stays put and
tolerates a slow sampler, a **delta across an instant** does not. Re-run at
0.25 s.

**`blocks_used` cannot prove this on its own.** It is the engine's own counter,
decremented by `_release` at `engine.py:1255` — the very function under test — so
a `free_block` that never landed would still print 0. `pool_used_blocks` is the
allocator's `num_blocks - len(self._free)` and is independent of it.

It initially looked like a contradiction: across two runs `pool_used_blocks` went
1 → 7 → 17 while `blocks_used` returned to 0 each time, the +10 on the second run
exactly the peak that run had held. It is the prefix cache, and the control says
so rather than the inference — one **normally completing** request moves the pool
17 → 19 with `prefix_published` 7 → 9 and `blocks_used` 0. A completion retains
the same way a cancel does, `free_block` only reaches the free list at refcount 0,
and `_blocks_used` is documented "retains excluded" at `engine.py:447`. The two
counters disagree by design, and the disagreement is not about cancel.

## Rule

"The client stopped" and "the server stopped" are two facts. A comment asserting
the second because the first is visible is a claim about code nobody read — and
here the server half was never implemented.

## The measurement nearly failed twice

The first attempt polled `/health` inline after the close. v100's agent trial was
holding the event loop at the time (a synchronous `engine.take` on an `async def`
route starved every other route), the poll timed out, and the run produced
nothing attributable — the leak and a starved endpoint look identical from a
client that only samples afterwards. The second attempt sampled from a thread
started **before** the socket opened, so the baseline and the close are one
series and a failed poll lands as an `err` row rather than a gap.

And a `curl -o /tmp/h.json` poll read back a healthy body with `slots_total: 4`
against this server's 8 — another session's write to a shared path. Every one of
those requests had returned 0 bytes. A stale file at a shared path read as a
successful poll.
