# Pool exhaustion failed the whole batch, not the one row — 2026-09-10

## Context

When the paged KV pool ran out, `alloc_block` raised `RuntimeError("PagedKvPool
exhausted")` inside the decode loop of `_run_forward`. `step()`'s handler caught
it and called `_finish(error=...)` on **every** running request, then re-raised.
On the training side `poll()` raises on any failed request, and `_drain` did not
catch, so one exhausted pool crashed the whole training step. The failure
surface was the whole batch; the `live` mask in `group_advantages` was never
reached.

## Root Cause

The decode allocation loop was the unguarded half of the engine. `_admit`
already checks the pool **by count** and returns False when a waiting request
does not fit — its comment says why: an exception out of that path reaches
`step()`'s handler, which fails EVERY running request. The decode loop had no
such check: it let `alloc_block` raise, and the handler did exactly what
`_admit`'s comment warns about.

## Fix

The decode loop now checks by count, mirroring `_admit`: a row whose chain
needs more blocks than the pool has free fails alone via
`_finish(reason="pool_exhausted")` and leaves the batch; the other rows keep
decoding, and the dead row's blocks return to the pool.

Failures are now structured: `RequestFailed(request_id, reason, message)`
carries a stable reason tag through `poll()`/`take()` instead of a bare
RuntimeError. `_drain` catches **only** `reason="pool_exhaustion"`: the dead
rollout gets an empty completion, `len(c) > 0` drops it from the live mask, and
the group trains on the rest. Every other failure class propagates — a broad
catch would turn any bug into a silently missing row.

Which row dies: **the row allocating when exhaustion hit**, in decode order —
NOT the row holding the most blocks. The pool is shared, so which request trips
it is a scheduling accident; the semantics are arbitrary by construction.

## Rule

- A failure that one row caused must land on that row. The batch-wide handler
  is for bugs in the tick itself, not for resource exhaustion a row can survive.
- A catch around a failure channel is narrow per failure class, with the class
  carried as data. Message-text matching dies on the next wording change; a
  broad catch dies by turning every failure into silence.
