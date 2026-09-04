# A rejected submit released the prefix store's blocks — 2026-09-05

**Date:** 2026-09-05
**Arch:** target-independent (`Engine.submit` and `PagedKvPool` are bookkeeping)
**Task:** sweep for defects the test suite does not reach

## Context

Two defects, found by reading `Engine.submit` and `DataParallelEngine` against
their call sites rather than by a failure.

### 1. `submit`'s unwind decremented refcounts the PrefixStore still holds

`submit` seeded its rollback list with the prefix-cache hits before it had
retained them, and allocated the state slot inside the same guarded region:

```python
blocks = list(hit_blocks)          # not retained yet
slot = self._states.alloc_slot()   # can raise
...
except Exception:
    for b in blocks:
        self._kv.free_block(b)     # decrements what the store owns
```

`alloc_slot` raises whenever the state pool is full — a normal condition at
`max_batch` concurrency, not an exotic one. When it did, the handler called
`free_block` once per hit block. Those blocks are owned by the `PrefixStore`,
which had a refcount on each and never released it. `free_block` decrements
without recording who is decrementing, so the spurious decrement is
indistinguishable from a real release: the store keeps serving those blocks as
a cache hit while the pool believes they are free, and hands the same page to
the next request.

The fix reorders rather than adds state. The slot is taken first and `blocks`
starts empty, so nothing is on the rollback list until this request has
incremented it; the hit blocks are retained one at a time and appended after
each `retain` succeeds.

Releasing all of `blocks` on the way out is only safe because `retain` cannot
fail inside that loop — it raises at refcount 0, and `hit_blocks` came from
`_match_prefix` under the same lock, so every entry is non-zero. Everything
after the loop *can* raise, and by then every entry is retained. That
invariant is documented in the code, because the alternative is tracking how
far the loop got, which is state for a case that cannot occur.

### 2. `--devices` could not stream

`DataParallelEngine` forwards 10 methods and defines no `__getattr__`. It had
no `peek`, and `server.py:265` calls `engine.peek(request_id)` unconditionally
as the first line of `_stream`'s loop. `_stream`'s handler catches
`(TimeoutError, RuntimeError)`, so the `AttributeError` escaped the generator —
after the SSE 200 header had already gone to the client.

The observable result was worse than an error: **HTTP 200 with an empty body,
0 SSE frames, no `[DONE]`.** A chat client reads that as an empty reply, not a
failure. Every streaming request through `serve --devices` was broken, and the
chat page always sends `stream=true`.

`peek` forwards on the same rid encoding as `take`:
`i = request_id % self._n`, then `peek(request_id // self._n)`.

## Measured

Both defects were reproduced before fixing.

The submit unwind took three attempts to reproduce. The first two tests passed
against unfixed code, because `_match_prefix` treats `matched >= len(tokens)`
as a miss — resubmitting an identical prompt never hits the cache. The gate
needs a **longer** prompt that shares the stored prefix, and asserts
`engine.stats()["prefix_hits"] >= 1` so it cannot silently go inert.

The `peek` gap: `DataParallelEngine` constructed directly on two CPU engines,
`stream=False` returns 200 (submit/take are shared and work), `stream=True`
returns 200 with `r.text == ''`. After the fix, frames arrive and the stream
terminates. Negative control with `peek` deleted: **CAUGHT**.

Exhaustive check that `peek` was the only gap — a probe comparing the real
attribute surfaces rather than a hand-written list: regex over every
`engine.<name>(` call in `server.py`, `messages.py`, `cli.py`, `rollout.py`,
`train.py` (10 names) against `dir(DataParallelEngine)` (11 names).
`MISSING: ['peek']`, `has __getattr__: False`.

## Fix

- `engine.py` — slot first, `blocks` empty, retain-then-append, with the
  `retain`-cannot-fail invariant stated in a comment.
- `parallel.py` — `peek` forwarding on the `take` rid encoding.
- `tests/test_kv.py::test_a_rejected_submit_does_not_release_the_prefix_stores_blocks`
- `tests/test_server.py::test_a_data_parallel_engine_can_stream`

## Rule

**A rollback list may only hold what this caller has already incremented.**
Seeding it with borrowed resources makes the handler release someone else's
reference, and a refcount decrement carries no owner, so the corruption is
silent.

**A forwarding layer with no `__getattr__` needs its surface compared against
its call sites, not grepped.** Grep confirms the name you already suspect; the
set difference answers the exhaustive question. No interface assertion was
added — one gap in one class is an instance, not a pattern, and a maintained
method list drifts the same way the class does.
