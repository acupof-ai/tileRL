# The SSD read path: proposed async-with-deadline, landed off-tick reader — 2026-09-09

**Status:** closed. The design doc (`design-ssd-read-path.md`, marked SUPERSEDED
2026-09-09) proposed an async fetch with a deadline; what shipped was an off-tick
reader thread.

## Proposed

The doc (2026-09-07) described the current state as sync fault-in under
`Engine._lock` and proposed:

1. **Issue at `submit`, not `_admit`** — start the fetch when the request arrives,
   not when the engine admits it.
2. **Deadline `n/R`** — drop the fetch if it cannot finish inside the prefill it
   replaces.
3. **Overlap** — other requests admit and prefill normally while a fetch is in
   flight.

## Landed

The read path that shipped runs the `torch.load` **off-tick, in a daemon reader
thread**, not inside `step()` under the lock (`kv_cache.py:549`). `_fault_in`
takes a pre-fetched blob via `self._ssd.take(key)` (`kv_cache.py:1258`), so the
admit path never blocks on disk I/O.

The off-tick reader achieves the doc's overlap goal (other requests are not
stalled) by a simpler mechanism: the fetch is already on a background thread by
the time `_fault_in` asks for it. The deadline and submit-issue parts of the
proposal were not implemented as designed — the reader thread model makes them
less urgent, since the fetch is already asynchronous.

## Rule

A design doc that says "the current state is X, propose Y" must be re-checked
against the tree before it is read as a plan. The "current state" it described
(sync under lock) was already superseded by the time the doc was two days old.
