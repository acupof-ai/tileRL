# alloc_slot popped before it zeroed, so a raise inside zero_ lost the slot — 2026-09-05

**Date:** 2026-09-05
**Arch:** target-independent (`LinearStatePool` is pure bookkeeping); found while
reviewing a peer's `submit()` unwind fix on sm70
**Task:** review, no GPU window

## Context

A peer asked me to challenge one claim in their `Engine.submit()` unwind fix: that
`LinearStatePool.alloc_slot` is atomic, so moving it relative to the `try` block is
safe. Their reading was "it checks `_free` then pops, which looks atomic."

It is not atomic. `kv_cache.py:244-251` before this fix:

```python
def alloc_slot(self) -> int:
    if not self._free:
        raise RuntimeError(...)
    slot = self._free.pop()          # state already changed
    self.states[slot].zero_()        # can raise
    if self.conv_windows is not None:
        self.conv_windows[slot].zero_()   # can raise
    return slot
```

Two mutating steps sit after the pop. If either raises, the slot has left `_free`
and has not reached the caller, so **nothing can ever `free_slot` it** — the pool is
one slot smaller for the life of the process. `Engine.submit`'s handler cannot
compensate: it guards on `slot is not None`, and on this path `slot` is still `None`
precisely because `alloc_slot` never returned.

An OOM inside `zero_` is not hypothetical on this card, and there is a recorded
instance: `errors/2026-09-04-the-recompiles-reproduce-and-are-not-a-shape-set.md:168-172`
logs `torch.OutOfMemoryError: Tried to allocate 272.00 MiB` with **186.38 MiB free of
31.74 GiB** on this V100. At that margin any allocation raises, `zero_()` included. The
state planes are also a multi-GiB allocation here — `engine.py:1202` records 2.94 GiB at
slots=3 depth=3, 79% of it the per-step verify states — and `alloc_slot` runs on the
request path, after weights, pools and the prefix store have taken their share. Not
claimed: that the state pool is larger than the KV pool; I did not compare them. Not
claimed either: that the recorded OOM was inside `alloc_slot` — it was in `silu_mul`.
What it establishes is that this card reaches margins where `zero_` can fail.

## Measured

Probe drives the pool directly — no engine, no model — replacing `states[slot]` with
an object whose `zero_()` raises:

```
num_slots=3  free=3
one normal alloc -> slot 2, free=2
freed            -> free=3
alloc_slot raised: simulated OOM inside zero_
free before=3  after=2
!! LEAK: slot 2 left _free and reached no caller -- unreachable forever
   pool now offers 2 of 3 slots
```

Slot capacity is the concurrency ceiling (`Engine.__init__` warns when
`usable_slots < max_batch` because a slot is held from submit to finish), so each
leaked slot permanently lowers how many requests the process can serve at once.
`serve` defaults to `--slots 8` (`cli.py:615`), so eight such failures would take
concurrency to zero — and the error that follows is `LinearStatePool exhausted` on a
pool that should have room, which reads as a config problem rather than a leak.

## Fix

Return the slot on any failure of the initialization that follows the pop:

```python
slot = self._free.pop()
try:
    self.states[slot].zero_()
    if self.conv_windows is not None:
        self.conv_windows[slot].zero_()
except Exception:
    self._free.append(slot)
    raise
return slot
```

After the fix the same probe reports `free before=3 after=3 / no leak`.

## The gate

`tests/test_kv.py::test_a_failed_alloc_slot_does_not_consume_the_slot` substitutes a
raising `zero_`, asserts `len(_free) == 3`, and then allocates all three — the last
part matters because a slot could be back in `_free` while the pool is otherwise
inconsistent.

Negative control: `alloc_slot` restored to the pre-fix body → **CAUGHT**; restored →
passes. 282 passed / 14 skipped, ruff clean.

## Not claimed

That this leak has occurred in a live run. It requires a raise inside `zero_`, which
means OOM or a device fault, and no log in the tree shows one. The reason to fix it
without a live sighting is that its symptom (`LinearStatePool exhausted` on a pool
that should have room) points away from the cause.

`alloc_block` in `PagedKvPool` was checked for the same shape and does not have it:
its only post-pop step is `self.refcount[block] = 1`, an assignment into an existing
tensor.

## Rule

An allocator's post-pop initialization is part of the allocation. If it can raise,
the pop needs an unwind — checking the precondition before mutating is not the same
as being atomic. And when a peer says a function "looks atomic", read every statement
after the state change rather than the guard before it.
