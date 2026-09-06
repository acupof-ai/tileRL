# A prompt that does not fit yet is queued, not refused — V100 sm70, 2026-09-07

> Status: fixed

## Context — the cause measurement, which came before any code

The live 503: a 30,485-token Claude Code prompt took 1,906 of the V100's 2,048 blocks (93.1%),
and the next request got `insufficient KV blocks for request`. Permanently — `submit`
allocated up front and has no later tick to retry on.

The ruling's premise needed one correction, read off the code: **allocation already evicted.**
`engine.py` called `self._prefix.evict_until_free(needed)` on the line *before* the refusal, and
`evict_until_free` loops until satisfied or the store is empty. So the defect was never a
missing eviction call; it was that **eviction ran and freed too little**, which has two causes
needing different fixes. Two arms on CPU, at the production ratio:

**Arm 1 — the store holds only completed requests:**

    needed 60,  free_blocks 34 -> 64 (freed 30),  entries 6 -> 0 (dropped 6),  satisfied

Eviction works when nothing else holds the blocks. Cause 1 — "the pool genuinely cannot hold
it" — is ruled out for this shape.

**Arm 2 — a live request shares the prefix, which is what production had:**

    live: running=1, prefix_hits=1
    needed 60,  free_blocks 44 -> 44 (freed 0),  entries 2 -> 0 (dropped 2),  NOT satisfied

**Dropped 2 entries covering 20 blocks and freed zero.** `free_block` (`kv_cache.py:94`) is a
refcount decrement that returns a block to the free list only at refcount 0; the live request
retained those blocks on its prefix hit, so the store dropping its reference leaves them held.
`evict_until_free` then loops until `_by_id` is empty having freed nothing.

This is **cause 2**, and it settles the fix: "evict harder" is not available, because the
blocks are not reclaimable at all while that request runs. It also explains why
`prefix_evictions` looked healthy in production at 76 and rising — **a drop that frees nothing
still increments that counter.** Measuring `free_blocks` before and after is what separates
them.

Three fixture defects on the way, each of which produced a plausible answer:

**The store was empty and I read it as "nothing to evict".** The first run reported
`dropped 0, freed 0, empty True`. The cause was a poll loop that spun 300 times without
sleeping, never yielding the GIL, so the loop thread never ran: **0 prefill forwards, 0
published.** The probe measured an engine that had done nothing. Same class as #208.

**A token id past the vocab surfaced as an engine error.** Ids ran `1 + i*200 + 160`, which
hits 320 against tiny's vocab of 320 on the third request, and it arrived from `poll()` as
`index 320 is out of bounds` — a failed request, not a bad fixture.

**Arm 1's "no 503" was correct and not the production shape.** Stopping there would have
reported that the defect does not reproduce.

## Fix

**Enqueue before allocating.** `submit` now takes an id and a place in `_waiting`, and nothing
else. `_admit` (in the planner, under `_lock`) takes the state slot and the blocks **together**,
so `_release` has one path and `state_slot is None` is the single test for "never admitted".

The admission condition is `free_blocks` after eviction — a measurement, not a forecast. But
the eviction itself is **guarded** by `reclaimable_blocks()`: without that guard a request
waiting on a live request would call `evict_until_free` on every planner tick, drop every
entry, free nothing, and flush every other client's prefix cache for the whole wait. The guard
asks first whether eviction could possibly cover the gap.

`reclaimable_blocks` is computed from the refcounts, and the obvious test is wrong. A block's
refcount above 1 does **not** mean a live request holds it: the store holds a block once per
entry, and a growing prefix republishes. Measured on a 160-token prompt, entries at 128 and 160
tokens share their first 8 blocks, so those sit at refcount 2 with nothing live — and testing
`refcount == 1` predicted **6 reclaimable where eviction delivered 30**, a 5x undercount that
as an admission test would refuse requests the pool can serve. The test is refcount minus the
number of store entries holding that block.

`blocks_freed` joins the stats, measured in `_drop` from the free list before and after, so a
reader can see that eviction is dropping entries without reclaiming anything. `evictions` is
left alone: redefining it touches 23 call sites and collides with `ssd_evictions`.

**A fourth defect, found in my own change and the one that would have been worst.**
`alloc_slot` raising inside `_admit` propagates to `step()`, whose handler **fails every
running request**. One queued request arriving with the slots full would have killed all the
live ones. `_admit` checks `self._states.free_slots < 1` and returns False before taking
anything — by count, not by catching the raise. `LinearStatePool` gained `free_slots` for it.

Head-of-line FIFO: `break`, not `continue`, when the head does not fit. Admitting a smaller
request past a blocked larger one starves the larger one for as long as the load lasts.

**The prefix match moved with the allocation**, because a hit found at submit can be evicted
before admission. Consequence: a queued request now matches against a store the earlier request
has published into, so **hit counts across this change are not comparable** — more hits, not
fewer.

## Gate

`test_kv.py`, three new tests, and every arm sleeps in its poll loop and asserts
`prefill_forwards > 0` before reading a block number.

1. **two clients at the production ratio** — 64-block pool, 59-block prompts (92.2% against the
   live 93.1%). The second waits and is served; both finish; `blocks_used` and `slots_used`
   return to 0.
2. **a failed admission returns every refcount it took** — `alloc_block` made to raise after the
   hit blocks are retained. Asserts the refcounts and the free-slot count, not "no exception
   escaped".
3. **an impossible prompt still refuses at submit** — the `blocks_for_tokens(total + width - 1)
   > usable_blocks` check survived the move, so a request that can never fit is told
   immediately instead of waiting for its timeout.

Three controls, each run separately, each red on its own assertion:

| mutation | result |
|---|---|
| admission reverted to the unconditional `popleft` | `insufficient KV blocks for request` — the production error verbatim |
| the `reclaimable_blocks` guard removed | blocked admissions evicted 1, 1, 1 |
| the unwind's `free_block` loop removed | every retained block one refcount too high |

**Two versions of one assert were wrong in opposite directions**, which is the instrument
lesson here. The store-untouched check first read `entries >= entries` and **passed with the
guard removed** — the entry count rises as the live request publishes its own chunk boundaries.
Comparing the global `evictions` counter was then **red on correct code**, because decode growth
(`engine.py:915`) evicts legitimately for a running request and `insert` trims at capacity;
neither is the waiting request's doing. It now wraps `_admit` and asserts per attempt:
`[0]*26 + [3]`, where the 3 is the admission that finally succeeded once the first request
finished. Asserting on all 27 attempts would have called that correct eviction a defect.

Five existing tests changed. Three are mechanical — they read engine state right after `submit`
and now need a `step()`, because that state is produced at admission. Two asserted behaviour
that is false by design now:

- `test_a_cancel_returns_the_blocks_a_disconnected_reader_was_holding` asserted a **waiting**
  request's blocks come back on cancel; a waiting request holds none. Inverted to "the count
  must not move", and #209's original behaviour **added back at the admitted level** so that
  PR keeps its test rather than losing it.
- `test_submit_rollback_and_terminal_failure` expected `LinearStatePool exhausted` from the
  second `submit`; that is now a wait. **The arm was also vacuous as first rewritten**: at
  `max_new_tokens=1` each request finished inside its own `step()` and freed the slot before the
  next was considered, so nothing ever contended. Traced rather than assumed, and both requests
  now run long enough to compete.

`448 passed, 14 skipped`. `ruff check` clean.

## Rule

**A counter that increments on an attempt is not a measurement of what the attempt achieved.**
`prefix_evictions` rose 76 times in production while the pool stayed full; the quantity that
decides a 503 is `free_blocks`, and the two only agree when nothing else holds the blocks.

**And an allocation that cannot be retried must not be the one that decides.** `submit` refused
permanently for a condition that was temporary. The ordering was the defect; a retry queue
bolted onto it would have kept the ordering and added a queue.
