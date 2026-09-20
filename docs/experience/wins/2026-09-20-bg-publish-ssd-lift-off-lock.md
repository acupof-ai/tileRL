# Background SSD lift moves disk IO off the tier lock — 2026-09-20

> Status: landed behind the existing `TILERL_CLOSE_BG_PUBLISH=1` gate (still
> default OFF). CPU gates green; device benefit pending-remote and, per
> perf1/b1, must be reported honestly if the #732 event-query arm shows most of
> the 3.3 s was cross-thread drain accounting rather than real step blocking.
> Follow-up to [2026-09-20-background-close-publish.md](2026-09-20-background-close-publish.md).

## Context

After the background publisher, a static read showed the worker's private-SSD
lift — `HostKvPages.share_hold_kv` lifting a cold page off the private spill
file and writing it into the shared `.prefix.bin` — ran with the **whole two
disk passes inside the tier `_tlock`**: the private `ColdSsdFile.read` and the
shared `_write_shared_ssd` (slot alloc, possible extent ftruncate+remap, byte
copy). On the V100 bg2 arm, with the queue holding every job, the close tick
still measured `ssd_mmap` ~3.3 s and `fwd_host` 7.4 s. The device event-query
split (real lock wait vs drain-attributed disk time) is not yet captured, but
"multi-second disk IO inside a lock a forward tick also takes" is a defect on
its own and moving it out is strictly correct regardless of that split.

## What Worked

A three-phase lift, and a separate lock per spill file:

- `ColdSsdFile` grew its own `_mlock` guarding slot map / free list / extent
  live counts / mmap rotation. Lock order everywhere is tier `_tlock` -> file
  `_mlock`, one direction; the worker never inverts it.
- **Phase 1 (short):** under `_tlock` resolve RAM-vs-SSD and pop the private
  key; for an SSD source borrow a read handle under the *private file's* lock
  (`borrow_read`), and reserve a shared slot under the *shared file's* lock
  (`reserve_slot`) — both milliseconds, no byte copy.
- **Phase 2 (no lock):** `read_borrowed` pulls the private bytes and
  `write_reserved` copies them into the reserved shared slot with NEITHER lock
  held. The slot is in a `_reserved` set — invisible to `_slot_of` and skipped
  by allocation — so no reader can see a half-written blob (the #743 future
  still gates follower adoption). The mapping being written is borrow-counted;
  a concurrent grow remaps to a new mmap and keeps the borrowed generation
  alive (`_prune_maps_locked`) instead of munmapping bytes mid-copy, and a
  trailing-extent shrink treats a reserved slot's extent as occupied.
- **Phase 3 (short):** under `_tlock` consume the private spill slot, install
  the shared record (spilled, or RAM on over-cap/disabled spill), and fold the
  pre-commit ref delta.
- **Source pin across the whole IO.** `_pub_private` pins the in-RAM namespace,
  but the private SPILL slot needed its own pin: the key is added to the spill
  file's `_pinned_keys` at `borrow_read`, the mapping borrow ends once the bytes
  are copied, and the KEY stays pinned until phase 3 `consume_pinned`
  (`rollback_pinned` on failure). The file's `forget` skips a pinned key, so —
  even if the tier marker were dropped in phase 1a (the original race) — the
  step thread cannot return the source slot to the free list (a later write
  reusing it would overwrite the bytes being copied) or reclaim-truncate its
  extent while the worker is in the lock-outside read.
- **Rollback:** a write failure after reservation calls `release_slot` (back
  to the free list, borrow dropped, no extent increment) and disables shared
  spill for the process; the blob is then kept in RAM exactly like the inline
  path, and the source key is unpinned. No reserved slot or extent count can
  leak.

The worker stays CUDA-free (the prior diagnosis stands): private reads are
`pin=False`, the shared write's `.cpu()` guard never fires on CPU tensors.

## Gates (CPU, hermetic)

- **IO off-lock:** inject a slow private `read_borrowed`; while the worker is
  parked inside it a peer `cold.stats()` (takes `_tlock`) returns immediately
  and the reserved shared slot is unreadable. Mutation-red on moving the read
  back under `_tlock`.
- **Rollback:** force the lock-outside shared `write_reserved` to raise; the
  key still serves from RAM, shared spill is disabled, `_borrowed` returns to
  zero, no slot stays reserved, and only the earlier good slot is live.
  Mutation-red on deleting the `release_slot` call.
- **Source-slot pin:** with the worker parked in the lock-outside read, drop
  the tier marker (reproducing the phase-1a race) and have the step thread
  `cold.forget` the source: the slot stays mapped, off the free list, with its
  extent live, and is handed to no other write; after release it is consumed
  exactly once. Mutation-red on `forget` ignoring `_pinned_keys`, and on
  `borrow_read` not pinning the key.
- Byte equality of a lifted page and every prior background-publish gate
  (RAM/hold/dup/failure/follower-hit) still pass.
- **Borrow balance / dup cleanup:** after a normal lift, a write-failure lift
  (RAM fallback), and a dup-content lift, both spill files show `_borrowed == 0`
  and no pinned keys; two same-content private publishers (one queued, one
  inline, so the worker takes the never-borrowed dup path) both have private
  slots recycled, extent live back to 0, and `_ssd_bytes` decremented exactly
  once. Mutation-red on not releasing the private mapping borrow, and on routing
  the dup consume through the pinned-only `consume_pinned`.
- **Hardening (structural rework, same PR):**
  - The destination reserve is a one-shot `_SlotToken` settled in a finally on
    EVERY exception, not just OSError: a non-OSError (RuntimeError/copy) in the
    write, or an arbitrary exception from bucket open/reserve after the read
    succeeded, releases the token and consumes the already-owned private source
    exactly once (free, extent down, bytes decremented once) → prompt miss.
    Mutation-red on catching only OSError around the write.
  - Phase 3 re-checks the content key under `_tlock`: an inline publisher that
    committed the same key during the lock-free IO wins; the worker releases its
    own destination token and folds a ref instead of overwriting the record,
    double-charging bytes, or orphaning the inline slot.
  - Pending ref deltas still fold on a failed/missed job whose key an inline
    commit won (refs = inline 1 + folded delta).
  - A failed trailing-extent reclaim on the private consume reports back
    (`consume_pinned -> (freed, reclaim_ok)`); the job then releases its
    destination token and resolves as a miss rather than committing against
    unsettled accounting.
  - `take()` refuses (`None`) a private key an in-flight job owns
    (`_pub_private`, or the spill file's per-key pin): taking it would race the
    worker's own consume and double-decrement `_ssd_bytes` / recycle a pinned
    slot. Deterministic gate verifies no decrement while queued and exactly one
    after commit.
  - Shared spill is bucketed by the blob's field-set signature
    (`{k,v,bounds}` cold; `{k,v,bounds,dk,dv}` warm), dtype NOT part of it.
    Buckets are sibling files with their own frozen stride; `_shared_ssd_bytes`
    is the SUM across buckets under one global cap; an unrecognized field set
    fails loud into RAM (never opens its own file, ≤4 sibling files). A cold and
    a warm page round-trip in both creation orders and a non-owned field reads
    as a clean miss, never KeyError.
- Full CPU suite **1082 passed / 14 skipped / 1 xfailed** (merged over latest main incl. #749/#756).

## Rule

Slow IO inside a metadata lock serializes the fast path against the disk for
no reason: reserve/account under the lock, do the bytes with neither lock held
against a hidden, borrow-protected resource, then publish under the lock — and
on failure release the reservation, never leak it.

## Device target (pending-remote)

Next V100 window must measure three things together (#745 depth, this lock
split, and a non-blocking event query before/after the close segment) to split
the ~3.3 s `ssd_mmap` into (a) real step time waiting on `_tlock`, which this
PR removes, and (b) worker disk time merely drained into the tick counter,
which does not stall the step. If (b) dominates, the close-latency win is
smaller than the lock removal suggests and the entry/wins number will say so.
Steady decode (~166 ms/tick, ~9.4 tok/s) must hold.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-20 | CPU (hermetic) | SSD lift disk IO off `_tlock` | IO-off-lock + source-pin + rollback + hardening gates green (all mutation-red); 1082 passed |
| next V100 window | V100 sm70, pending-remote | close ssd_mmap ~3.3 s | target: real lock-wait removed; lock-vs-drain split to be measured |

Raw artifacts: `tests/test_sparse_kv_tier.py`; changes `src/tilerl/kv_tiers.py`.
