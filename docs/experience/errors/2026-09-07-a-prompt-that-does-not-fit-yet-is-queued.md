# A prompt that does not fit yet is queued, not refused — V100 sm70, 2026-09-07

> Status: fixed — **verified live on the V100 (`3401476`, child 2917009) at the production
> config, `--max-batch 1`, cold store.** A at 30,000 chars took 1,877 of 2,048 blocks leaving
> 171 free; B was sized from that live reading to 211 blocks (3,312 tokens), so it could not
> fit then and fits the pool. Both returned **200** — A in 532.81 s, B in 551.44 s — and
> `waiting` peaked at **1**, which is the proof B queued rather than being refused: a 200 alone
> could mean B simply arrived after A finished. **B waited 518.6 s in `_waiting`** (queued at
> t+14.4 s, admitted at t+533.0 s, within one 2 s sample of A releasing). Before this change
> the same arm returned `insufficient KV blocks for request`.
>
> **The cause-2 signature appeared, and its mechanism is NOT `_admit`.** During B's wait
> `prefix_evictions` went 0 → 47 while `pool_used_blocks` held at 1,877 and never fell —
> 47 evictions, zero blocks freed. But those are `insert`'s state-byte trim
> (`kv_cache.py:910`), not `evict_until_free`: A republishes its prefix every chunk, and
> `prefix_state_bytes` sat at 1,725,825,024 of a 1,845,067,776 budget — **93.5% full** — with
> `prefix_published` tracking `prefill_forwards` 1:1. So the trim ran continuously and dropped
> roughly one entry per chunk, freeing nothing because A retained the blocks. **The two
> readings invert the conclusion and look identical on the counter**; they were told apart by
> the publish-per-chunk 1:1 ratio and the 93.5% state fill, not by the eviction count.
>
> **`_admit` was never called for B in this arm** (`engine.py:693`, the batch cap), so nothing
> here tests the `reclaimable_blocks` guard; see the five-arm section below, which also corrects
> the claim this line first made.
>
> **First measured queued wait.** `messages.py:72` says of the 1800 s cap: "this is the ceiling
> for a request the scheduler may hold behind a full batch, not the cost of one; nothing has
> measured that." 518.6 s, 3.5x inside the cap. Two such clients queued would be ~1,037 s,
> still inside; three would not be.

## One flag moves three quantities, which cost an arm

The second arm was to be `--max-batch 2`, the only configuration that reaches the
`reclaimable_blocks` guard with two slots free. It could not: **`--max-batch` also resizes the
KV pool.** `cli.py:133` derives `max_blocks = (ctx * max_batch) // BLOCK_TOKENS` and
`_fit_blocks` returns `min(fit, cap)`, so at `--max-batch 1` the **cap** binds at exactly 2,048
(which is what `serve_v100.sh`'s comment means by "32768 keeps CAP the binding one"), and at 2
the cap rises to 4,096 and the pool becomes fit-bound instead. Measured on that child:
`blocks_total` **2,048 → 3,494** (+70%) and `prefix_state_bytes_budget` **1,845 MB → 1,087 MB**
(−41%). B then fit immediately, `waiting` never left 0, and the guard was not reached — so the
arm shows concurrent admission and nothing else, and its eviction counts are not comparable to
arm 1's 47 either, since a 41% smaller budget trims earlier per chunk. Same class as one knob
crossing two thresholds: **before reading a one-flag arm as one variable, ask what else the flag
derives.** The one-variable arm pins the pool: `--max-batch 2 --blocks 2048`.

## Five arms, and what each one could not say

Both of this change's claims are measured: the guard **declines** when eviction cannot help, and
**permits** when it can. It took five arms because the first four each failed to reach the guard
for a different reason, and each failure looked like a result.

| arm | config | `_admit` called for B | B fits? | guard | the observable |
|---|---|---|---|---|---|
| 1 | mb 1, pool 2048, cold | **never** — batch cap | n/a | untested | B waited **518.6 s**, both 200 |
| 2 | mb 2, pool **3494** | yes | yes, bigger pool | reached, declined | no wait; A **1.31x** slower |
| 2b | mb 2, pool 2048 | yes | yes, **prefix hit** | reached, declined | 211 needed, 144 allocated, **67 adopted** |
| 2c | mb 2, pool 2048, distinct filler | yes | **no** | **negative path** | `pool_used` pinned **1877**, B waited **529.3 s** |
| 3 | 2c + **72 warm blocks** | yes | **after eviction** | **positive path** | never waited; `pool_used` **2037** where 2,088 was demanded |

**Arm 1 never reached `_admit`.** `engine.py:693` is
`while self._waiting and len(self._running) < self.limits.max_batch`, so at `--max-batch 1` the
loop body does not run while A holds the only slot. B waited on the **batch cap**, not on a block
decision. This entry first claimed the guard "declined on every tick for 518 s"; that was wrong,
and the queue-and-wait result — which is what the fix claims — does not depend on it.

**Arm 2b's fixture defect: both clients shared one filler.** `"w " * n` for A and B makes B's
prompt a literal prefix of A's, so B adopted A's published blocks and `needed` fell from 211 to
144, which fits in 171 free — `_admit` returned before `evict_until_free`. A client sized from
live state can still be wrong in its **content**. Arm 2c gives B its own filler and a distinct
leading token, so no block hash can match: the shared chat header `<|im_start|>user\n` is ~3
tokens against `BLOCK_TOKENS` 16, and `"aa "` vs `"bb "` diverges inside block 0. A hit is
impossible by construction rather than merely unobserved, and `prefix_hits == 0` is now asserted
as a postcondition.

**Warming does not widen the guard's window; it moves `free` down.** The positive path needs
`free < needed <= free + reclaimable`, and **`free + reclaimable` is `pool − A` = 171 whatever is
warmed**, because a warm block comes out of free 1:1. So the lever is B's size relative to free,
not the amount warmed: at `warm=0`, `free=171 < B=211 <= 171` is false (that is arm 2c); at
`warm=72`, `free=99 < B=139 <= 171` holds. Predicted before the run and matched to the block.

**The eviction COUNT is not the discriminator, in either direction.** The signature agreed before
arm 3 was "one eviction burst above the constant published−evictions gap". Arm 3 shows no burst:
the gap climbs 3→11 over the first 34 s and then holds at **11** for the remaining 540 s, and arm
3's *total* evictions (51) are **fewer** than arm 2c's (58) — the arm that freed nothing evicted
more. Eviction runs continuously in both arms because every finished decode publishes and the
store is at capacity; whether it *frees* depends on whether the blocks are retained elsewhere,
which no count on the wire distinguishes. Had the pre-agreed signature been the criterion, arm 3
would have been scored a failure. The evidence is `pool_used`, and as a **level**, not a delta: B
needs 139 blocks on top of A's 1,949, `1949 + 139 = 2088` exceeds the 2,048-block pool, so the
observed 2,037 is only reachable if eviction returned at least **51 blocks** — with `prefix_hits`
0, none of it adoption. The pool ceiling settles it without sampling. A delta between two polls
attributes to whatever happened between them: the "2037 → 2013 fall" first written here is a
24-block move at t+37.6 s, with `running` already 2 at t+5.4 s — it happened *after* both
requests were admitted and was not B's admission at all.

**Arm 3's positive path is proved by the absence of a wait, and that is why the level matters.**
`waiting` never left 0 in 324 samples: B was admitted on its **first** `_admit` call, so there is
no wait window to point at and no eviction burst to time. The only observable is that B got 139
blocks the free list did not have. Arm 2c is the control that makes this readable — same pool,
same sizing rule, `waiting` pinned at 1 for **529.3 s** and `pool_used` pinned at 1877 across the
whole window. One arm waits and frees nothing; the other frees and never waits.

**And two measurement notes, both instrument defects in this window's own tooling.** The gap
series above is arm 3's, because arm 2c's rows carry **no `published` at all** — 0 of 274 —
`published` was added to the sampler after the pod already had its copy, so an earlier draft of
this entry reported a "constant gap of 11 in arm 2c *and* arm 3" when arm 2c could not produce a
gap: the missing field read as 0 and `0 − evictions` looked like a series. A field added after
the copy that runs reads as absent, the same shape as `blocks_freed` reaching no wire. Arm 2c's
independent evidence is its `pool_used`, pinned at 1877. The sampler now asserts its keys are
present in the first row — written after this defect, so it did not catch this one.

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

`blocks_freed` joins the store's stats, measured in `_drop` from the free list before and after,
so a reader can see that eviction is dropping entries without reclaiming anything. `evictions` is
left alone: redefining it touches 23 call sites and collides with `ssd_evictions`.

**And that counter reaches no wire, which I claimed it did.** Verified against the live server
after this was first written: `'blocks_freed' in /health["stats"]` is **False**.
`Engine._build_stats` does not forward the store's dict — it names four keys (`evictions`,
`state_bytes`, `state_bytes_budget`, `demoted`) plus one prefix splat,
`**{k: v for k, v in store.items() if k.startswith(("dram_", "ssd_"))}`. A key matching neither
rule is dropped silently, so a `dram_*` counter would have arrived automatically and this one did
not. The counter and its test are correct; only the observability claim was wrong, and I made it
by reading the module that publishes the field instead of the outermost consumer. One `curl` would
have caught it. Exposing it is a follow-up PR, not smuggled in here; until then the same
measurement reads as `prefix_evictions` rising while `pool_used_blocks` does not fall, since
`pool_used_blocks` is the allocator's own count and independent of the store.

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
(`engine.py:886`) evicts legitimately for a running request and `insert` trims at capacity;
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
