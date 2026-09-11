# Cross-tick hot pin: selected sparse pages stay resident, one H2D per tick — 2026-09-12

> Status: **CPU gates green; the decode-recovery number is the V100/H20 re-run.**
> Follows #528. F's first cut demoted every private page at each finalize, so
> each decode token re-promoted up to n_groups*k pages and paid a
> `cuda.synchronize()` per page — cc measured decode 1.25 vs 8.28 tok/s dense
> (6.6x slower) at V100 32k while prefill was already 1.74x faster.

## Root cause

Two costs the first cut paid on every decode token:

1. **No residency across ticks.** `_sparse_finalize` demoted ALL resident
   private pages, including the pages it had just selected. The next tick's
   selection overlapped the old set almost entirely (the hot set moves slowly),
   so it re-promoted nearly the whole set.
2. **A device sync per promoted page.** `promote_keyed` ended in
   `torch.cuda.synchronize` so the pinned blob could be released on return.
   Promoting 512 pages in a tick meant 512 full-device syncs.

## Fix

**Pin the selected union, demote only what left it.** `SparseForward` reports
`selected_pages(bi)` = the own span unioned with every source group's top-k
(this is the n_groups*k union #528 sizes the pool for). `_sparse_finalize` keeps
those pages resident in `tr.resident`/`r.blocks` (rebuilt in logical order,
which paged_attention derives causality from) and demotes to the host only the
resident pages that dropped out of this tick's set. A stable selection therefore
moves zero pages tick to tick.

**Within-tick room for a changed selection.** When a newly selected page needs a
frame, `_sparse_resolve` evicts one resident page that THIS tick does not reserve
(the own span plus every group's already-chosen page), demoting it before
allocating. The tick carries a per-row `reserved` set so the victim picker can
never evict a page the current attention will read. The n_groups*k + window +
chunk pool sizing guarantees a victim exists.

**One batched H2D per tick.** `PagedKvPool.promotions()` is a context that flags
promotes non-blocking, retains their source blobs, and synchronizes exactly once
at exit. The sparse forward runs `_sparse_rows` and the model forward inside one
context, so a tick that fetches N pages — own-span resolves plus in-forward
selection promotes — launches N copies and syncs once, not N times.

## Gates (tiny CPU cell)

- `test_sparse_pins_selected_pages_across_ticks_and_demotes_what_leaves` — over a
  12-page context (pages genuinely fall outside k+window), each decode tick's
  promotions equal the pages that started the tick cold (newly chosen only); a
  kept page is never re-fetched.
- `test_sparse_a_stable_selection_promotes_nothing_after_the_first_tick` — a
  forced-identical selection moves zero pages between device and host from the
  second stable decode tick on (the direct gate for the 6.6x regression).
- `test_batched_promotions_copy_many_pages_but_sync_once` — three promotions
  inside `pool.promotions()` on a cuda-flagged pool observe zero syncs before
  context exit and exactly one after, with every page byte-equal on return.
- The #528 long-ctx recycling gate (13-frame pool, 512 tokens) and the dense
  #500 tier seam stay green; full hermetic suite 702 passed.

## Rule

Under a tiered KV cache the selection, not the tick, is the residency boundary:
keep a page as long as the selector still names it, move it only when it leaves
the union, and batch the refill for the changed set behind one synchronization
point. Per-transfer synchronization inside a loop over cached pages turns a
bandwidth problem into a launch-count problem.

## Pending card number

cc re-runs the V100 32k point on this head: the design expectation is decode at
or above the dense 8.28 tok/s (attention reads ~1/8 the keys), against the
current 1.25. Promotions must track only selection changes (zero on a stable
sliding window) and the tick must show one H2D sync.
