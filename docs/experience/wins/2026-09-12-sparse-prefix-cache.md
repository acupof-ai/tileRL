# Sparse prefix cache under the hot pin: drop-only publishing, no holes — CPU, 2026-09-12

> Status: **CPU gates green** — a pinned request publishes nothing while its
> pages stay in the resident union; once the early pages leave, a same-prompt
> follower adopts the published prefix and emits the same tokens as a
> prefix-miss sparse run (sharing is transparent). Stacked on #534
> (cross-tick hot pin). sm70/sm90 share the code; the prefix-read ms is
> pending a card.

## Context

#526 first added the host-blob-backed `SparsePrefixCache` on the pre-pin
engine, where finalize demoted every page every tick. Two defects surfaced
once #534 pinned the cross-tick hot set:

1. It published a clone of every whole page on EVERY tick, although under the
   pin nothing leaves the device for a stable selection — wasted clones and
   entries.
2. `publish()` built a full-length entry while skipping pages whose blob was
   absent (`if p not in page_blobs: continue`): a follower adopted a bounds
   dict with a missing page, and `_sparse_rows` filtered it silently, so the
   follower never attended those pages. A hole in a claimed-contiguous prefix.

## Mechanism

Publishing is keyed to ONE event: a page **leaving the resident union**. In
`_sparse_finalize` the same local `dropped = [p for p in live if p not in
kept]` that drives #534's demote loop feeds the prefix index, so a stable pin
publishes nothing.

Pages leave out of order — the pin keeps a selected page hot for arbitrary
ticks — so the index cannot publish the dropped page in isolation. Per
publisher it tracks:

- `_pending[rid][p]`: the cloned host blob from the page's first drop. A page
  need only drop once: the clone is independent of the private frame, so a
  later re-selection that promotes the private blob does not invalidate it.
- `_snap[rid][m]`: the GDN state snapshot at the boundary after m whole
  pages, captured only on a finalize that lands EXACTLY on it (`q_hi % 16 ==
  0` — a decode page crossing or an aligned prefill chunk). The recurrent
  state then advances past the boundary and is unrecoverable, exactly the
  constraint the dense `PrefixStore` publishes under. Snapshots move to host
  RAM immediately (the device copy is ~144 MiB at 27B).
- `_grow[rid]`: one live entry that (re)attaches at the longest length m for
  which pages 0..m-1 have ALL dropped and the boundary-m snapshot exists.
  Every entry therefore lists exactly the pages it can serve — contiguous
  blob, bound, key and state per page. The old hole is structurally
  impossible.

The grow entry moves into generated-token territory during decode, so a
follower sharing the prompt (different continuation) would never match it.
Mirroring the dense store's first-boundary + prompt-end publishes, an
immutable copy of the grow entry is **frozen** at the first frontier closure
and at the prompt end, each with its own share refs and normal LRU lifetime.

On a hit (`_admit`): adopt the bounds and restore the GDN snapshot, record
page→content-key, set `prefill_from` to the matched length; `_sparse_resolve`
promotes a selected shared page lazily into a private fresh block. A
publisher re-selecting its own dropped page prefers its PRIVATE blob and
falls back to the shared clone only if the private one was byte-LRU evicted.


## Batched-demote ordering

The share clone must happen AFTER the tick's demotions, not inside them. A
batched `pool.demotions()` (#538) launches non-blocking D2H copies and only
holds each blob in the cold tier at the context's single sync; peeking the
private blob inside the scope returns None for every page, so prefix sharing
published nothing (measured: 82/82 peeks empty on a k=2 long-context run,
with and without the draft). Finalize now collects dropped pages inside
`with pool.demotions()` and calls `publish_dropped` after the scope exits,
when every blob is guaranteed held. On the synchronous path `demotions()` is
a passthrough, so the same call site is correct before and after #538 lands;
a deferred-context unit test pins "peek None in scope, blob present after".

## Gates

- `test_sparse_prefix_publishes_only_when_pages_leave_the_hot_union`: a
  5-page prompt wholly inside k+window publishes ZERO; a 24-page prompt
  eventually freezes a full entry; a follower adopts all 24 pages and its 8
  greedy tokens equal a prefix-MISS sparse engine on the identical prompt
  (k=2 is approximate, so the oracle is sparse-miss, not dense).
- `test_sparse_prefix_out_of_order_drops_never_publish_a_hole`: drops in the
  order 2, 1 publish nothing while page 0 is pinned; after page 0 drops the
  3-page entry lists exactly 3 pages, every content key resolves to a held
  byte-equal blob, and the fourth drop closes the prefix.
- `test_sparse_prefix_republished_after_repin_keeps_the_first_blob`: a page
  dropped, re-pinned and dropped again keeps serving the FIRST captured blob.
- Full CPU suite: 675 passed, 20 skipped, 6 xfailed; the one red is the
  pre-existing environmental `websockets` import in test_chat_ui.

## Rule

Under a cross-tick pin, publish a shared page exactly once — when it leaves
the resident union — buffer out-of-order drops behind the contiguous
frontier, and never let an entry name a page whose blob, bound or exact
boundary state you do not hold. Freeze at prompt boundaries; the live entry
belongs to the continuation.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-12 | Mac CPU (tiny cell, real pool) | cpu | pinned short prompt publishes 0; 24-page prompt freezes a full entry; follower hit == sparse-miss tokens; out-of-order/hole and re-pin gates green |
