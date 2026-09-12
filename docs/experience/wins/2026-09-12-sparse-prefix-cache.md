# Sparse serving keeps prefix caching: host-blob-backed SparsePrefixCache — CPU, 2026-09-12

> Status: **Shipped on CPU** — a same-prefix follower adopts the published
> prefix (zero prefill of it) and emits dense-identical tokens. sm70/sm90 share
> the code target-independently; the prefix-read ms is pending a card.

## Context

The first sparse engine forced `NoPrefixStore`: finalize demotes every private
device page to host RAM, so the block-retaining dense `PrefixStore.insert(
req.blocks[:N])` received an empty list and crashed every request crossing the
first 64-token publish boundary ("64 tokens need 4 blocks, got 0"). That was
correct as a stopgap but it turned prefix caching off for every sparse
deployment, which cannot stand once sparse is default. This PR removes the
stopgap and implements the host-tier publishing path (design note PR #522).

## Mechanism

Sparse prefix entries cannot retain live device blocks (the device frame is
freed on demote), so the index retains **host blobs**, content-addressed:

- `HostKvPages.share_hold/take/release` hold a page blob keyed by the rolling
  hash of its 16-token content span, refcounted across every entry/follower —
  distinct from the private block-id cold blobs (`PagedKvPool.shared_promote`
  copies a read-only shared blob into a fresh private block).
- `SparsePrefixCache` maps a block-aligned token prefix to its per-page content
  keys, each page's fp16 Quest bounds, and one GDN snapshot at the boundary,
  with LRU eviction that releases blob refs.
- **Publish** (in `_sparse_finalize`): a whole page stays in the request's
  PRIVATE cold tier (the publisher's continuation is untouched) AND a clone is
  share-held for the index. Keeping the private copy is essential: moving the
  publisher's own pages out to shared-only corrupted its continuation (a
  measured failed experiment — the publisher must resolve its hot window
  through the private path).
- **Hit** (in `_admit`): adopt bounds (zero recompute) and restore the GDN
  snapshot, record page→content-key, and set prefill_from to the matched
  length — no device blocks allocated. `_sparse_resolve` promotes a selected
  shared page lazily into a private fresh block only when the selector names
  it.
- The dense block `PrefixStore` stays `NoPrefixStore` for sparse (its
  retain/free path never touches sparse pages); sharing is entirely on the
  tracker index. An explicitly-passed `NoPrefixStore` still disables sharing
  (training/parity).

## Gates

- `test_sparse_engine_publishes_and_a_same_prefix_follower_matches_dense`:
  the old 83-token crash shape now completes with `prefix.published > 0`; a
  second request sharing its 80-token prefix hits (`sparse_matched == 80`,
  prefills only the 5-token tail) and its 8 greedy tokens equal a dense engine
  on the identical prompt.
- Existing sparse/tier gates unchanged (sharing-off path with an explicit
  NoPrefixStore is byte-identical).

## Rule

A content-addressed page in host RAM is a prefix-cache entry that needs no
device frame: key the shared blob by the page's token hash, carry the bounds
and the boundary state with it, and promote lazily as a private copy. The
publisher keeps its private copy; only followers read the shared one.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-12 | Mac CPU (tiny cell) | cpu | 83-token publish, 80-token follower hit, follower tokens == dense |
