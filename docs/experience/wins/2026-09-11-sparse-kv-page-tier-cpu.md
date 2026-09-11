# Sparse-KV unit C: page location and demote/promote through the pinned-host path — CPU

> Status: **CPU cell shipped** — the page moves byte-losslessly on the CPU cell;
> the sm90/sm70 device half (pinned H2D throughput, the V100 256k run) is the
> next card sitting. Unit C of design #486.

## Context

Sparse KV keeps the selected pages on the device and moves the rest out. Unit C
is only the mechanism that moves a page: its `location`, the demotion to a host
tier, and the promotion back when a selector names it. Selection itself is
units B (page bounds) and D (learned indexer); this unit is selector-agnostic.

The reuse the design mandates is already in the tree: `DramSnapshots` pins GDN
state snapshots to host RAM through `empty_like(pin_memory=...)` + `copy_`.
KV pages take the same path, so there is one transfer idiom, not two.

## What shipped

- `HostKvPages` (`kv_cache.py`): a byte-LRU pinned-host tier keyed by block id,
  holding one page blob — every plane's K and V and, under fp8, both f32
  `k_scale`/`v_scale` planes. The 27B page is 2.1 MiB, so dropping the scale
  planes would silently reload dequantized garbage.
- `PagedKvPool.demote_page(block)` / `promote_page(old)` and `page_location(b)`
  in `{device, host, free}`. A demotion copies all planes to the host and frees
  the device frame **back to the same pool** — there is no second pool. A
  promotion takes the blob back into a **freshly allocated** block (the id
  changes across a round trip; in a free pool the allocator may hand the same
  frame back, which is correct). A prefix-shared page
  (`refcount > 1`) is refused: it is read-only wherever it lives.
- `build_engine(..., kv_cold_bytes=N)` attaches the tier. `Engine.sparse_retier(keep)`
  applies one selector decision to live requests: a private page not in `keep`
  demotes, a selected host page promotes. Pages keep an **immutable logical
  index** (`_Req.cold_pages`), so promotion splices the fresh block back at the
  page's original position and the rebuilt block table stays in sequence order —
  `paged_attention` derives causal positions from that order, so a score-sorted
  table would mis-cause every page. A selected physical page shared across rows
  is fetched once (one old→new remap per retier).
- `stats()["memory"]` shows demoted bytes as a single held allocation
  `Row("host", "kv_cold", n)` (kind `allocation`) only while pages are demoted,
  plus `kv_cold_pages/bytes/demotions/promotions`. The runtime row is appended
  after `memory_table`'s totals, so today it reports on its own; once unit A's
  derived `--sparse-k kv_cold` enters `plan` (same owner/tier/nbytes), it joins
  `host_total` there. It never enters the device `peak = Σ static + transient`.

The V4.1 grouping (one selection shared by 4 full-attn layers, one host fetch
per group) and the never-demoted 128-token window are **caller responsibilities**:
`keep` already names physical block ids and `demote_page` already moves every
plane of one id, so the API did not change when the grouping did.

## Gates (CPU, `tests/test_sparse_kv_tier.py`)

1. a demoted→promoted bf16 page reads **byte-equal** across K and V of both
   planes, and promotion gives a fresh block when the freed frame is occupied.
2. the fp8 gate asserts all four tensors (K, V, `k_scale`, `v_scale`) byte-equal
   after the round trip; deleting the scale planes from the gather is red only
   here.
3. a prefix-shared page raises on demote and stays on the device.
4. real `RefBackend.paged_attention` over a 4-page context is byte-equal and
   argmax-equal after demoting three pages and promoting them into a rebuilt
   table.
5. the end-to-end gate the design names: two identical tiny engines decode the
   same greedy 6 tokens when one demotes all six private pages to host then
   selects the whole context back before decoding — every device block id may
   change, the tokens may not. The host `kv_cold` row appears while demoted and
   disappears after promotion.

## Rule

A page that leaves the device must carry every plane that makes it readable —
quantized KV without its scale plane is silent corruption, not an approximate
round trip. And when a frame's physical identity changes across a tier round
trip, identity must live in a stable logical index; splicing by block-table
position alone reorders causal positions.
