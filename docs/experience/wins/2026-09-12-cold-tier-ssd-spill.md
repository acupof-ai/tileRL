# Cold KV pages spill past the host budget to one mmap'd file — CPU, 2026-09-12

> Status: **Shipped on CPU** — demote/promote parity gated through a real
> PagedKvPool, ledger split gated through the sparse engine. sm70/sm90 take the
> same code path target-independently; the SSD ms number is pending a card.

## Context

The sparse cold tier (`HostKvPages`, #500/#518) caps demoted pages at a pinned
host-RAM budget; pages past it were dropped. ckl's requirement is that serving
keep the whole long context on cheaper storage ("能从 ssd 启动" for serving,
not just the prefix-boot store #514): once host RAM fills, cold pages spill to
disk and promote back when the selector names them. Boot (#514) keys pages by
a prefix hash with a manifest; this is the different case — an in-process
capacity tier keyed by the block id already in every block table.

## What changed

`HostKvPages(budget_bytes=B, ssd_path=FILE)` keeps two levels:

- **pinned host RAM up to `budget_bytes`** — the LRU hot cold set, unchanged;
- **one mmap'd spill file past it** — fixed-stride slots at
  `offset = 4096-header + block_id*stride`, so there is no on-disk index: every
  page blob is identical in shape/dtype, described once by a JSON header. The
  file grows on a larger block id (seek-to-end + remap, never ftruncate alone —
  truncate is not visible to this process's Python-side file size). Presence is
  an in-memory set, so a reopened tier does not resurrect pages — this is the
  serving spill, and boot stays KvBootStore's prefix-keyed job.

Demotion, promotion and release go through the existing seams:

- `hold` evicts LRU to SSD instead of dropping once a path is attached; a page
  larger than the whole budget demotes straight to the file;
- `take` reads host RAM or the SSD slot through the same
  `promote_page` → fresh-block → copy path (pinned source when the pool is on a
  card), so the engine's `_sparse_resolve` is unchanged;
- `forget` drops a page from whichever tier holds it.

The ledger splits the owner by tier: `kv_cold` (host) and `kv_cold_ssd` (ssd),
both held non-device allocations, so `/health` shows host and SSD bytes
separately and neither enters the device peak residual. New flag
`--cold-ssd-path FILE` (distinct from `--ssd-path`, the prefix-boot store).

## Gates

- `test_pages_past_the_host_budget_spill_to_ssd_and_promote_byte_equal`:
  two-page host budget, four demoted pages; two sit in RAM, two in the file;
  every page promotes back through the real demote/promote seam byte-equal
  across all planes; file size is header + stride per occupied slot.
- `test_a_spill_file_is_one_process_not_boot`: a reopened tier over the same
  file returns None — presence is in-memory, boot is a separate component.
- `test_cold_tier_spills_past_the_host_budget_and_the_ledger_splits_tiers`:
  a one-page budget in a live sparse engine spills most of a six-page context;
  the memory ledger carries both `kv_cold`/host and `kv_cold_ssd`/ssd rows
  matching the tier byte counters.

## Rule

A spill tier that already has stable integer keys (here, block ids) needs no
on-disk index: fixed-stride slots plus an in-memory presence set are enough,
and the index's failure modes (corruption, rebuild, skew) disappear. Capacity
spill and cold boot are two components — one keyed by live block id for one
process, one by content hash across processes.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-12 | Mac CPU (tiny cell) | cpu | 4 pages across a 2-page host budget: 2 RAM / 2 file, all byte-equal on promote; ledger host+ssd split reconciles |
