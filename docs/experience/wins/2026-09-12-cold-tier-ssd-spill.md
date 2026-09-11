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
capacity tier keyed by the cold key the engine already carries.

## What changed

`HostKvPages(budget_bytes=B, ssd_path=FILE)` keeps two levels:

- **pinned host RAM up to `budget_bytes`** — the LRU hot cold set, unchanged;
- **one mmap'd spill file past it** — fixed-stride slots at
  `offset = 4096-header + slot*stride`, so there is no on-disk index: every
  page blob is identical in shape/dtype, described once by a JSON header. The
  caller keys a page by an OPAQUE cold key (an int physical id on the #500
  retier seam, a `(req, page)` tuple on the sparse path); `ColdSsdFile` maps
  key -> monotonic file slot itself. The key cannot BE the slot: the pool
  recycles a freed physical frame while an older page stays cold (#528 fixes
  the same collision in host RAM). The file grows on a new slot (seek-to-end
  + remap, never ftruncate alone — truncate is not visible to this process's
  Python-side file size); freed slots return to a LIFO free list. Presence
  is the in-memory key->slot map, so a reopened tier does not resurrect
  pages — this is the serving spill, and boot stays KvBootStore's
  prefix-keyed job.

Demotion, promotion and release go through the existing seams:

- `hold` evicts LRU to SSD instead of dropping once a path is attached; a page
  larger than the whole budget demotes straight to the file;
- `take` reads host RAM or the SSD slot through the same
  `promote_page`/`promote_keyed` -> fresh-block -> copy path (pinned source
  when the pool is on a card);
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
- `test_a_recycled_frame_spills_two_pages_to_ssd_under_distinct_keys`:
  a two-frame pool demotes page A under key `(0,0)` and reissues the freed
  frame to B demoted under `(0,1)`; the two land in distinct file slots and
  both promote back byte-equal. Keying the slot by the physical id aliases
  the two and the second demote fails on the stale key.
- `test_a_spill_file_is_one_process_not_boot`: a reopened tier over the same
  file returns None — presence is in-memory, boot is a separate component.
- `test_cold_tier_spills_past_the_host_budget_and_the_ledger_splits_tiers`:
  a one-page budget in a live sparse engine spills most of a six-page context;
  the memory ledger carries both `kv_cold`/host and `kv_cold_ssd`/ssd rows
  matching the tier byte counters.

## Rule

Identical fixed-stride slots need no on-disk CONTENT index, but the key-to-slot
map is one level of indirection the file must own: a key that is ever recycled
(here, a freed physical frame) cannot be used as the slot number, or two live
pages alias one slot. Keep the indirection in memory and the spill stays
single-process and index-free on disk. Capacity spill and cold boot stay two
components — one keyed by the engine's opaque cold key for one process, one by
content hash across processes.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-12 | Mac CPU (tiny cell) | cpu | 4 pages across a 2-page host budget: 2 RAM / 2 file, all byte-equal on promote; ledger host+ssd split reconciles |
