# Sparse prefix spill no longer copies every cold page into anonymous host RAM — 2026-09-13

> Status: **fix landed on the CPU cell; the H20 256k rerun is
> pending-remote** and closes [the OOM entry](../errors/2026-09-12-sparse-256k-spill-host-rss-oom.md).

## Context

Two V100 runs of the 27B sparse 256k prefill with the mmap cold spill
host-OOM'd (SIGKILL 137) at 16–19 min while VmRSS climbed monotonically to
30.7 GiB at ~22 MiB/s — ~3.6 GiB per 16k-token window against the 1.0 GiB of
f16 KV a window demotes. Page cache was ruled out (`Cached` 3.98 GiB,
`Dirty` 0), so the bytes were anonymous process memory, and changing the
host/SSD split (10/8 → 6/12) changed only the time to the kill: the
accumulating copies were not bounded by `HostKvPages.budget_bytes`.

A direct probe settled it: demoting 4096 pages through the PRIVATE SSD tier at
a 4-page host budget keeps `HostKvPages` host bytes flat at the budget. The
retaining copies were on the sparse PREFIX path, which `tilerl serve` enables
by default (the fidelity harness uses `NoPrefixStore`, so its runs did not
show it):

1. every page leaving the resident hot union was `_sparse_clone_cold`-cloned
   and `share_hold`-held in `HostKvPages._shared` — a SECOND full f16 KV copy
   with no byte budget and no spill, and the live grow entry exempt from LRU;
2. `SparsePrefixCache._snap` kept every consumed chunk-boundary GDN snapshot
   until the request ended.

## Fix (`src/tilerl/kv_cache.py`, `sparse_engine.py`, `engine.py`)

- **Transfer, not clone.** A dropped page's private blob is rehomed to its
  content key (`HostKvPages.share_hold_kv`): one allocation moves from the
  private to the shared namespace; the `dict(pend.pop())` clone is gone.
- **One budget, one LRU.** Private and shared-prefix bytes now share the one
  pinned budget and a single RAM LRU (`_ram_order`). Over-budget shared pages
  spill to a prefix `ColdSsdFile` (`<spill>.prefix.bin`, the slot carries the
  bounds plane) and `share_take` reads them back through mmap; a promoted copy
  does not pin the page in RAM. Bounds are read by field (`read_field`) on a
  prefix hit instead of being pinned inside the entry dict.
- **Snapshots consumed once.** When a frontier closes at boundary m the GDN
  snapshot it attaches is popped from `_snap`; the entry owns the only copy.
- A page whose private blob is already on the private SSD when the frontier
  closes is lifted off disk once and written to the prefix spill file — still
  nothing added to host RAM.

## Gates (CPU tiny)

- `test_ssd_spill_with_prefix_sharing_keeps_host_bytes_under_budget`: 4096
  pages demoted through the SSD tier with the prefix share path active and a
  4-page host budget; total host bytes (private + shared) stay ≤ budget + one
  page on every page, all 4096 still resolve (RAM or file), and releasing the
  refs leaves zero host/shared-file residue. This is the "a third copy cannot
  hide" assertion.
- `test_shared_transfer_of_an_already_spilled_private_page_reads_back`: a page
  already on the private SSD when its frontier closes transfers to the shared
  namespace without growing host RAM and reads back byte-equal with bounds.
- `test_prefix_publish_consumes_boundary_snapshots_no_second_copy`: after a
  full-prefix closure no consumed boundary snapshot remains in `_snap`.
- the existing prefix/SSD/tier suites and full hermetic run stay green
  (709 passed).

## Rule

A capacity tier that spills past a host budget must put EVERY retained copy of
a page under that budget, including a cache that "shares" it — a clone taken
so a follower can reuse a page is still anonymous RSS. When a blob moves
between namespaces, transfer the allocation; when it can't fit RAM, spill it;
and assert total host bytes across all containers, not each container in
isolation.
