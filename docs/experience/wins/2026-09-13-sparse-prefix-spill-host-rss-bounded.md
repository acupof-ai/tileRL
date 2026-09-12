# Sparse prefix spill no longer copies every cold page into anonymous host RAM — 2026-09-13

> Status: **fixed and verified on H20 card 3 (2026-09-13).** 256k sparse
> prefill with a 6 GiB host tier + 12 GiB SSD keeps cold host bytes pinned at
> 6.00 GiB and total process RSS at ~7.9 GiB through the whole prefill+decode
> (the pre-fix run reached VmRSS 30.7 GiB and SIGKILL at 16–19 min). Closes
> [the OOM entry](../errors/2026-09-12-sparse-256k-spill-host-rss-oom.md).

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

## H20 card-3 256k measured (2026-09-13, head 2fb12451)

262144 tokens, sparse k=128 bounds, slots=1, host cold 6 GiB, SSD capacity
12 GiB, `scripts/trace_256k_spill_rss.py`, VmRSS + cold COLD_STATS every 10 s.

| t (s) | phase | RSS GiB | cold-host GiB | shared-SSD GiB | demotions |
|---:|---|---:|---:|---:|---:|
| 20 | prefill | 2.54 | 0.83 | 0.00 | 800 |
| 120 | prefill | 5.61 | 4.02 | 0.00 | 3872 |
| 220 | prefill | 7.64 | 6.00 | 0.31 | 6080 |
| 420 | prefill | 7.89 | 6.00 | 3.79 | 9440 |
| 620 | prefill | 7.91 | 6.00 | 6.62 | 12160 |
| 821 | prefill | 7.77 | 6.00 | 9.04 | 14496 |
| 1022 | decode | 7.82 | 6.00 | 10.99 | 16379 |
| 1028 | done | 7.68 | 5.97 | 10.99 | 16380 |

max cold-host over 103 samples **6.001 GiB** (budget 6.0 + one-batch slack);
zero over-budget samples; max RSS **7.91 GiB**, flat from ~t=200 while shared
SSD climbed 0.3 → 11.0 GiB. Pre-fix the same run reached 30.7 GiB RSS and
died SIGKILL at 16–19 min.

**256k sparse prefill: 997.6 s ≈ 16.6 min** (H20 eager sm90, post-#546) —
the long-ctx sparse wall-clock row cc could not take on V100 (the V100 was
host-RAM bound for 128k/256k regardless of speed). 64 decode tokens appended;
total 1027.5 s.

The card run also caught one CUDA-only defect the CPU gates miss: the shared
Quest bound is a device tensor and broke `ColdSsdFile.write` (numpy) on spill;
fixed by `.cpu()` at transfer plus a defensive host-move in `ColdSsdFile.write`.
