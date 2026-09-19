# Shared-prefix SSD spill ignores the cold-SSD byte cap and never returns disk — 2026-09-19

> Status: landed behind an env gate (`TILERL_COLD_PREFIX_SSD_CAP=1`), default
> OFF, device confirmation pending-remote. Observed on V100 during the 2026-09-18
> headroom window: the `.prefix.bin` shared-prefix spill grew to 13.6 GiB logical
> / 25 GiB physical against an explicit `--cold-ssd-bytes` of 8 GiB.

## Context

Two distinct spill files sit behind one `--cold-ssd-bytes` setting:

- the **private** cold page spill (`sparse_cold.bin`), whose capacity is counted
  in admission (`HostKvPages.ssd_capacity_bytes`, `cold_capacity_blocks`);
- the **shared-prefix** publish cache (`sparse_cold.prefix.bin`), which
  `_write_shared_ssd` wrote with **no capacity check at all**. Every published
  prefix page was appended regardless of the cap. A published prefix is
  serve-only (never read back after restart), so this was pure disk growth, not
  retained state.

The #735 extent growth made the mapping cheap (one ftruncate+mmap per 64 slots)
but only **reused** freed slots through the LIFO free list; it never shrank the
file, so a release wave left physical disk at the high-water mark forever.

## Fix (env-gated, default unchanged)

`TILERL_COLD_PREFIX_SSD_CAP=1` enables two behaviors on the shared spill only;
the private spill and every default (gate unset) path is unchanged:

1. **Capacity admission.** `_write_shared_ssd` refuses a page that would push
   `_shared_ssd_bytes` past `ssd_capacity_bytes` (the same `--cold-ssd-bytes`
   figure the private spill reports). It returns False, which is the existing
   "shared page stays in host RAM / spill disabled" safety path — a published
   cache may be forgotten but never raises or wedges a tick. The shared page
   remains readable from RAM; a later follower misses the prefix rather than
   reading a corrupt/oversized file.
2. **Trailing-extent reclaim.** `ColdSsdFile(reclaim=True)` tracks live slots
   per extent; `forget` truncates the file back when every extent at the
   high-water end is empty (one ftruncate + remap), so a release wave returns
   physical disk. Interior free extents still cycle through the LIFO free list.

Tail-collapse is the safe, portable reclaim: it uses ftruncate only (no
platform punch-hole ioctl). Mid-file fragmentation is left as a follow-up
(FALLOC_FL_PUNCH_HOLE / F_PUNCHHOLE) if physical bytes ever exceed the cap with
a non-trailing free pattern. `stats()` adds `kv_cold_shared_ssd_bounded`.

## Gates

CPU tier tests (hermetic, no device):

- bounded mode refuses the page past an N-page explicit cap, keeps it in RAM,
  and the counter stays at the cap; with the gate unset the same write is
  allowed (default behavior pinned);
- reclaim truncates on a freed trailing extent, leaves the file untouched when
  only an interior extent frees, collapses to the header after the last live
  slot, and surviving extent-0 pages still round-trip byte-exact; with
  reclaim=False forgetting never shrinks (default pinned).

Both assertions were mutation-verified red with the cap test and the shrink
call independently disabled. Full CPU suite 1008 passed. The real on-disk
reclaim and the cap interaction with a 25 GiB-high-water file are
pending-remote on the next V100 window (the fix is opt-in until that read).

## Rule

A publish-only cache file does not inherit the capacity of the state spill
beside it — admission has to be enforced on every spill path, not assumed from
a shared flag. Extent growth needs a matching trailing-extent collapse, or the
LIFO slot reuse hides unbounded physical growth behind a bounded live slot
count.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-19 | pending PR | CPU (hermetic) | HostKvPages shared-prefix spill cap + reclaim | — | — | — | bounded cap enforced; trailing extents truncated to header (device disk reclaim pending-remote) |

Raw artifacts: tests `tests/test_sparse_kv_tier.py`; change `src/tilerl/kv_tiers.py`.
