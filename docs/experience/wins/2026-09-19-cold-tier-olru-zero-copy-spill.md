# Cold-tier O(1) LRU eviction + zero-copy/extent SSD spill — host path, 2026-09-19

> Status: pending-remote (CPU measured; device finalize-tail delta awaits the
> next V100 window). The 2026-09-18 headroom arms proved `--device-headroom-mib`
> cannot touch this tail: 0/384/768/0 all size the same 2213-block device pool and
> all show fill `sparse_finalize` p50 190–222 ms / max 729–1138 ms. The work is in
> the HOST cold tier (`kv_tiers.py`), which the build-time headroom solver never
> prices.

## Context

The sparse cold tier (`HostKvPages`) pins demoted KV pages in host RAM under a
byte budget and spills the LRU pages to an mmap'd file. During a 5×32k cold fill
on V100 sm70 the budget binds and thousands of pages spill. Two host-side costs
sat on that path:

1. `_enforce_budget` rebuilt `list(self._ram_order.items())` on **every hold** and
   inner-scanned it, so eviction was O(resident RAM pages) per page even when the
   oldest entry spilled immediately. Stale LRU records (a blob already
   promoted/forgotten, or a shared record already spilled) were skipped with
   `continue` but left in the dict, so they were re-scanned on every later hold.
2. `ColdSsdFile` grew one slot at a time, and every growth did
   `flush → seek → write → flush → remap` (a whole-mmap rebuild), and each
   `write` did `t.numpy().tobytes()` (a full-page bytes double copy + allocation)
   before an mmap slice assignment.

## What Worked

Three changes, host-only, in `src/tilerl/kv_tiers.py`:

- **O(1) LRU.** `_enforce_budget` pops from the `OrderedDict` front
  (`popitem(last=False)`); stale records are consumed by the pop and gone, never
  re-scanned. A page that cannot leave RAM (refcounted shared, no spill / shared
  spill disabled) is re-parked at the back; one circuit counter keeps the old
  snapshot-for-loop guarantee that a pass which frees nothing returns (no live
  lock). Private/shared eviction order and content semantics are unchanged.
- **Zero-copy spill write.** `_write` mirrors the existing `_read`: an
  `np.frombuffer(mmap, offset, count)` torch view receives the bytes via
  `copy_`, dropping `numpy().tobytes()` and the per-page bytes allocation.
- **Extent growth.** the file and mapping grow once per `GROWTH_SLOTS=64` via a
  single `ftruncate` + one `mmap`, instead of three syscalls plus a whole-mmap
  rebuild per high-water slot. Tail slack is at most one 64-slot chunk.

CPU microbench (hermetic, 256 B fake pages — isolates Python LRU/remap cost, NOT
the real ~2 MiB/page device D2H + SSD I/O), pure drop path (no SSD, isolates the
scan), 4096 holds per row:

| RAM-resident pages R | OLD scan µs/hold | NEW µs/hold |
|---:|---:|---:|
| 256 | 5.5 | 0.8 |
| 1024 | 26.6 | 0.9 |
| 4096 | 167.5 | 0.9 |
| 16384 | 747.5 | 0.9 |

OLD is linear in R; NEW is flat ~0.9 µs (830× at R=16384). On the SSD spill path
at R=4096 (eviction + file growth, still fake pages), 3.90 → 0.068 ms/hold (57×),
the bulk of the OLD constant being the per-slot flush/remap this change removes.

Gates: full CPU suite 1000 passed / 22 skipped / 1 xfailed. Added targeted tests:
constant popitem count and zero `items()` snapshot calls over a 16k fill against a
32-page budget; stale private/shared LRU records are popped and never reappear;
unspillable shared pages re-park instead of KeyError (negative control: the
move-to-end variant fails red); extent growth remaps exactly once per 64 slots and
round-trips every pre-growth slot byte-exact. The two existing tests that pinned
file size to `n*stride` now assert logical slot count plus extent-rounded capacity.

## Rule

A host budget tier that spills per hold must evict from an ordered-map front in
O(1) and grow the backing file an extent at a time; an O(resident) snapshot scan
per hold and a per-slot mmap rebuild are paid thousands of times in one cold fill.
Keep a circuit counter so a pass over only unspillable entries still returns.

## Follow-up

- Device delta pending-remote: re-measure fill `sparse_finalize` p50/max on the
  V100 5×32k cold fill with this host path. The 81→629-1138 ms tail is a mix of
  these host costs and the real page I/O; the split is not yet measured.
- A close-time published-page count (pp-grow keys) is not in the step log; the
  release-path investigation needs that counter to size the close transfer.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-19 | pending PR | CPU (hermetic) | HostKvPages cold-tier evict/spill | — | — | — | 747.5→0.9 µs/hold scan @R=16384 (830×); 3.90→0.068 ms/hold SSD path @R=4096 (57×) |

Raw artifacts: `lru_scan_tmp.py` / `lru_bench_tmp.py` microbench (scratch, not
vendored); tests `tests/test_sparse_kv_tier.py`; changed file
`src/tilerl/kv_tiers.py`.
