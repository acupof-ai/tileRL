# Sparse 256k prefill host-OOMs on the SSD spill path: anonymous RSS grows to 31 GiB regardless of the host/SSD split — 2026-09-12

> Status: **fixed (CPU gates; H20 256k rerun pending-remote).** The retaining
> site was not the private SSD tier but the sparse PREFIX path (serve attaches
> `SparsePrefixCache` by default; the fidelity driver's `NoPrefixStore` does
> not, which is why its runs did not show it). Two uncapped anonymous
> containers: (1) `HostKvPages._shared` — every page leaving the resident
> union was cloned once for the shared prefix index and held with no byte
> budget and no spill (a second full f16 KV copy, the ~3.6x/window);
> (2) `SparsePrefixCache._snap` retained every consumed chunk-boundary GDN
> snapshot until request end. Fix: private→shared is a blob TRANSFER under one
> pinned budget with one RAM LRU, shared pages spill to a prefix `ColdSsdFile`
> with read-through, and consumed snapshots are popped. CPU gate demotes 4096
> pages through SSD with sharing on and asserts total host bytes ≤ budget + one
> page. The H20 card-3 256k rerun is this entry's closing bench line.
>
> Original report below.

## Context

V100-SXM2-32GB (31 GiB visible host RAM), sm70, torch 2.5.1+cu121, 27B NVFP4
(~27.6 GiB GPU weights, ~19 GiB device held during the run). Sparse k=128
bounds, f16 cold pages (1 MiB K+V/page), slots 1, eager, 262144 prompt tokens,
host cold tier + one mmap'd SSD spill file (`ColdSsdFile`, #550). The f16 KV
set at 256k is ~16 GiB, so admission needs host + SSD capacity ≥ that.

| run | host tier | SSD cap | result |
|---|---:|---:|---|
| M-tile baseline (main fadb2726) | 20 GiB | none | **completed**, prefill 47.4 min / 10.84 ms/tok |
| attempt 1 (#537 b2d49730+#550) | 10 GiB | 8 GiB | SIGKILL rc=137 at **15.5 min**, no traceback |
| attempt 2 (#537 5c12e7e7+#550) | 6 GiB | 12 GiB | SIGKILL rc=137 at **19.3 min**, no traceback |

Changing the split 10/8 → 6/12 changed only the time to the kill, not the
outcome: the bytes that accumulate are not bounded by `budget_bytes`.

## Measured signature

A 10 s sampler (`/proc/meminfo`, system-wide) on attempt 2, seconds from launch:

```
  t   MemAvail  MemUsed  GPU
 10s    17 GiB   13 GiB   2.8 GiB   (weights loading)
 90s    22 GiB    8 GiB  18.6 GiB   (weights resident, prefill underway)
190s    17 GiB   13 GiB  18.7 GiB
391s    10 GiB   20 GiB  18.7 GiB
692s     4 GiB   26 GiB  18.7 GiB
892s     0 GiB   30 GiB  19.0 GiB
1157s    0 GiB   31 GiB  19.0 GiB   -> OOM kill at 19m22s
```

After weights settle (~50 s, GPU steady at 18.7 GiB), host used climbs
8 → 31 GiB over ~1070 s: **~22 MiB/s of steady growth during prefill**. At the
M-tile prefill rate (~10.8 ms/tok) that is **~3.7 GiB per 16k-token window**,
against 1 MiB/page × 1024 pages = **~1 GiB of f16 KV** for that window — a
~3.6x over-retain, so this is not "the cold set, once."

Decisive end-state snapshot on the python pid moments before the kill:

```
VmRSS:   30717748 kB     (~30.72 GiB, anonymous process memory)
VmHWM:   30725028 kB     (RSS == HWM: monotonic growth, no release phase)
VmSwap:  0
/proc/meminfo: MemAvailable 88 MiB, Cached 3.98 GiB, Dirty 0, Writeback 64 kB
```

**It is process RSS, not the spill file.** The mmap-backed growth hypothesis
would show as page cache (`Cached`) and/or dirty-unwritten pages (`Dirty`);
Cached is only 3.98 GiB and Dirty is 0 while anonymous RSS holds 30.7 GiB.
Device OOM is ruled out separately — that exits rc=1 with a CUDA traceback,
not SIGKILL, and GPU memory was flat at 18.7–19.0 GiB. The run's own
5400/6600 s watchdog is excluded (it was still sleeping when both kills hit).

## What is and is not proven

Proven: ≥ ~22 MiB/s of **anonymous, unreleased** host allocation on the
spill-enabled prefill path, independent of the host-budget/SSD-cap split,
absent without the spill file.

Not proven: the retaining site. Candidates in `kv_cache.py`, in reading order:

1. `ColdSsdFile.write` does
   `blob[k]...view(uint8).numpy().tobytes()` then a second copy into the mmap
   slice — two full-size temporaries per plane per page. They should die with
   the call; a retained reference or GC-not-collected cycle would leak at
   exactly this rate. Needs the per-pid curve plus refcount tracing to confirm.
2. Spill demotion pops the pinned blob from `HostKvPages._blobs`
   (`_evict_to_ssd`), but pinned pages freed through the host allocator
   may not be returned to the OS (cudaHostUnregister/heap behaviour) — if the
   budget tier churns pin/unpin every tick, the arena can grow even though
   `bytes_held` is capped at 6 GiB. This matches "growth independent of split."
3. A per-page host copy retained outside the LRU (promotion staging, pending
   promotion context, or a key kept in two dicts).

The instrument that separates them (not yet run): per-pid `VmRSS` and
`HostKvPages._used` / `_ssd_bytes` / demotion counters every tick, plus
`torch.cuda.memory_stats` host pinned bytes. The sampler shipped here logged
system `MemAvailable`, which proves anonymous RSS vs page cache at the end but
does not attribute the growth to a call site. That per-pid curve is the first
deliverable of the fix PR.

## Rule

A capacity tier that spills past a host budget must prove the spilling process
does not retain a host copy per spilled page; otherwise the SSD raises the
admission ceiling without raising the RSS ceiling, and the run dies later
instead of fitting. System `MemAvailable` plus a single end-state
`VmRSS`/`Cached`/`Dirty` distinguishes process leak from writeback pressure;
attribute to a call site with a per-pid RSS curve, not from the system number.
