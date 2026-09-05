# The SSD tier's read path did not exist — H20 sm90 card 6, 2026-09-05

> Status: Shipped

## Context

The disk prefix tier's claim is that after a restart HBM is empty, so a returning
conversation's first turn faults its prefix off disk instead of re-prefilling. The
metric is the wall clock of one 2729-token `/v1/messages` request on the first
turn after a restart.

The tier already existed. Its **read path did not**: `spill_kv` and `spill_state`
had a caller in `PrefixStore.insert`, and `load_kv`, `load_state` and `has` had
**zero call sites anywhere in the tree** — grepped, not inferred. So every lookup
missed by construction and the disk accumulated bytes nothing ever read.

Why disk is worth building here at all, when the DRAM snapshot tier shipped off by
default: a runtime LRU tier below DRAM inherits DRAM's condition — a lookup has to
reach past the newest entry, which needs concurrent sessions. After a restart HBM
is empty, so **every** lookup reaches back, with one session.

## What Worked

`lookup` now asks the tier at each prefix length it fails to match in memory. A
hit faults the prefix into fresh blocks and hands it to `insert`, so afterwards it
is an ordinary resident entry — one code path owns retain, eviction and the byte
accounting.

The candidate filter is `resident()`, a dict probe against the in-memory index.
Without it a 2729-token query would pay one or two `torch.load`s per prefix length
to answer a question the index already answers. That same probe suppresses the
write-back, so a fault-in does not re-spill the bytes it just read — asserted,
`ssd_offered == 0` on the faulted arm.

Wired through `--ssd-path`, keyed on `_weight_fingerprint(cfg)`: **every**
dataclass field, not a hand-picked list. The first draft picked seven and named
`cfg.num_heads`, which does not exist on this config — the field is
`num_attention_heads` — so the list was already wrong when written, and a field
left out is how a restart serves KV computed under other weights.

**Two numbers, two scenarios, both measured.** A process restart empties HBM and
leaves the host page cache alone, so the fault-in reads from memory — that is the
common case and what was asked about. A host reboot, or a spill old enough to be
evicted, pays the disk.

| scenario | tier path | baseline | speedup |
|---|---:|---:|---:|
| **process restart** (host cache warm) | **1.690-1.732 s** | 3.004-3.087 s | **1.738-1.821x** |
| host reboot / spill evicted | 1.942-1.944 s composed | 3.041-3.044 s | 1.543-1.565x |

Four consecutive runs, not one: 1.821, 1.761, 1.784, 1.738x, with the empty-tier
control at 0.987-1.030x of cold throughout. The composed row is a measured 1.756 s
standalone disk read (320.6 MiB at 182.6 MiB/s) plus 0.188 s of prefill for the 169
tokens the hit did not cover. Break-even bandwidth is **112.3 MiB/s** against the
device's 182.6, so the read could be 1.60x slower before the tier stops paying.

**Three controls, one of which is the reason to believe the rest.** The empty-tier
control directory stayed at 0.987-1.030x of cold across four runs, so arm order
moves nothing. Compiles are counted per window and were 0 in every one — an earlier
run had 6 in the cold window alone, which by itself made an empty-tier control
3.956x faster than cold, fixed with a throwaway JIT-warming start. And the ceiling:
a hit saves at most the prefill of the tokens it covered, 2560 of 2729 read off the
spill size ladder (the largest entry is the whole prompt, which `_match_prefix`
treats as a miss), so 2.85 s — against 1.34-1.35 s saved, 46.9-47.4% of the bound.

**The bytes/bandwidth division is what made this honest.** Four runs reported
2.585x, 1.738x, 1.784x and 1.821x and every one read like a disk number. 320.6 MiB
at 182.6 MiB/s is 1.756 s against a 1.690 s arm: the arm was faster than its own
disk traffic, so it never did it. All three arm-order controls passed throughout. Recorded in
[errors/2026-09-05-the-ssd-benchmark-never-touched-the-ssd.md](../errors/2026-09-05-the-ssd-benchmark-never-touched-the-ssd.md),
along with writing `3` to `/proc/sys/vm/drop_caches` on a shared host before
reasoning about it.

## What write-through costs, and four wrong guesses at why

Every publish is offered to the tier, so the copy lands inside the prefill that published
it. On H20 card 6, one 2729-token prompt, `--ssd-path` the only variable, arms alternated,
`ssd_hits` 0 in both arms (this is the write path), 0 refusals throughout:

| config | on | off | cost | % | offers | n |
|---|---:|---:|---:|---:|---:|---:|
| every publish, pageable copy | 2.966 | 2.041 | +0.925 | **45.3%** | 6 | 4 |
| every publish, pinned + non_blocking | 2.391 | 2.008 | +0.383 | 19.1% | 6 | 3 |
| every publish, pinned via a (numel,dtype) pool | 2.472 | 2.051 | +0.421 | 20.5% | 6 | 3 |
| every publish, pinned via a 2-slot arena | 3.692 | 2.052 | +1.640 | 79.9% | 6 | 3 |
| last publish only, pageable | 2.276 | 2.069 | +0.207 | 10.0% | 1 | 3 |
| **SHIPPED: last publish, pinned** | 2.196 | 2.015 | **+0.180** | **8.96%** | 1 | 4 |

Two mechanisms, both inside the tier:

* **A GDN snapshot is a CONSTANT ~157 MB at every prefix length** (four .st files measured
  156896685–156901293 B, a 4.6 KB spread). So a 2729-token prompt's six publishes spilled
  941 MB of state plus 683 MB of KV = 1624 MB, to serve **one** 325 MB entry — 5.0x the
  bytes that can ever be read. `insert(..., spill=False)` on the intermediate chunk
  boundaries keeps them in HBM and skips the disk: 1624 MB → 325 MB.
* **`.cpu()` is a synchronous pageable copy.** Pinned destination plus `non_blocking=True`,
  with the flush daemon waiting on a recorded event before it reads the buffer.

**Four guesses at the residual, three of them wrong, and the split is what settled it.**
After the two fixes the cost was still 0.199 s for 325 MB — 1585 MB/s, pageable-class,
which said the remaining cost was not the copy I had just fixed. The guesses:

1. `cudaHostAlloc` per spill. **Wrong**: a (numel, dtype) buffer pool measured *worse*,
   0.421 s against 0.383 s, and in that bench the lengths repeat so the pool was hitting.
2. Pool too shallow. **Wrong in the same direction**: a two-slot arena measured 1.640 s,
   because a request's six publishes outrun a depth the daemon only frees after
   `torch.save`, and 16 of 18 spills fell back to pageable (`ssd_pin_misses` 16).
3. The 171-slice `torch.stack` block gather. **Wrong**: stage timers put it at **1 ms**.
4. The device-to-host copy. **Right**: `ssd_copy_ms` **170 ms** of a 180 ms total cost.

So the cost is one 325 MB D2H, and it does not go below that on the prefill stream: a
pinned `non_blocking` copy returns immediately only if the source is not still being
written by kernels queued ahead of it, and here it is. Moving the spill to a side stream
that waits on a prefill event is the upgrade path, not done.

**Do not build a buffer pool above torch's pinned allocator.** Both hand-written layers
lost to it — `CachingHostAllocator` already reuses pinned blocks, and the pools added their
own contention on top. Both were deleted; the shipped path is `torch.empty(pin_memory=True)`
per spill. The stage timers stay on `/health` (`ssd_gather_ms`, `ssd_copy_ms`) so the next
person reads the split instead of guessing at it a fifth time.

**Verdict: `--ssd-path` stays off by default.** 8.96% on every publishing request against
1.821x recovered once per restart is a good trade only when prefixes are actually re-served
across restarts; that is a deployment property, not something this bench can settle. The
flag is one word and the counters say whether it is paying.

## Durability window

The spill is **not durable at the moment of publish**. `insert` does a GPU→CPU copy and
enqueues; a daemon thread does the `torch.save` off-tick, because a ~100 ms save inside a
prefill tick would cost more than the tier saves. So an entry published within the last
flush is on disk **partially or not at all** when the process dies.

Losing it is correct — the prefix re-prefills, which is exactly what happens today without
the tier. Two things had to be handled so that it stays a loss rather than a fault:

* `_recover` adopts entries **by file size**, so it cannot distinguish a truncated blob
  from a whole one and will index a half-written pair.
* therefore the read path must survive one. Without a guard, `torch.load` on a truncated
  spill raises `PytorchStreamReader failed reading zip archive` **out of `lookup`** — a
  crash for what should be a cache miss. Both loads now treat an unreadable file as a miss
  and drop the key. `test_a_spill_truncated_by_a_crash_is_a_miss_not_a_raise` is
  parametrized **one arm per guard**, because the first version truncated every spill file
  at once and that arm can only ever reach `load_state`: `_fault_in` reads the state first
  and returns on its miss, so `load_kv`'s `except` can be deleted and a both-files test
  still passes. 25 measured exactly that. The `.kv` arm truncates only the KV file, leaves
  the state whole, and asserts `ssd_faults == 1` — that counter is incremented only when
  `load_kv` returns False, so it is what proves the arm reached the second guard. Verified
  by cross-control: deleting `load_kv`'s guard fails `.kv` alone, deleting `load_state`'s
  fails `.st` alone.

What is NOT claimed: no fsync on the publish path, so a host power loss can lose an entry
the daemon believes it wrote. The tier is a cache — every entry is reconstructible by
prefill — so durability past process death was not bought.

## Rule

A restart is the disk tier's whole case: unlike a host-memory tier it needs no
concurrent sessions, because an empty HBM makes every lookup reach back.

And before reporting a storage number, divide the bytes by the device's measured
bandwidth. If the wall clock is smaller, the device was not involved — no
arm-order control detects this, and three of them passed while it was happening.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-05 | eae658e | H20 card 6 | sm90 | Qwen3.8-27B-NVFP4 | 0.619 (restart hit) | — | — |
| 2026-09-05 | eae658e | H20 card 6 | sm90 | Qwen3.8-27B-NVFP4 | 1.114 (cold, no tier) | — | — |

Raw artifacts: `/work/ssdr6.log` (three arms + both scenarios), `/work/ssdr{3,4,5}.log`, `/work/ssdbw2.log`
(bandwidth, three ways), `/work/ssd_restart_{cold,faulted,control}.log` (server
logs, compile counts).
