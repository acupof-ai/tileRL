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
  and drop the key. `test_a_spill_truncated_by_a_crash_is_a_miss_not_a_raise` truncates
  every spill file to a third of its length and requires the lookup to return None;
  removing either guard reproduces the raise.

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
