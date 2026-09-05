# The SSD tier's benchmark never touched the SSD — 2026-09-05

> Status: instrument fixed; the disk number is now composed rather than measured
> as a wall clock, and the wall clock is relabelled rather than withdrawn

## Context

The SSD prefix tier's claim is that after a restart HBM is empty, so a returning
conversation's first turn faults its prefix off disk instead of re-prefilling.
The bench for it is three server starts on H20 card 6, one measured request each:

| arm | spill dir | wall_s | compiles | ssd_hits |
|---|---|---:|---:|---:|
| cold | empty | 3.041 | 0 | 0 |
| faulted | what cold just wrote | **1.690** | 0 | 1 |
| control | a different empty dir | 3.077 | 0 | 0 |

Every control passed. The empty-tier control landed at **1.012x** of cold, so arm
order was not moving the clock. Compiles were 0 in all three windows (an earlier
run had 6 in the cold window alone, which by itself made an empty-tier control
3.956x faster — fixed with a throwaway JIT-warming start). The faulted arm took
exactly one SSD hit and recovered 6 entries. `faulted/control = 1.821x`.

Then I measured the device the tier writes to: **182.6 MiB/s** for the two files
one fault-in reads, reproduced three ways (serial over all 14 files 199.1,
one entry 182.6, one entry at 16 MiB blocks 182.1).

One entry is 320.6 MiB (156.9 MB of GDN state + 167.8 MB of KV). At 182.6 MiB/s
that read takes **1.756 s**. The whole faulted arm — HTTP, template, tokenize,
the fault-in, the remaining prefill, 8 decode steps — took **1.690 s**.

**The arm was faster than reading its own bytes off the disk, so it never read
them off the disk.**

## Root Cause

The bench creates the condition it is trying to avoid. The cold arm **writes** the
spill; the faulted arm starts seconds later and reads it. A file just written is
in the host page cache by construction, and write-through guarantees every byte
of it is — that is what write-through means. The host has 795 GB of buff/cache
against a 1.9 GB spill, so nothing evicts it.

The mistake was the **label**, not the measurement: I was reporting an arm that
reads from memory as the disk tier's number. (A process restart really does leave
the host cache warm, so that arm is the right number for the restart scenario —
it just is not the disk's. That distinction came from review and is the reason
this entry says "relabelled" rather than "withdrawn".)

The three controls I did build were all pointed at the *other* confound. Arm
order, JIT compiles, an empty control directory: every one of them asks "is this
speedup the tier or the schedule?" and none asks "is this tier the disk or the
cache?" I checked the axis I had already been burned on today — the 09-05 self-hit
entry is the cold/warm arms sharing a process — and the new axis went unexamined
until an unrelated bandwidth probe contradicted it.

The tell was in the first run's own row and I read past it:
`ms_per_prompt_token: 0.399` for the faulted arm. Reading a 2560-token prefix from
a 182.6 MiB/s device cannot cost 0.4 ms per token no matter what the tier does.

## Fix

`_evict_cache` in `scripts/bench_ssd_restart.py`, called between the cold and
faulted arms: `fsync` then `posix_fadvise(POSIX_FADV_DONTNEED)` per spill file.
The fsync is load-bearing — `DONTNEED` silently skips a dirty page, and these were
written seconds earlier by the tier's flush daemon — but it is **not sufficient**:
the probe went from 4477.8 MiB/s (fully cached) to ~509, against 182.6 for a
genuinely cold read. `DONTNEED` only drops pages nothing else references.

So the disk number stopped being chased with more eviction and is **composed from
measured parts** instead. Two numbers, two scenarios, both real — which is the
correction to my first draft of this entry, where I called the wall clock
withdrawn:

| scenario | number | how |
|---|---:|---|
| **process restart** (host cache warm) | **1.821x** | the faulted arm's own wall clock, 1.690 s vs control's 3.077 s |
| host reboot / spill evicted | 1.564x | 1.756 s measured standalone read + 0.188 s tail prefill vs 3.041 s |

A process restart empties HBM and leaves the host page cache alone, so a fault-in
reading from memory is not a confound there — it is what production does. The
confound was only ever the *label*: three runs of that arm read as the disk tier's
number. Break-even bandwidth is 112.4 MiB/s against the device's 182.6, so the
read could be 1.60x slower before the tier stops paying.

Two checks added, both from the review that asked for them:

* **the win is against the control, not against cold.** `control` carries whatever
  start-order effect survives, so `faulted/control` is the tier's number.
* **a ceiling, computed before the verdict.** A hit can save at most the prefill of
  the tokens it covered, at the cold arm's own per-token rate. The matched length
  comes off the spill size ladder — the largest entry is the whole prompt and
  cannot serve, since `_match_prefix` treats a full-length hit as a miss, so the
  servable one is the second largest: 2560 of 2729 tokens. Ceiling 2.852 s, saved
  1.351 s = 47.4% of it.

## Rule

A cache benchmark that writes its own fixture has already invalidated itself. The
write puts the bytes in every layer above the one under test, so "read it back"
measures the topmost layer that still holds them — and every arm-order control in
the world will pass while it does.

Before trusting a storage number, divide the bytes by the device's measured
bandwidth and compare with the wall clock. If the wall clock is smaller, the
device was not involved. That one division is cheaper than the three controls I
did build, and it is the only one that caught this.

**No global cache or memory operation on a shared host.** I wrote `3` to
`/proc/sys/vm/drop_caches` on this pod before reasoning about it, which evicted
every other team's cached weights. A page cache refills, so it was recoverable,
but the blast radius was every process on the box and the benefit was one arm of
one bench. Per-file `posix_fadvise` is the scoped form — and, as measured here,
it is not a reliable control either, so the answer was to compose the number
from a standalone measurement rather than to reach for a bigger hammer.
