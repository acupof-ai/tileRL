# The SSD tier's benchmark never touched the SSD — 2026-09-05

> Status: instrument fixed; the number it produced is withdrawn

## Context

The SSD prefix tier's claim is that after a restart HBM is empty, so a returning
conversation's first turn faults its prefix off disk instead of re-prefilling.
The bench for it is three server starts on H20 card 6, one measured request each:

| arm | spill dir | wall_s | compiles | ssd_hits |
|---|---|---:|---:|---:|
| cold | empty | 3.019 | 0 | 0 |
| faulted | what cold just wrote | **1.168** | 0 | 1 |
| control | a different empty dir | 3.132 | 0 | 0 |

Every control passed. The empty-tier control landed at **0.964x** of cold, so arm
order was not moving the clock. Compiles were 0 in all three windows. The faulted
arm took exactly one SSD hit and recovered 6 entries. `faulted/cold = 2.585x`.

Then I measured the device the tier writes to: **198.9 MiB/s**, reading the
bench's own 1869.5 MiB of spill files.

One entry is 309.6 MiB (156.9 MB of GDN state + 167.8 MB of KV). At 198.9 MiB/s
that read takes **1556 ms**. The whole faulted arm — HTTP, template, tokenize,
the fault-in, the remaining prefill, 8 decode steps — took **1168 ms**.

**The arm was faster than reading its own bytes off the disk, so it never read
them off the disk.**

## Root Cause

The bench creates the condition it is trying to avoid. The cold arm **writes** the
spill; the faulted arm starts seconds later and reads it. A file just written is
in the host page cache by construction, and write-through guarantees every byte
of it is — that is what write-through means. The host has 795 GB of buff/cache
against a 1.9 GB spill, so nothing evicts it.

So `2.585x` is the speedup of faulting a prefix in **from memory**. It is a real
number about a real code path — the read path did run, `ssd_hits` was 1 and the
KV came back correct — but it is not the disk tier's number, and the disk tier is
what the design document claims.

The three controls I did build were all pointed at the *other* confound. Arm
order, JIT compiles, an empty control directory: every one of them asks "is this
speedup the tier or the schedule?" and none of them asks "is this tier the disk
or the cache?" I checked the axis I had already been burned on today — the
09-05 self-hit entry is the cold/warm arms sharing a process — and the new axis
went unexamined until an unrelated bandwidth probe contradicted it.

The tell was there in the first run's own row and I read past it:
`ms_per_prompt_token: 0.399` for the faulted arm. Reading a 2560-token prefix
from a 198.9 MiB/s device cannot cost 0.4 ms per token no matter what the tier
does.

## Fix

`_evict_cache` in `scripts/bench_ssd_restart.py`, called between the cold and
faulted arms: `posix_fadvise(POSIX_FADV_DONTNEED)` per spill file.

Per file, deliberately, **not** `/proc/sys/vm/drop_caches`. I wrote `3` to
drop_caches on this pod before reasoning about it, which evicted every other
team's cached weights on a shared host. It was recoverable — a page cache refills
— but it was mine to not do, and the per-file call is both correct and scoped.

Two checks added, both from the review that asked for them:

* **the win is against the control, not against cold.** `control` carries
  whatever start-order effect survives, so `faulted/control` is the tier's number
  and `faulted/cold` is an upper bound on it.
* **a ceiling, computed before the verdict.** A hit can save at most the prefill
  of the tokens it covered, at the cold arm's own per-token rate. The matched
  length is read off the spill size ladder — the largest entry is the whole
  prompt and cannot serve, since `_match_prefix` treats a full-length hit as a
  miss, so the servable one is the second largest: 2560 of 2729 tokens. The
  ceiling is 2560 × 1.106 ms = **2.831 s**, and the 1.851 s saved is 65.4% of it.
  Within the bound, which is the one thing the invalidated run did establish.

## Rule

A cache benchmark that writes its own fixture has already invalidated itself.
The write puts the bytes in every layer above the one under test, so "read it
back" measures the topmost layer that still holds them — and every arm-order
control in the world will pass while it does.

Before trusting a storage number, divide the bytes by the device's measured
bandwidth and compare with the wall clock. If the wall clock is smaller, the
device was not involved. That one division is cheaper than the three controls I
did build, and it is the only one that would have caught this.
