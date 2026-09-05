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

`resident()` is two conditions, and only the `_lru` one was tested. An entry sits in
`_pending` from the enqueue until the daemon's `torch.save` returns — ~100 ms, and the whole
reason the save is off-tick — and in that window there is no file, so both loads have a
pending-table branch and `resident()` has the second half of an `or` for it. Deleting that
half left all 70 tests passing. Getting it wrong is not a crash: the lookup walks past a
prefix that is in memory and re-prefills, silently, in exactly the window a burst of
same-prefix requests lands in. Now gated by blocking `torch.save`, asserting no file exists
yet, and then asserting the probe and both loads serve from memory — three assertions with
three separate negative controls (drop the `or` half, drop `load_kv`'s pending read, drop
`load_state`'s; each fails only its own).

Wired through `--ssd-path`, keyed on `_weight_fingerprint(cfg)`: **every**
dataclass field, not a hand-picked list. The first draft picked seven and named
`cfg.num_heads`, which does not exist on this config — the field is
`num_attention_heads` — so the list was already wrong when written, and a field
left out is how a restart serves KV computed under other weights.

**The fingerprint control asserted the safe half and missed the disk half.** It read
`other_tier.recovered == 0 and other.lookup(toks) is None`, which is one claim written as
two: a mismatch sets `prev = None`, so nothing is indexed, so `resident()` is False, so
`_fault_in` is never called — the `lookup` half cannot fail while the first half holds.
What the pair never touched is that a mismatch must **unlink**. Deleting the unlink loop
from `_recover` leaves all 327 tests passing (measured). That matters because this unlink is
the tier's ONLY disk reclamation: `invalidate()` deliberately writes one marker instead of
walking a 20 GiB directory inside a training step, so each generation's spill sits on disk
until some later `_recover` mismatches and removes it. Now asserted on the directory listing
rather than on `recovered`, since `recovered` is 0 either way. Negative control: the same
mutation fails it with "left 2 spill files on disk".

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
  boundaries keeps them in HBM and skips the disk: 1624 MB → 325 MB. **Gated only after
  the fact:** dropping the kwarg left all 328 tests passing, so the entire difference between
  8.96% and 45.3% was a one-word regression no local run would catch — the number is measured
  on a machine this suite never runs on. `test_an_intermediate_chunk_publish_stays_out_of_the_disk_tier`
  asserts `prefix_published == 5, ssd_offered == 1` at a 48-token budget over 240 tokens;
  without the kwarg it is 5 and 5, the same 5.0x. Both operands are asserted because the
  warm-up length decides whether either is even reachable: a ragged one never fires the
  prompt-complete publish, so the expected offer count is 0 and "only the last spilled"
  cannot be told from "nothing spilled".
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

**A `drop()` inside that same window is handled by two guards, and neither had a test.**
`drop()` clears the pending tables and unlinks, but the daemon is already past that point
holding the blob: its `torch.save` completes *after* the drop and rewrites the file the drop
removed. Two separate lines catch it, and they cost different things:

| mutation | `ssd_entries` | files left | consequence |
|---|---:|---:|---|
| `still_pending = table.get(k) is blob` forced True | **1** | 1 | the dropped prefix is indexed, so a lookup can serve it — after `invalidate()`, KV from the previous weights |
| `os.remove(dst)` after the save deleted | 0 | 1 | one unpaired `.kv`; the next `_recover` drops it because it adopts by pair — a leak until restart, not stale service |
| both | 1 | 1 | as row 1 |

**I first reported row 1's numbers as row 2's**, because the mutation I ran replaced the whole
`if still_pending: ... else: remove` block rather than only the rollback, so it removed both
guards at once and I attributed the resurrection to the unlink. Codex caught it on review of
#132. The fix is one assertion per guard, **index first** — pytest stops at the first failure,
and both mutations leave a file, so asserting the file first makes the index assertion
unreachable in every arm. That ordering is exactly what let the wrong claim stand: the
assertion whose message said "indexed again" was never executed in the run I quoted.

**The gate's own first version was inert, and its negative control passed.** It waited with
`_flushed`, which polls the pending tables — and `drop()` had just emptied them, so the helper
returned while the daemon was still inside `torch.save`. The second attempt waited on
`_q.unfinished_tasks`, which never reaches 0 because nothing calls `task_done()`; that one
worked but spent 6.46 s in a timeout, and a wait condition that cannot change is the same
defect as one already satisfied. What works is waiting on the save actually in flight (one
`torch.save`, not two: the state half is skipped at the `table.get(k) is not blob` check
without ever calling save) and then polling the directory. 0.01 s, control red.

**One mutation that survived is not a gap — it is redundant code.** The eviction victim search
skips keys still in the pending tables, and deleting that skip changes no test. It is not
untested: the skip *does* filter, 6 of 12 candidates in a forced-eviction probe (a key enters
`_lru` when its first half lands, so it is indexed and pending at once). But orphan files,
half pairs and unaccounted bytes all came out identical with and without it, because the
`table.get(k) is blob` re-check plus the rollback above already cover the same race one layer
down. Left in place; recorded so the next person does not spend the same hour writing a gate
for a condition whose removal has no observable effect. The distinction is worth the two
probes it took: "the suite stays green" means either a missing gate or dead weight, and only
comparing the observable state tells you which.

What is NOT claimed: no fsync on the publish path, so a host power loss can lose an entry
the daemon believes it wrote. The tier is a cache — every entry is reconstructible by
prefill — so durability past process death was not bought.

Also not covered, deliberately: the hop from `args.<flag>` to `_build_engine` inside `serve`.
Replacing `ssd_min_tokens=args.ssd_min_tokens` with a constant leaves all 330 tests passing,
and the same is true of the seven other arguments in that call and of all 70 `add_argument`
entries — it is one direct-pass call site, so covering it per-argument is 70 assertions for one
line of plumbing. The proportionate gate is one end-to-end CLI start that reads the values back
off `/health`, which needs a server process and is not in this change. Recorded rather than
quietly left: the flags are covered from `_build_engine` inward (both, with separate controls
for "did not arrive" and "arrived and does nothing"), and uncovered from argparse to that call.

## Rule

A restart is the disk tier's whole case: unlike a host-memory tier it needs no
concurrent sessions, because an empty HBM makes every lookup reach back.

And before reporting a storage number, divide the bytes by the device's measured
bandwidth. If the wall clock is smaller, the device was not involved — no
arm-order control detects this, and three of them passed while it was happening.

**Three of this change's four gates each covered one branch of a two-branch fact, and each
was found by mutating rather than by reading.** The truncation test reached `load_state` only
(`_fault_in` returns on its miss, so `load_kv`'s guard was deletable); the fingerprint test
asserted the mismatch does not serve but never that it unlinks, which is the tier's only disk
reclamation; the `resident()` probe was tested on `_lru` and not on `_pending`, the state
every entry is in for its first ~100 ms. All three passed the whole suite with the untested
branch deleted. The generalization is not "test both operands of an `or`": these were an
early `return`, a side effect next to a return value, and an `or` — what they share is that
one fact needs two assertions and a plausible test satisfies one. Delete each branch and
re-run; a suite that stays green names the gap. And assert a counter only the second path
increments — "it did not raise" is also satisfied by never getting there.

**A mutation must be as narrow as the claim it supports, and the assertion must actually
execute.** Writing one gate for two guards, I replaced the whole `if still_pending: ... else:
remove` block, saw the resurrection, and published it as the cost of deleting the unlink — two
guards' worth of damage attributed to one line. Codex caught it. The mechanism that hid it is
assertion order: both mutations leave a file on disk, the file assertion came first, and pytest
stops there, so the assertion whose message said "indexed again" never ran in the run I quoted.
So: mutate one line per claim, order the assertions so the one carrying the claim is reached
first, and read which assertion actually failed rather than that the test went red.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-05 | eae658e | H20 card 6 | sm90 | Qwen3.8-27B-NVFP4 | 0.619 (restart hit) | — | — |
| 2026-09-05 | eae658e | H20 card 6 | sm90 | Qwen3.8-27B-NVFP4 | 1.114 (cold, no tier) | — | — |

Raw artifacts: `/work/ssdr6.log` (three arms + both scenarios), `/work/ssdr{3,4,5}.log`, `/work/ssdbw2.log`
(bandwidth, three ways), `/work/ssd_restart_{cold,faulted,control}.log` (server
logs, compile counts).
