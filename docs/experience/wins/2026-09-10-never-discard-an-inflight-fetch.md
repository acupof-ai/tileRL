# Never discard an in-flight SSD fetch: a deadline-missed read parks and serves the next request — sm90, 2026-09-10

> Status: Shipped

## Context

The 27B measurement ([errors/2026-09-10-the-27b-snapshot-fetch-exceeds-the-prefetch-deadline](../errors/2026-09-10-the-27b-snapshot-fetch-exceeds-the-prefetch-deadline.md))
found the prefetch structurally failing on 27B sm90: the 157.6 MiB snapshot
takes 80–117 ms to load but the seed-rate deadline is 75 ms
(192 tokens / 2558.6 tok/s), so every fetch missed its deadline and
`_build_plan` called `abandon_prefetch()`, which discarded the finished read.
Two costs followed: the fetch's bytes were thrown away after the load was
paid for, and the next same-prefix request queued a fresh 80–117 ms read; on
27B a later tick faulted in the dropped pair **on the calling thread** — a
blocking ~80 ms `torch.load` under the step lock.

Workload: warm 128-token prompt spills to SSD; a cold engine recovers it and
submits the 192-token extended prompt; after it drains, a second request asks
for the exact 128-token prefix with its HBM entry evicted, so only the SSD
pair can serve it. Full 27B-NVFP4, SSD on, H20 card 0,
`scripts/probe_spin_cost_27b.py`.

## What Worked

The deadline decides whether a row **waits**, never whether finished work is
discarded.

- **`KvTier`**: removed `_abandoned` / `discard_fetch`; a completed read always
  parks into `_fetches`. The existing `lookup → take → _fault_in` path already
  serves parked pairs, so no serving path changed. A 2-entry FIFO
  (`_park_keys`) bounds parked buffers that nothing follows: eviction sheds
  only the warm host copy, and the entry stays on disk, so a later lookup
  faults it in like any resident entry — nothing is lost.
- **`Engine._build_plan`**: on deadline expiry the row clears its deadline and
  admits with a full prefill; the fetch keeps reading in the background.
- **Engine spin**: narrowed from the tier-global `any_fetching()` to a waiting
  row whose **own** fetch is in flight with a **live** deadline, bounded by
  that row's remaining deadline (50 ms cap). An unrelated fetch — or one whose
  requester already recomputed past its deadline — no longer makes every tick
  burn the bound; this closes the global-`any_fetching()` remainder carried in
  the spin wins entry.

A/B on the same instrument (the fetch genuinely misses the 75 ms deadline in
both arms; only the code differs):

| signal | before 6bf298a1 | after this fix |
|---|---:|---:|
| finished fetch discarded (`ssd_fetch_drops`) | 1 | **0** |
| request 1 tick-side `torch.load` | 1 (blocking ~80 ms under the step lock) | **0** |
| request 2 new background prefetches | 1 (fresh 80–117 ms re-read) | **0** (deduped against the parked pair) |
| parked pair present at request-2 submit | False | **True** |
| request 2 SSD hit | +1, re-fetched, 0.61 s | **+1, from parked memory, 0.51 s** |

The first request that misses the deadline still full-prefills: 80–117 ms is
genuinely longer than the 75 ms a 192-token prefill costs, so waiting would
lose. The win is that the load is paid **once**, off the tick, and the load's
owner becomes the next request — the restart / resend case the SSD tier exists
for. The probe's verdict is timing-independent (prefetch-count dedup and
parked membership), because a warm page cache can let a fresh re-fetch win by
jitter and make "hit + 0 tick reads" pass on the pre-fix code.

## Rule

A deadline gates whether to WAIT for a result, not whether to KEEP it.
Discarding an already-finished async read makes the producer pay and throws
away the product; the next identical consumer re-pays, and on a slow snapshot
that re-pay lands on the tick. Let the read finish and park — the cache is the
read's owner after the requester leaves.

## Results

| date | commit | machine | target | model | snapshot | fetch ms | deadline ms | req2 prefetch added | req2 tick reads | req2 wall |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-09-10 | 6bf298a1 (before) | H20 card 0 | sm90 | 27B-NVFP4 | 157.6 MiB | 81 | 75.0 | 1 | 0 | 0.61 s |
| 2026-09-10 | this fix (after) | H20 card 0 | sm90 | 27B-NVFP4 | 157.6 MiB | 80 | 75.0 | 0 | 0 | 0.51 s |

Raw artifacts: `/work/spinbefore.log`, `/work/spinafter.log` on the pod.
