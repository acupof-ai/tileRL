# The `max_pending` cap is not the queue that binds — the device is 5.1x oversubscribed

**Status:** closed
**Date:** 2026-09-06
**Commit:** 962efa6 (probe lands with this entry)
**Card:** H20 card 6

## Context

`max_pending=32` bounds in-flight tier writes. Its drain side was measured on 09-06
(240 MiB/s durable, 1337 ms per 320.6 MiB entry, so 42.8 s for a full queue) and its
arrival side was not: `ssd_offered` is a count, so nothing reported offers/second, and the
[ssd_save_ms entry](2026-09-06-ssd-save-ms-is-page-cache-time.md) left the cap explicitly
unsized — "a bound set without measuring what it bounds."

This measured the arrival side. It found the cap is not the constraint at all, and that
both halves of the comparison I set out to make were wrong.

## What the arrival rate is

12 conversations x 3 turns, served strictly sequentially (`--max-batch 1`), 72 offers,
0 poll gaps. Two poll intervals, to test the instrument:

| | `--interval 0.25` | `--interval 0.05` |
|---|---:|---:|
| samples | 160 | 366 |
| mean offers/s | 0.952 | 0.926 |
| max offers in 1 s | 3 | 3 |
| max offers in 5 s | 12 | 12 |
| max offers in 42.8 s | 54 | 54 |
| `per_window_max_per_s` | 7.96 | **19.54** |
| `ssd_refusals` | 0 | 0 |
| `ssd_pending` peak | 4 | 4 |

## The contradiction, and what it was

Mean arrival 0.95/s against a drain of 0.748/s is 1.27x — sustained for 74 s, that should
have filled a 32-deep queue and started refusing. **Refusals were 0 and `pending` never
exceeded 4.** Two measured numbers cannot both be right, so neither shipped until this
resolved. Both candidates tilerl-27 named turned out to be real, and they compound:

**1. The drain rate was measured on the wrong entry size.** 1337 ms was for a 320.6 MiB
entry. Under this workload the files are bigger: mean `.kv` **435.3 MiB** over 35 surviving
entries, `.st` a constant **149.6 MiB**, so **585.0 MiB per offer — 1.82x** the
microbenchmark's entry. An entries-per-second drain rate cannot be carried between
workloads when the quantity that binds is bytes.

> **Provenance (2026-09-10 cleanup):** the one-off `scripts/bench_h2d.py` was deleted here; rerun it by hand with `TILERL_TARGET=cuda python3 scripts/bench_h2d.py`. The 149.6 MiB size is now derived through precision.nbytes (144 MiB GDN state + 5.6 MiB conv window).

**2. What empties `_pending` is not the device.** `_flush_loop` pops an entry, calls
`torch.save`, and removes it from `_pending` on return — and that return is a page-cache
accept, not a durable write. Measured: `ssd_save_ms` 23611 ms over 144 saves = **164 ms per
save** for 292.5 MiB, i.e. the queue drains at **~1784 MiB/s**. Against 541.5 MiB/s of
arrival that is 0.30x, so the queue empties three times faster than it fills and `pending`
sits at 4 forever. The 240 MiB/s figure describes the device; it was never the queue's
service rate.

Bytes, all three rates over the same 77.8 s span:

| stage | MiB/s | vs arrival |
|---|---:|---:|
| arrival (72 x 585.0 MiB) | **541.5** | 1.00x |
| `_pending` drain (`torch.save` accept) | **1784** | 0.30x |
| device (`/proc/diskstats` vda2 written) | **106.5** | **5.08x oversubscribed** |

Byte accounting closes: `saves` is 144 for 72 offers, exactly 2.0 (a `.kv` and a `.st` per
offer), so nothing was skipped; 35 surviving + 37 evicted = 72; 72 x 585.0 = 42119 MiB
written against 8282 MiB the device counter saw in the window. The gap is writeback still
in flight plus the eviction path deleting entries before their pages were flushed — a file
`unlink`ed while dirty never reaches the platter.

## Verdict: the cap is the wrong knob, and it is not the risk it was thought to be

- **Refusals cannot happen from arrival at this rate.** The cap is 5.9x the deepest queue
  the workload produced (4). Raising or lowering 32 changes nothing observable here.
- **The real oversubscription is 5.08x at the device**, and it is invisible to every
  counter the tier reports, because the tier's own timer stops at the page cache. What
  absorbs it is host RAM: 1928 GiB with `dirty_ratio=20`, so 386 GiB of dirty pages before
  the kernel throttles writers. A full 32-deep queue is 18.3 GiB, **4.7%** of that ceiling.
- **The OOM framing was already dropped as wrong; this says what the cap actually buys:**
  nothing, at this arrival rate. It is not harmful, and it is not load-bearing.
- **What would bind:** sustained arrival above 106.5 MiB/s of *retained* bytes, which is a
  disk-capacity question (`max_bytes`, and the 37 evictions in 72 offers say that path is
  already hot), or dirty-page throttling at 386 GiB, which is 660 entries away. Neither is
  `max_pending`.

**A cap needs a measured arrival rate AND the service rate of the thing it caps.** I had
the second one wrong in the same direction as the published figure: 240 MiB/s is the
device, `_pending` is served by the page cache at 7.4x that, and the queue those two
describe behaves completely differently.

## The burst metric was an artifact of my own instrument

The first version reported max(offers in one poll window)/interval. It read **7.95/s at
0.25 s and 19.50/s at 0.05 s** — both exactly (1 or 2)/interval, on a workload whose mean
interarrival is 1.07 s, **21x the poll window**. A per-poll rate rises without bound as the
poll shrinks and reads as a discovered burst; I reported 7.95 to a peer before catching it.

Replaced by a count in a fixed wall-clock window, which the table above shows invariant
(3/12/54 at both intervals while the artifact moved 2.5x). `per_window_max_per_s` is still
printed, labelled as an artifact, so the next reader sweeping `--interval` watches it move
while the counts hold. The selfcheck asserts the invariance directly: same arrivals, poll
halved, per-poll rate 2 → 4 and the 1 s count stays 1.

**54 was almost published as a required queue depth.** It is offers arriving within one
full-drain time, which assumes zero drain during the window — a bound, not a depth. The
measured depth is 4.

## Two crashes in teardown, two runs lost

Before any of the above, two complete 12-session runs produced zero numbers by crashing
*after* collecting everything:

- `HTTPError` from the request loop propagated instead of breaking. Cause was `--max-ctx
  8192` too small for turn 2; `urllib`'s `str(exc)` is only `HTTP Error 400: Bad Request`,
  so the body naming the real limit was discarded unless `exc.read()` is called.
- `self._stop = threading.Event()` on a `Thread` subclass **shadows `Thread._stop`**, a
  real method `join()` calls. `TypeError: 'Event' object is not callable`, raised only at
  join time, after all 3 turns had finished.

The selfcheck passed both times because it built the sampler with `Sampler.__new__` to test
the arithmetic and never called `start`/`stop`/`join` — the three methods carrying the bug.
It now drives the real lifecycle against a dead port. Verified by reintroducing the shadow
and confirming the control went red *inside `join()`*, not at an earlier assert.

**A probe's measurement is not delivered until teardown returns.** A per-item failure
records its reason and breaks; the samples already collected are the result, and a config
mistake is a finding, not an abort.

## Also caught by asserting rather than assuming

`_sectors()` parses `/proc/diskstats`, which does not exist on this Mac, so the field index
could not be checked by running it. Asserted against a real pod row instead — and the
first assertion I wrote named field 6 as sectors-read when it is field 5. Had I sampled
field 6, the byte rate would have mixed reads into writes and still looked plausible.

## Results

| date | commit | host | arm | value |
|---|---|---|---|---|
| 2026-09-06 | 962efa6 | H20 card 6 | mean arrival, 12 sessions serial | **0.926-0.952 offers/s** |
| 2026-09-06 | 962efa6 | H20 card 6 | max offers in 1 s / 5 s / 42.8 s | **3 / 12 / 54** (both intervals) |
| 2026-09-06 | 962efa6 | H20 card 6 | mean bytes per offer | **585.0 MiB** (435.3 kv + 149.6 st) |
| 2026-09-06 | 962efa6 | H20 card 6 | arrival, bytes | **541.5 MiB/s** |
| 2026-09-06 | 962efa6 | H20 card 6 | `_pending` service rate (page cache) | **~1784 MiB/s** |
| 2026-09-06 | 962efa6 | H20 card 6 | device written, `/proc/diskstats` | **106.5 MiB/s** |
| 2026-09-06 | 962efa6 | H20 card 6 | `ssd_pending` peak / `ssd_refusals` | **4 / 0** |
| 2026-09-06 | 962efa6 | H20 card 6 | per-poll burst artifact, 0.25 s → 0.05 s | 7.96 → **19.54** |

## Limitations

- **Serial, so B=1 throughout.** One request in flight, so every tick is a GEMV and the
  offers/second is the rate a sequential workload produces. Concurrency would raise
  arrival and, per tilerl-48, cross the sm90 dispatch boundaries (mma8 at B≤8, WGMMA16
  above), so a cap sized for concurrent serving needs the probe re-run at B>1.
- **One workload shape.** 12 conversations, 3 growing turns, ~8k tokens by the last turn.
  Entry size is what makes the drain comparison work, and it is workload-specific.
- **The device rate is a whole-disk counter.** `/work` is the only active writer during the
  run (idle baseline 0.027 MiB/s, measured), but it is not attributed per-process.
