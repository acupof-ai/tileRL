# Background publisher for request-close host/SSD bytes — 2026-09-20

> Status: landed behind an env gate (`TILERL_CLOSE_BG_PUBLISH=1`), default OFF,
> CPU hermetic gates green; device delta pending-remote (V100 stopped 2026-09-16,
> ckl physical wall). Second of two PRs on the deterministic per-request
> close-time lock stall. The first ([2026-09-19-batched-close-publish-d2h.md](2026-09-19-batched-close-publish-d2h.md),
> #741) collapsed the per-page device syncs (close 2940→2321 ms, −21%) but left
> the two host/SSD-byte terms flat: `ssd_mmap` 1833→1865 and `pub_cold_transfer`
> 1726→1783 ms. This PR moves both off the close critical path.

## Context

After #741, a publisher close still did, inside `engine._lock`, the physical
private→shared rehome of every close page whose bytes were already off device:

- a RAM private blob popped into the shared namespace (and possibly LRU-spilled
  to `.prefix.bin` under the pinned-budget loop);
- a private-SSD page read off the private spill file and written to the prefix
  spill file (two disk IO passes + mmap);
- a synced frame blob committed to the shared namespace (`pub_share_hold`).

These are ~3.5 s of the post-#741 2.32 s median close (the terms overlap and
include unmarked index work). None of them touches a live device frame: #741
already snapshotted the device-resident pages to host under one sync before any
frame was freed. So every byte the close still moves is a HOST byte and can move
after `close_request` returns, provided a follower that adopts the meanwhile
visible prefix entry can wait for the bytes instead of raising.

## What Worked

One bounded single-consumer thread owned by `HostKvPages`
(`TILERL_CLOSE_BG_PUBLISH=1`), and a reservation/future split:

- **Reserve synchronously, transfer asynchronously.** `close_request` has
  already put the entry on the lookup chains. The close loop now calls
  `cold.offer_publish(private_key, content_key, extra)` (host/SSD source) or
  `cold.offer_hold(content_key, blob, n)` (the #741-synced frame blob); the
  tier reserves the content key (`_pub_pending` + a per-key `threading.Event`)
  and enqueues one job, atomically under the tier lock. The worker later runs
  the existing `share_hold_kv`/`share_hold` unchanged.
- **Bounded, never drops.** The queue is `queue.Queue(maxsize=$TILERL_CLOSE_BG_DEPTH`,
  default 512 pages). A full queue makes `offer_*` return False and the close
  loop runs that one transfer inline (a `kv_cold_bg_degraded` count), so queue
  memory is bounded and no publish is lost. Reservation and enqueue are one
  locked step: a key cannot be reserved without its job queued (no permanent
  hang from a dangling entry).
- **Follower waits, bounded, then misses.** `share_keys()` includes pending
  keys so the freeze-ref pass in `publish_dropped` sees them; `share_ref` /
  `share_release` landing before commit accumulate as a signed delta folded
  into the record's starting refcount 1 (a −1 entry evicted pre-commit
  releases the record the moment it commits). Before adopting, engine `_admit`
  calls `cold.wait_ready(keys, $TILERL_CLOSE_BG_WAIT_S`, default 30 s): a
  follower blocks off the engine lock until every page commits; a timeout or a
  worker transfer failure abandons the key (event fires, no record) and the
  follower adopts nothing — a full prefill cache miss, never a raise.
- **Source lifetime.** A queued host/SSD page's private blob is protected from
  the publisher's own `cold.forget` (the key stays in `_pub_private` until the
  worker consumes it) and from the RAM-budget eviction loop (re-parked, not
  spilled/dropped). Device bytes are never the async source: the frame blob was
  produced by #741's batch before `free_block`.
- **Lock order.** One new `RLock` inside the tier; the step thread and worker
  take it only for short sections. The engine never holds its lock while
  waiting on a future (the wait is in `_admit` between locks), so the order is
  engine-lock → tier-lock one way; the worker takes only the tier lock.
- **Shutdown join.** `engine.shutdown` drains the queue and joins the worker
  before `prefix.clear()` releases refs; a crash with jobs queued is
  indistinguishable from the prefix never having been published (the shared
  cache is non-durable), so it is a miss, not wrong data.

Setting `TILERL_CLOSE_BG_PUBLISH=1` also enables the #741 batch context: the
background handoff needs the same prepare/defer split (device pages must be
host before `free_block`). Default (both gates unset) is byte-for-byte the old
inline path; the worker thread is not started.

## Gates (CPU, hermetic)

- Full CPU suite **1021 passed / 22 skipped / 1 xfailed** with the changed tier
  (mechanism both enabled and default-off).
- Tier mechanism (`tests/test_sparse_kv_tier.py`, 9 new): RAM page future
  resolves and the private copy is gone; frame `offer_hold` path; private-SSD
  lift through the worker; queue-full → False + inline source untouched;
  worker failure fires a miss (no hang); refs landing before commit fold
  correctly in both signs; `forget` while queued keeps the source; `close()`
  drains and joins; default-off starts no worker.
- Engine e2e (`tests/test_sparse_engine.py`, 2 new): with both gates on, a
  fully-resident short prompt closes all pages through the background queue
  (`kv_cold_bg_queued ≥ 5`), a same-prompt follower admitted next tick blocks
  on the futures, HITS the full 24-page prefix, and decodes exactly the
  prefix-miss oracle tokens; a worker parked past `TILERL_CLOSE_BG_WAIT_S`
  makes a follower take a full miss (`sparse_matched == 0`,
  `kv_cold_bg_timeouts ≥ 1`) rather than raising on a missing blob.

## Device target (pending-remote)

The terms this must move, from the V100 2026-09-19 #741 measurement
(`659c2fbb`, medians over the release tail): `ssd_mmap` ~1.86 s and
`pub_cold_transfer` ~1.78 s of the ~2.32 s close. Expected effect: both leave
the `release_close_request` segment (paid later on the worker, overlapping
forward ticks); close should approach the remaining index/frame work. Steady
decode must not regress (model ~166 ms/tick, eff ~9.4 tok/s in #741): the
worker moves only host bytes under a short tier lock and never touches CUDA.
Re-measure with the #732 five-subsegment parse on the next V100 window before
any default flip; keep the gate default-off until that number is captured.

## Rule

When a lock-critical boundary must PUBLISH bytes that are already off the
device, reserve the key and enqueue the byte move atomically, move the bytes
on one bounded single consumer, and let a same-tick reader block on a per-key
event off the lock with a timeout that degrades to a cache miss — never move
host bytes under the device-tick lock, and never leave a lookup entry naming a
key with neither a blob nor a future.

## Follow-up

- Device re-measurement (V100) and, if green, a default flip after a served
  soak.
- Pinned-host-buffer pooling with a cold-tier release hook (the #741 leftover;
  orthogonal to this PR).

## Results

| date | commit | machine | target | model | close terms | steady decode |
|---|---|---|---|---|---|---|
| 2026-09-20 | pending PR | CPU (hermetic) | background close-publish mechanism + wiring | — | ssd_mmap/pub_cold_transfer move off the close segment (host-only on CPU; timing n/a) | unchanged |
| next V100 window | pending-remote | V100 sm70 | request-close host/SSD bytes | Qwen3.8-27B-NVFP4, 1G/8G f16 | target: ssd_mmap ~1.86 s + pub_cold_transfer ~1.78 s off-lock | must hold ~166 ms/tick, ~9.4 tok/s |

Raw artifacts: `tests/test_sparse_kv_tier.py`, `tests/test_sparse_engine.py`;
changes `src/tilerl/kv_tiers.py`, `src/tilerl/kv_cache.py`,
`src/tilerl/sparse_runtime.py`, `src/tilerl/engine.py`.
