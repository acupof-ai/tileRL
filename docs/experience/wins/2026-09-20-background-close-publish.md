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
- **Follower waits OFF the tick lock, bounded, then misses.** The engine is a
  single step thread but `submit`/`poll`/`shutdown` share its `self._lock`; a
  wait taken inside `_admit` (which runs under the tick lock in `step`) would
  freeze all of them for up to the wait budget and re-pay the moved bytes on
  the admit path. So the wait is split out: `step` calls
  `_await_waiting_publishes` BEFORE taking the tick lock — a short locked
  snapshot of the waiting sparse heads, then lock release, a read-only
  `peek_hit` per head (no hits++/LRU side effect), and one
  `cold.wait_committed(keys)` blocking on the worker with no engine lock held.
  Back under the tick lock, `_admit` only does a non-blocking
  `has_all_keys(keys)` commit probe; a timeout, a failed/abandoned publish, or
  a release that lands in between makes the entry None and the follower does a
  full prefill miss, never a raise on a missing blob. `share_take` /
  `share_take_field` are non-blocking (None while queued); `wait_committed` is
  the only call that blocks on the worker and no forward/admit path calls it
  under the engine lock.
- **Concurrency in the wait window.** While a follower waits, the only thread
  mutating the cold shared namespace is the worker committing jobs; another
  tick cannot run (the step thread has not taken its lock) and submit/cancel
  only touch queues. The queued private blob is pinned against the publisher's
  own `cold.forget` and the RAM-budget evictor (re-parked, not spilled/dropped).
  After the wait, the locked `has_all_keys` re-check closes the race with a
  release; adopting a committed key adds no content-key store ref (`resolve`
  takes a read reference and drops the page→key link, the store entry keeps
  its ref), so an adopted follower cannot be evicted under itself.
- **Source lifetime.** A queued host/SSD page's private blob is protected from
  the publisher's own `cold.forget` (the key stays in `_pub_private` until the
  worker consumes it) and from the RAM-budget eviction loop (re-parked, not
  spilled/dropped). Device bytes are never the async source: the frame blob was
  produced by #741's batch before `free_block`.
- **Lock order.** One new `RLock` inside the tier; the step thread and worker
  take it only for short sections. The engine never holds its tick lock while
  waiting on a future (the wait is an explicit pre-lock phase of `step`), and
  the worker takes only the tier lock — engine-lock → tier-lock stays one-way.
- **Shutdown join.** `engine.shutdown` drains the queue and joins the worker
  before `prefix.clear()` releases refs; a crash with jobs queued is
  indistinguishable from the prefix never having been published (the shared
  cache is non-durable), so it is a miss, not wrong data.

Setting `TILERL_CLOSE_BG_PUBLISH=1` also enables the #741 batch context: the
background handoff needs the same prepare/defer split (device pages must be
host before `free_block`). Default (both gates unset) is byte-for-byte the old
inline path; the worker thread is not started.

## Gates (CPU, hermetic)

- Full CPU suite **1025 passed / 22 skipped / 1 xfailed** with the changed tier
  (mechanism both enabled and default-off).
- Tier mechanism (`tests/test_sparse_kv_tier.py`, 10 new): RAM page future
  resolves (share_take non-blocking until commit) and the private copy is gone;
  frame `offer_hold` path; private-SSD lift through the worker; queue-full →
  False + inline source untouched; worker failure fires a miss (no hang); refs
  landing before commit fold correctly in both signs; `forget` while queued
  keeps the source; the RAM-budget evictor re-parks (does not drop) a queued
  source; `close()` drains and joins; default-off starts no worker.
- Engine e2e (`tests/test_sparse_engine.py`, 4 new): with both gates on, a
  fully-resident short prompt closes all pages through the background queue
  (`kv_cold_bg_queued ≥ 5`), a same-prompt follower admitted on the NEXT step
  (whose pre-lock wait blocks on the futures) HITS the full 24-page prefix and
  decodes exactly the prefix-miss oracle tokens; a worker parked past
  `TILERL_CLOSE_BG_WAIT_S` makes a follower take a full miss
  (`sparse_matched == 0`, `kv_cold_bg_timeouts ≥ 1`) rather than raising on a
  missing blob; while the follower is parked in the off-lock pre-wait a peer
  `cancel` (same `self._lock`) returns immediately (mutation-red on moving the
  wait back under the tick lock); and `engine.shutdown` drains/commits every
  in-flight publish before `prefix.clear` runs (mutation-red on removing the
  stop call).

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
on one bounded single consumer, and let a same-tick reader wait on per-key
events in a dedicated phase that holds NONE of the critical locks (short
snapshot → release → wait → re-check under the lock), with the wait timing out
into a cache miss — never move host bytes under the device-tick lock, never
wait on the worker while holding it, and never leave a lookup entry naming a
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
