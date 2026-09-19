# Background publisher for request-close host/SSD bytes — 2026-09-20

> Status: landed behind an env gate (`TILERL_CLOSE_BATCH_D2H=1`/`TILERL_CLOSE_BG_PUBLISH=1`),
> default OFF; device partial confirmation on V100 2026-09-20 (0041fb14): the
> shared-SSD write leaves the step thread when the bounded queue fits
> (`pub_cold_transfer` 1783→9 ms), correctness/latency all green, but the default
> queue depth 512 overflows a 32k close (34.5% inline) and the private-spill
> read (`ssd_mmap`) is still charged into the close tick. Not ready for a default
> flip.

Second of two PRs on the deterministic per-request close-time lock stall. The
first ([2026-09-19-batched-close-publish-d2h.md](2026-09-19-batched-close-publish-d2h.md),
#741) collapsed the per-page device syncs (close 2940→2321 ms, −21%) but left
the two host/SSD-byte terms flat: `ssd_mmap` 1833→1865 and `pub_cold_transfer`
1726→1783 ms. This PR was meant to move both off the close critical path.

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

## Device confirmation (V100 sm70, 2026-09-20, 0041fb14)

Three served arms, same fill-then-warm protocol as the prior window (W=2048,
1 GiB-RAM / 8 GiB-SSD f16, 37.6k prompts): bg0 `TILERL_CLOSE_BATCH_D2H=1` only
(the 1-PR state, fill-2/warm-2), bg1 `TILERL_CLOSE_BG_PUBLISH=1` with the
default `TILERL_CLOSE_BG_DEPTH=512` (fill-2/warm-2), bg2 the bg flag with
`TILERL_CLOSE_BG_DEPTH=8192` (fill-1/warm-1, kept small for disk headroom).

Per-request close medians over the release tail ticks (ms):

| arm | release_close_request | ssd_mmap | pub_cold_transfer | pub_frame_d2h | bg degraded |
|---|---:|---:|---:|---:|---:|
| bg0 batch only | 2326 | 1919 | 1874 | 43 | — |
| bg1 bg, depth 512 (default) | 2752 | 2258 | 960 | 41 | 34.5% |
| bg2 bg, depth 8192 | 1020 / **6478** (n=2) | 3341 | **9** | 154 | **0%** |

### What worked

- **Shared-SSD write leaves the step thread when the queue fits.** At depth
  8192 `kv_cold_bg_degraded=0` and `pub_cold_transfer` drops 1783→9 ms. The
  physical proof is the size timeline (`bg2-sizes.txt`, bytes in 2³⁰): after a
  publisher returns, `.prefix.bin` keeps growing as the worker commits —
  6.7 → 8.8 → 10.7 GiB across the following requests — i.e. the bytes land off
  the request's wall.
- **No steady regression.** model 165–166 ms, decode tick p50 182–183 ms, raw
  4.6–5.2 tok/s, acceptance unchanged across all three arms.
- **Correctness and cancel latency, all vendored** (not terminal reads): a
  follower repeating the identical 32k prompt returns byte-identical tokens,
  `finish_reason=length`, `/health` `prefix_hits +1`, `kv_cold_drops=0`
  (`bg{1,2}-follower.json`); a client cancel while a follower is in flight frees
  the slot in 0.40–0.81 s and `/health` answers in 1 ms (`bg{1,2}-cancel.json`)
  — the off-lock `wait_committed` design holds, the tick lock is never held on
  the worker. Zero `kv_cold_bg_timeouts` / `bg_failed` all window.

### What did not

- **Default queue depth 512 overflows a 32k close.** One 37.6k close enqueues
  thousands of page jobs (~3100–4700 observed); against a 512-deep bounded
  queue the surplus returns False and transfers **inline**, so bg1 still ran
  34.5% of transfers under the lock and its close median was not lower (2752).
  Depth needs to be derived from the maximum close burst (follow-up).
- **The private-spill read did not cleanly leave the close tick.** Even at
  depth 8192 with zero inline fallback, `ssd_mmap` self-time (~3.3 s median on
  bg2's two tail ticks) is still charged into `release_close_request`, and the
  two bg2 close ticks were 1020 and **6478 ms** — too thin (n=2) and too
  long-tailed to claim "close is sub-second". Only three facts are established:
  (1) `pub_cold_transfer` → 9 ms and the spill grows asynchronously, (2) the
  ColdSsdFile self-measured `ssd_mmap` lands in the close tick, (3) close wall
  was 1.0–6.5 s. Whether that residual time is the device busy or the step
  thread host-blocked (the worker's `share_hold_kv` private-spill lift and
  shared write contend with the close path on `_tlock`) is **not decided**: the
  `fwd_gpu=7457` on the 6478 ms tick is a no-`synchronize` CUDA-event span that
  inflates under any mid-forward blocking wait even with the device idle
  (`sync_streams=0`), so it must not be read as "the GPU was busy 7.3 s". Static
  read for an implicit CUDA call on the worker path is assigned to fixkv; if
  inconclusive, a non-blocking event-query busy/idle probe plus worker-thread
  mmap timing (not drained into the step timer) folds into the next device
  window alongside the depth fix.
- **Backlogged follower wait.** When the worker is behind, a follower spends
  ~4.9 s off-lock in `wait_committed` before adopting (`bg2-follower.json`). It
  holds no tick lock, but the interaction latency is real.

Verdict: the mechanism is correct and improves cancel/interaction latency, but
it stays **default OFF**; flipping needs the queue-depth fix and a follow-up
that takes the worker's private-spill lift fully off the close path.

Vendored in `wins/bg-publish-device-2026-09-20/`: `bg0/bg1/bg2.json` (arm
summaries), `bg{1,2}-follower.json` and `bg{1,2}-cancel.json` (response +
health snapshots), `bg1/bg2-sizes.txt` (physical-size timelines, 2³⁰ units).
Raw per-tick logs are `~/tilerl-logs/serve-bg{0,1,2}.boot` on the box.

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

- Derive `TILERL_CLOSE_BG_DEPTH` from the maximum close burst (default 512
  overflows a 32k close; 8192 fit with 0 inline in this window) — small,
  statically testable.
- Take the worker's private-spill lift (`share_hold_kv` off `sparse_cold.bin`)
  fully off the close path; determine device-busy vs host-blocked with the
  non-blocking event-query + worker-thread mmap probe in the next device window.
- A default flip only after both land and a served soak.
- Pinned-host-buffer pooling with a cold-tier release hook (the #741 leftover;
  orthogonal to this PR).

## Results

| date | commit | machine | target | model | close terms | steady decode |
|---|---|---|---|---|---|---|
| 2026-09-20 | pending PR | CPU (hermetic) | background close-publish mechanism + wiring | — | ssd_mmap/pub_cold_transfer move off the close segment (host-only on CPU; timing n/a) | unchanged |
| 2026-09-20 | 0041fb14 | V100 sm70 | request-close host/SSD bytes | Qwen3.8-27B-NVFP4, 37.6k sparse, 1G/8G f16 | depth8192: pub_cold_transfer 1783→9 ms + spill grows async, 0 inline; ssd_mmap ~3.3 s still in close tick, close 1020/6478 ms (n=2); default depth512 degrades 34.5% | held: model 165–166 ms, tick 182–183 ms, raw 4.6–5.2 tok/s |

Raw artifacts: `tests/test_sparse_kv_tier.py`, `tests/test_sparse_engine.py`;
changes `src/tilerl/kv_tiers.py`, `src/tilerl/kv_cache.py`,
`src/tilerl/sparse_runtime.py`, `src/tilerl/engine.py`.
