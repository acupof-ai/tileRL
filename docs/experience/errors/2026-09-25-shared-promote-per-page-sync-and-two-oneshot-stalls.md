# Three distinct 2-6 s decode stalls after the 8 GB cold tier removed SSD reads — 2026-09-25

> Status: the code defect is fixed (#832) and device-verified 2026-09-25; the
> other two are the CUDA allocator and request-finish teardown, measured and
> left (rows in OPEN.md). After `--kv-cold-bytes` 1 GB → 8 GB every `ssd_mmap`
> tick was 0, but 3 decode ticks still crossed 2 s and the second of three
> identical 37.6k requests had eight 1.5–1.9 s refresh ticks. They are three
> different mechanisms; lumping them as "cold tier" would fix none.

Data: V100 sm70, ThinkingCap 27B, W1024/R32/q1/d1, three back-to-back
streamed runs of serve805 prompt 0 (37.6k → 512 tokens), `TILERL_STEP_TIMING=1`
per-tick log (908 ticks, 847 decode). Run 1 is a cold miss; run 2 is a full
prefix-prefix hit (slow one); run 3 is the same hit fully warm.

## Class A (dominant): per-page sync in shared_promote — 8 refresh ticks, 1.5–1.9 s

Run 2's eager refresh ticks (8 of 10) took **model 1472–1725 ms vs 173–235 ms
for the same refresh positions in runs 1 and 3**, with `ssd_mmap=0`,
`offers_pub` only 3–122 ms, `sparse_finalize` ≤43 ms, identical geometry
(`own_w=64 table_w=193`) and the same `offers_pages` (matched pairs: 78, 120,
198, 270, 143, 186 — run1/run3 model 178–235, run2 1472–1725). So neither
selection, demotion, SSD nor the finalize D2H explains it; the cost is inside
the forward envelope on the promote side.

A prefix hit resolves its selected pages through `shared_promote`
(`kv_cache.py`), which copies the pinned shared blob K/V into a fresh private
block and then called `torch.cuda.synchronize()` **unconditionally per page**.
A refresh names up to ~200 shared pages; each non-blocking H2D launch was
forced to drain before the next — ~200 full-stream stalls. `promote_keyed`
(the private-cold path) had already been batched under `pool.promotions()`
(one sync at context exit), but `shared_promote` never looked at the batching
flag, so the same fix that killed the 6.6x private-cold slowdown was absent on
the prefix path. Run 3 is fast because the adopted pages are now this
request's private pinned copies/pins; the churn happens exactly once, on the
first hit.

Fix: `shared_promote` honours `_promote_batching` — non-blocking launches, one
sync at the `promotions()` exit; the per-call sync stays outside that context
(the blob is reusable on return). Gate
`test_batched_shared_promotions_sync_once_not_per_page`: three shared promotes
inside `promotions()` = one sync and byte-equal copies; outside = one sync per
call. Fails on the old code.

## Class B: CUDA allocator reclaim — tick 76, 2.97 s, once

The first decode graph tick after the 8 s prefill (`total=2970ms`,
`graph=2947`, `fwd_host==fwd_gpu`) was `why=alloc_reclaim`: `seg+=6 seg-=23`,
`sync_streams=1`, `num_sync_all_streams` fired, free only 954 MiB. The prefill
tick grew 23 segments the steady decode graph never needed; its first replay
forced the caching allocator to reclaim/coalesce them before 6 decode-shaped
allocations fit. One-time per service, not repeated per request. This is the
known sm70 allocator-at-low-free behaviour (see
2026-09-15-sm70-long-step-tick-holds-engine-lock). A capture-time reservation
is the right lever but is not free: the reclaim happens because the steady
graph's working set is genuinely smaller than the prefill's, so holding the
prefill peak costs VRAM the 31 GB card has little of; left as-is pending a
measured reservation proposal.

## Class C: request-finish prefix publish — tick 352, 2.65 s, once per request

The run-1 request's **last** decode tick carried
`pub_frame_d2h=1035 pub_draft_clone=434 pub_bounds_d2h=296
release_cold_forget=2610` while `graph=2643`. This is `publish_at_finish`
publishing the prompt prefix synchronously on the response's final tick
(#796): the still-device-resident own-window pages are snapshotted D2H so
followers hit. It is the same publish work, just charged to the finish tick;
the following run-2 prefill additionally paid `plan=3656ms` (the
hit-admission re-forward plus churn). It happens once per unique prompt and is
what makes the prefix cache exist; the lever is moving finish-publish off the
response path (background), a separate change. Run 2/3 finish ticks were
14/4 ms (prefix already published).

## What dominates

For a repeat-prompt workload the dominant cost is **class A** (8 ticks ×
~1.5 s ≈ 12 s per first hit) and it is a real defect; B and C are one-shot
2.6–3.0 s events. Discriminator that separated all three: `ssd_mmap` (0
throughout), the phase fields (`graph` vs `model` vs `pub_*`/
`release_cold_forget`), and matched `offers_pages` across runs with divergent
`model` ms.

## Device verification of #832 (2026-09-25)

Same service/geometry, fix deployed to a separate tree, one miss followed by
three identical 37.6k hits (1108 decode ticks, `ssd_mmap` still 0). The first
hit's eight refresh ticks, matched one-for-one to the baseline positions
(same offers_pages 45–270):

| offers_pages | baseline total | fixed total | baseline model | fixed model |
|---:|---:|---:|---:|---:|
| 123 | 1740 | 216 | 1513 | 189 |
| 78 | 1623 | 202 | 1520 | 187 |
| 120 | 1583 | 197 | 1472 | 182 |
| 45 | 327 | 190 | 304 | 176 |
| 198 | 1852 | 223 | 1722 | 206 |
| 270 | 1770 | 213 | 1634 | 195 |
| 143 | 1847 | 219 | 1725 | 203 |
| 186 | 1770 | 215 | 1635 | 192 |

First-hit refresh **p50 1755 → 214 ms, max 1852 → 223 ms** — the hit is now
as fast as a miss or a warm repeat. The only >2 s decode tick left was the
class-C request-finish publish (tick 352, 2578 ms; one per unique prompt);
class B did not recur in this window. Hit decode tok/s 39.0 on the fixed
first hit vs 12.0 baseline; acceptance unchanged 0.8484 across all four runs.

## Class C fix (B): batch every finish-publish D2H — and why reordering alone does not help

The device-resident pages finish-publish snapshots (`transfer_to_shared` →
`PagedKvPool._page_blob`) were blocking copies: `_page_blob` defaults to
`non_blocking=False`, so each page's D2H drained the stream before the next —
the same per-page-sync defect shape as class A, on the D2H (snapshot) side.
The per-page bounds `bv.cpu()` and warm-spec draft `dk/dv` `.cpu().clone()`
were separate blocking D2Hs as well, so batching only the trunk frames would
have left two stalls per page. The finish loop now opens
`pool.frame_snapshots()` for the whole `keys` batch: frame, bounds and draft
copies all launch non-blocking into their own pinned blobs and there is ONE
`cuda.synchronize` at the context exit. Frames are not freed/demoted there
(the caller keeps their lifecycle).

One correctness condition the first draft got wrong: the cold-tier commit
(`share_hold` / `share_hold_kv`) can spill an existing RAM blob to disk under
budget pressure, which reads host bytes. Committing inside the context could
therefore spill-read a page whose own D2H was still in flight. The batched
`transfer_to_shared(..., pending=[])` only launches copies and appends a
descriptor; `publish_at_finish` commits them all after the sync. The
non-batched offer_drop path is unchanged (inline, synchronous).

Gates: `test_batched_frame_snapshots_sync_once_and_byte_equal` — three
snapshots in one context = zero syncs in-context, one at exit, blobs
byte-equal; two separate contexts sync twice; red on the old code
(`frame_snapshots` does not exist).
`test_finish_publish_commits_no_blob_until_the_batch_sync_drains` spies on
both commit entry points during a real run and asserts the batching flag is
down at every call (a commit inside the window fails it), then proves the
deferred commits landed — shared bytes plus a follower adoption — so it
cannot pass vacuously on a no-publish build. Device re-measurement of tick
352's `release_cold_forget` sub-phases is pending a card slot.

Question asked before building: is the stall before or after the last token /
finish? Answer from the code — **before delivery, and reordering in the same
thread is not enough**. `_commit` calls `_finish` (→ `_release` → publish)
the moment max_new/stop is reached; `_finished[rid]` is only assigned inside
that same `_finish`, and `step()` holds `engine._lock` across the entire tick,
so `poll`/`take` cannot read the result until `_release` returns the lock.
Writing `_finished` before the publish within one tick therefore exposes
nothing earlier — the client still waits out the release. The only way to
fully move publish off the response path is a background thread, which needs
(a) delayed return of the snapshotted frames until the D2H lands (the pool is
sized to the n_groups*k pin ceiling with 0.5–0.9 GB free, so holding a
finished request's frames risks the evict/池-full path), and (b)
half-published-visibility and request-state-lifecycle gates. That is plan A,
not taken; B is taken first because it removes the per-page sync overhead
with zero concurrency/pool risk, leaving only the unavoidable D2H byte time.
