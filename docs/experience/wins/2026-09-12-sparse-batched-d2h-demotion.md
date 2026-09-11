# Batched non-blocking D2H demotion: one sync per tick's departing pages — 2026-09-12

> Status: **CPU gates green; the decode-tick ms delta is the V100/H20 re-run
> (the non-blocking launch + single sync is a card-only latency change).**
> Stacked on #534 (cross-tick hot pin). #534 batched the promote H2D; this is
> the demote half. Only the `PagedKvPool` demote path and the two engine demote
> sites change — no selection, table, or `SparseTracker` code.

## Root cause

#534 stopped a stable selection from demoting/promoting anything, but when the
set changes the departing pages still demoted one at a time. `_page_blob`
issued the D2H of each page's K/V (and fp8 scales) with a **blocking**
`host.copy_(t)` and `demote_page` freed the frame immediately, so a tick that
evicted N pages paid N serialized D2H stalls before any frame came back to the
pool. The host-op census measured this as the whole sparse-minus-dense delta on
main: at B=8, `_page_blob` D2H K/V +112/+112 ops.

## Fix — one `pool.demotions()` batch

Mirrors `pool.promotions()`:

- `_page_blob(block, non_blocking=…)` launches each plane's D2H into its
  **pinned** staging buffer with `copy_(t, non_blocking=True)`;
- inside `with pool.demotions():`, `demote_page` launches the copy but **does
  not free the frame** — the frame is retained so a recycled allocation (or an
  interleaved promote) cannot overwrite data an in-flight D2H is still reading;
- on context exit one `torch.cuda.synchronize()` waits for every page, then
  all blobs are held in the cold tier and all frames return to the pool.

N departing pages now cost one sync, not N, with a single happens-before edge
before frame reuse. Off cuda the copies are plain synchronous clones, so the
context only changes bookkeeping there.

Both engine demote sites are wrapped for the tick: `_sparse_finalize` (the
automatic sparse path, one batch across all rows) and `sparse_retier` (the #500
dense seam — batching here also prevents an interleaved promote's
`alloc_block` from recycling a frame mid-D2H).

## Gates

- `test_batched_demotions_copy_many_pages_but_sync_once`: three pages under one
  `demotions()` on a pool whose device is mocked to cuda — zero
  `cuda.synchronize` inside the batch, **exactly one at exit**; frames stay live
  (refcount 1) until the sync and are freed after; every held blob is byte-equal
  to its frame.
- `test_batched_demotion_survives_frame_recycling`: a 2-frame pool demotes two
  distinct pages `(0,0)`/`(0,1)` in one batch — the second must not reuse the
  retained first frame; after the single sync both promote back byte-equal with
  their own K/V (the recycled-id collision, enforced through a real CPU pool).
- existing tier/sparse/ledger/kv gates unchanged: 68 passed, 1 skipped, 6 xfailed.

## Census delta (host ops, one CPU decode tick, tiny k=2)

The aten-op count is unchanged on a **stable** selection because #534's pin
already demotes zero pages there (sparse B=8 1302 ops / 123 copies both before
and after this change). The win is on a **changed-selection** tick, and it is a
device-latency change not visible in CPU aten counts:

| measure, N pages depart one tick | before | after |
|---|---:|---:|
| blocking D2H `copy_` stalls | N (K) + N (V) | 0 |
| non-blocking pinned D2H launches | 0 | N (K) + N (V) |
| explicit `cuda.synchronize` | N (one per demote, serialized) | **1** (before frame reuse) |
| frame reusable before D2H done | — | no (retained until sync) |

The correctness boundary is the retained frame + single sync: reusing a frame
earlier would race the D2H, and dropping the sync would hand
`promote_keyed`/`alloc_block` buffers whose data is still in flight. The card
ms for a set-changing B=8 tick is the pending-remote number.

## Rule

A non-blocking cross-device copy needs two things before its source/destination
may be reused: the buffers retained until a single happens-before sync, and
exactly one such sync for the whole batch. Batch the departing set and keep its
frames live; do not free each frame behind a per-page blocking copy.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-12 | Mac CPU (tiny, real pool + recycling gate) | cpu | N demoting pages sync once not N; frames retained until the batch sync; blobs byte-equal through a 2-frame recycling case; set-changing tick ms pending card |
