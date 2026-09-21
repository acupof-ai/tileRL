# Batched one-sync D2H for request-close prefix publish — 2026-09-19

> **Superseded 2026-09-21 by #787 (Epic #779 M1)**: #741's close-batch D2H context is the first item of the deletion list. The mechanism this entry measures is deleted; the page now publishes once, when it leaves the pool. See [errors/2026-09-21-optimizing-at-the-wrong-layer-close-scheduling.md](../errors/2026-09-21-optimizing-at-the-wrong-layer-close-scheduling.md).
> Status: landed behind an env gate (`TILERL_CLOSE_BATCH_D2H=1`), default OFF,
> device delta measured on V100 2026-09-19 (659c2fbb): request-close median
> 2940 → 2321 ms (−21%), steady decode unchanged. First of two PRs attacking
> the deterministic per-request close-time lock stall (the 1.8–8.9 s in the
> step timer's `sample` bucket measured on V100 2026-09-18). This PR removes
> the per-page device sync count; the second PR moves host/SSD bytes off the
> critical path.

## Context

When a sparse publisher finishes with `sparse_matched==0`, `engine._release`
(under `engine._lock`, inside the `sample` bucket) calls `prefix.close_request`
and then `transfer_to_shared` for every prompt-end page not already published.
Pages still DEVICE-resident (the own window never dropped them) were snapshotted
one page at a time: bounds `.cpu()`, draft K/V `.detach().cpu().clone()`, and
`PagedKvPool._page_blob()` for the trunk frame — each a blocking cross-device
copy with its own implicit sync, each allocating fresh pinned tensors, all
serialized under the lock before the publisher's blocks were freed.

The byte volume never explained the multi-second stall (~2 GiB at 12 GB/s is
~0.2 s); the ~10k independent per-page synchronizations did. #732's five release
sub-segments localized it (`pub_bounds_d2h` / `pub_draft_clone` /
`pub_frame_d2h` / `pub_cold_transfer` / `ssd_mmap`).

## What Worked

A prepare / single-sync / commit shape, gated by `TILERL_CLOSE_BATCH_D2H=1`,
mirroring the existing `demotions()` batch:

- `PagedKvPool.close_publishes()` is a context. Inside it `host_snapshot()` and
  `_page_blob(non_blocking=True)` launch the bounds/draft/frame D2H into
  per-page pinned buffers without per-page sync; the page's cold-tier commit is
  deferred. At `__exit__` exactly one `torch.cuda.synchronize` makes every
  snapshot valid.
- `engine._release` runs all the close transfers inside the context, then
  commits the deferred pages (`SparseRuntime.transfer_deferred`) **after that
  sync and strictly before the `free_block` loop** — so a recycled publisher
  frame cannot overwrite bytes a non-blocking D2H is still reading.
- Host/SSD-source pages (already off device) defer their tier commit too so the
  ordering is uniform, but they move no device bytes.
- Per-page independent pinned blobs are retained (the batch collapses syncs, not
  allocations): a published frame blob is held in the cold tier until spilled,
  so a single reusable staging buffer would alias. Pinned-buffer pooling is the
  follow-up PR and needs a cold-tier release hook.

Default (gate unset) is the byte-for-byte old per-page blocking path. The gate
is backend-agnostic: on the CPU cell copies are synchronous clones and the tail
sync is a no-op, but the identical prepare/defer/post-sync-commit split runs, so
the e2e gate covers the exact code path cuda uses.

## Gates (CPU, hermetic)

- Full CPU suite 1008 passed / 22 skipped / 1 xfailed with the gate both unset
  and set.
- Engine e2e (gate set): a prompt fully inside the hot union publishes all its
  pages from live frames at close (no DROP publishes); every published shared
  blob is present post-commit with its bounds plane, and a same-prompt follower
  hits the full prefix and decodes exactly the prefix-miss oracle tokens.
  Mutation-verified RED: deleting the post-sync deferred commit leaves those
  close-frame pages unreadable (`share_take → None`).
- Pool-level gate: the batch is unsynced inside / synced at exit, and per-page
  blobs are byte-correct in independent storage (no shared staging buffer).
- The pre-existing batched-demotion gates (one batched `cuda.synchronize`) pin
  the sync-before-frame-free contract this reuses.

## Device confirmation (V100 sm70, 2026-09-19, 659c2fbb)

Three served arms on the 27B, same fill-then-warm protocol (2× 37.6k
independent fills to a full tier, 2 warm reps of 32 tokens, W=2048, 1 GiB-RAM /
8 GiB-SSD f16, `SLOW_MS=0`): cb0 both gates off, cb1 `TILERL_CLOSE_BATCH_D2H=1`,
cb2 plus `TILERL_COLD_PREFIX_SSD_CAP=1`. Only the cb0→cb1 pair isolates this PR.

Per-request close, five `#732` sub-segments, median ms over the release tail
ticks (n=6 cb0, n=7 cb1):

| sub-segment | cb0 off | cb1 batch | delta |
|---|---:|---:|---:|
| `release_close_request` total | 2940 | 2321 | **−21%** |
| `pub_frame_d2h` | 340 | 41 | −88% |
| `pub_bounds_d2h` | 293 | 111 | −62% |
| `pub_draft_clone` | 352 | 177 | −50% |
| `pub_share_hold` | 210 | 161 | −23% |
| `ssd_mmap` | 1833 | 1865 | flat |
| `pub_cold_transfer` | 1726 | 1783 | flat |

Steady decode did not regress: tick p50 184→183 ms, model segment 168→166 ms,
draft 12→12; effective tok/s (decode ticks + accepted bonus over decode wall)
9.49/9.01 → 9.37/9.38.

Correctness (client-terminal observation, not vendored): a follower repeating
an identical 32k prompt returned byte-identical tokens, `finish_reason=length`,
and `/health` `prefix_hits` moved +1 in the client-side `follower_smoke.py`
run. That stdout was not saved and the boot log has no per-request prefix-hit
line, so treat this as a live operator read, not an artifact — re-capture the
follower response and health delta to a file next window.

The two largest close terms — `ssd_mmap` (~1.83→1.86 s) and
`pub_cold_transfer` (~1.73→1.78 s) — are unchanged. They are host/SSD byte
movement, not device-sync count, which is exactly the scope reserved for the
second PR. This PR's −21% is the frame/bounds/draft per-page sync collapse and
is the clean low-risk lock-in baseline predicted at landing.

Vendored: `close-batch-cap-device-2026-09-19/{cb0,cb1,cb2}.json` (arm summaries)
and `close-segments.txt` (the full sub-segment parse). Raw per-tick timing is in
`~/tilerl-logs/serve-cb{0,1,2}.boot` on the V100 box.

## Rule

When N page blobs leave the device at one boundary under a lock, launch every
non-blocking D2H first and synchronize once before any source frame is freed or
any host byte is read — never one blocking `.cpu()` per page. Defer the tier
commit to after that single sync; the sync must precede frame recycling.

## Follow-up

- Pinned-host-buffer pooling with a cold-tier spill/release return hook (remove
  the per-page pinned allocator churn).
- Move the host/SSD close bytes (`ssd_mmap` 1.8–2.0 s + `pub_cold_transfer`
  1.7–1.8 s) off the lock-critical path (bounded background publisher /
  defer), chosen after the device sub-segment re-read.

## Results

| date | commit | machine | target | model | close med ms | per-page D2H sub-segments | throughput |
|---|---|---|---|---|---:|---|---|
| 2026-09-19 | pending PR | CPU (hermetic) | engine close-time prefix publish D2H | — | — | N per-page syncs → 1 | — |
| 2026-09-19 | 659c2fbb | V100 sm70 | request-close prefix publish D2H | Qwen3.8-27B-NVFP4, 37.6k sparse, 1G/8G f16 | 2940→2321 (−21%) | frame −88%, bounds −62%, draft −50%; ssd_mmap/cold_transfer flat | steady unchanged; eff 9.0–9.5 tok/s |

Raw artifacts: `tests/test_sparse_engine.py`; changes `src/tilerl/kv_cache.py`,
`src/tilerl/sparse_runtime.py`, `src/tilerl/engine.py`.
