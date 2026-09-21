# adopted 前缀重离池零 D2H — sm70/V100, 2026-09-22 设备旁证

> Status: **device-verdicted 2026-09-22 (V100 sm70, serve a38c3bc4 = #790 +
> #800).** Confirmed as a side-channel of the M6 adoption run, not a dedicated
> long-warm window: after the first adoption the shared set never grows and the
> follower re-leave window carries zero re-materialization D2H. A long
> warm-cache multi-cycle window would quantify bytes/tick saved at scale; the
> zero-cost property itself is observed.

## Context

Publish-once refactor M4 (#783), follow-up to
[2026-09-21-close-zero-bytes.md](2026-09-21-close-zero-bytes.md). A follower
adopts a published prefix into fresh PRIVATE device blocks; as decode advances
those blocks leave the k+window union. The old path demoted each re-leaving
adopted page — one full-page trunk D2H plus bounds and (spec) draft copies — to
republish bytes already held under the identical content key, after which
`share_hold` discarded the new blob and only bumped a reference. Under repeated
prefix traffic (warm cache) every adopted page paid that D2H exactly once per
follower per eviction cycle for zero new content. Metric: `demote_page` calls
and the `pub_bounds_d2h`/`pub_draft_clone`/`pub_frame_d2h` segments on a
follower that adopts and decodes past its hot window.


## What Worked

- `resolve` keeps the page→content-key label after promoting the shared blob to
  a private block.
- `finalize` checks the label against the live shared key set before demoting:
  a key still held returns the frame to the pool with no D2H (the page is still
  offered, so the contiguous frontier stays hole-free); a key evicted by the
  shared LRU falls back to a normal demote and republish.
- `transfer_to_shared` on an existing content key skips bounds/draft/frame D2H
  and the private lift, taking one `share_ref`.

The check, frame release and frontier closure are separate steps (the closure
can land a later tick), so a prefix-index capacity LRU could evict the key in
that gap and leave the closure naming a dead blob. The pin is now taken
atomically (`share_ref_if_present`: live-check and ref bump in one critical
section) before the frame is freed, on both release paths (finalize and
evict_victim); the ref is parked in `request_pins`, consumed by the closing
frontier (handed to the grow entry, no second bump) or released at request
drop when the frontier never closes. A missing shared blob at resolve falls
back to a fresh block instead of raising.

CPU gate: an adopted prefix re-leaving the union triggers zero `demote_page`
calls on adopted pages and zero private→shared lifts, with follower tokens
equal to a prefix-miss oracle; a forced post-adopt eviction exercises the
real-demote fallback.

## Device verdict (2026-09-22, V100, serve a38c3bc4, M6 #785 run)

Same `repro_adopt_796.py` run as the finish-publish errors entry; artifacts
`~/closewin-m6/r800/d796/` (`d796_adopt.json`, `d796_health_poll.csv`,
`serve.log`).

- After the publisher finish the shared set is one value for the rest of the
  run: **1286 pages / 1601306624 bytes across all 170 post-publish poll rows**
  (`d796_health_poll.csv` distinct `kv_cold_shared_pages`/`kv_cold_shared_bytes`
  = {1286}/{1601306624}) while `sparse_prefix_warm_adoptions` advances 0→1→2→3.
  Repeated adoptions add zero bytes — the dup-content re-leave is a reference
  bump, not a second materialization.
- The follower re-leave window (serve.log ticks 155–165, after the follower
  adopted and advanced) carries **zero** non-zero
  `pub_bounds_d2h`/`pub_draft_clone`/`pub_frame_d2h` (count 0); adopted pages
  re-leave without a re-materialization D2H.
- Follower tokens equal the temperature-0 oracle (`tokens_equal_oracle=true`).
- Cold-tier state at verdict: shared RAM 1.491 GiB / 1286 pages, shared SSD 0,
  private 0 (`d796_adopt.json` cold_after_publisher/follower).

Scope note: this is a short-advance follower (128-token suffix, 48 generated),
so it exercises adopt + re-leave of the pages its decode crosses, not a long
multi-cycle warm cache. The per-adopted-page zero-D2H property is observed;
quantifying aggregate bytes saved over many eviction cycles needs a dedicated
warm window and is the remaining measurement, not a red line.

## Rule

When a promoted copy's content identity survives promotion, re-leaving the pool
is a reference operation, not a second materialization — check identity against
the live cache at drop time and keep the LRU-miss fallback.

## Results

| date | serve sha | machine | target | model | shared growth on repeat adopt | re-leave D2H |
|---|---|---|---|---|---:|---|
| 2026-09-22 | a38c3bc4 | V100 sm70 | hybrid serve | Qwen3.8-27B NVFP4 | 0 pages (stays 1286/1.491 GiB, adoptions 0→3) | 0 (follower ticks 155–165) |

Sources: `~/closewin-m6/r800/d796/d796_health_poll.csv`
(`kv_cold_shared_bytes`/`kv_cold_shared_pages`/`sparse_prefix_warm_adoptions`
columns), `serve.log` ticks 155–165, `d796_adopt.json` cold_* states.
Long warm-cache throughput/tick window still pending.

