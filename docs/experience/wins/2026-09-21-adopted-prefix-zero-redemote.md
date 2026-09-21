# adopted 前缀重离池零 D2H — sm90/H20, 2026-09-21

> Status: pending-remote

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
real-demote fallback. Device delta (D2H bytes/tick saved per adopted page on a
warm follower) pending remote.

## Rule

When a promoted copy's content identity survives promotion, re-leaving the pool
is a reference operation, not a second materialization — check identity against
the live cache at drop time and keep the LRU-miss fallback.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-21 | d7fe6329 | pending pod | sm90 | Qwen3.8-27B NVFP4 | | | |

Raw artifacts: pending remote warm-cache follower window.
