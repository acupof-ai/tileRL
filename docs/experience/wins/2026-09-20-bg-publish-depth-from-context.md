# Background-publish queue depth derives from context — 2026-09-20

> Status: landed behind the existing `TILERL_CLOSE_BG_PUBLISH=1` gate (still
> default OFF). CPU gates green; device numbers are the #743 review arms
> (perf1, V100, 2026-09-20). A follow-up to
> [2026-09-20-background-close-publish.md](2026-09-20-background-close-publish.md).

## Context

The background close-publisher shipped with a hard-coded queue depth of 512
pages. A real request close is one job per whole page — measured on the V100
37.6k fill as **2345 jobs per close** (`ceil(37567/16)`), not the 1-2/page
upper bound the depth was estimated from. One release burst therefore overflowed
the queue by 4.6x: the bg2 arm degraded **34.5%** of jobs back to the inline
locked transfer, so a third of the pages still moved under the tick lock. With
an 8192 depth the same run had **degraded=0** and `pub_cold_transfer` fell
1783 → 9 ms — the mechanism is sound; the bound just could not hold one close.

## What Worked

`build_engine` derives the depth from the engine shape instead of hard-coding
either 512 or 8192:

- **Depth** `= (num_slots + 1) * ceil(max_total_tokens / 16)` — room for every
  slot to close a full context in the same tick while the worker is still
  draining the previous wave, plus one spare wave. At 4 slots / 37.6k that is
  ~11.7k queue ENTRIES (entries mostly hold a key reference, not a slot), so
  the observed 2345-job close never degrades on count.
- **Payload byte cap.** Queue depth bounds object count, but two payloads exist
  before they enter the pinned cold budget (and its LRU spill), which happens
  only at worker commit: the page-sized "hold" frame blob the 1PR batch made,
  and a warm-spec "kv" job's attached draft `dk`/`dv` host snapshots (bounds
  ride with them). A cold page and its draft block (pages `p <= written_page`)
  overlap almost entirely, so a close burst can attach dk/dv to most cold
  pages — on sm70 (f32, DFlash2 5 layers × 4 heads × 256) that is 640 KiB per
  warm page; the static upper bound over all 2348 pages is ~1.43 GiB per
  request, over a 1 GiB budget. The kv BASE blob is not charged (already
  budgeted, or on disk for an SSD-source job). `TILERL_CLOSE_BG_MAX_BYTES`
  (build default = the cold RAM budget) caps these late-accounted queued bytes,
  so budget + queued stays **≤ 2× the cold budget**; an over-cap offer commits
  inline and spills at once, never OOM. In practice the dk/dv attach only over
  the prompt's real warm draft coverage (~128 pages/slot ≈ 82 MiB on this
  shape), far below the theoretical full-context bound, so the cap is a static
  guarantee rather than a normal-run tightness.
- Both stay overridable by env (`TILERL_CLOSE_BG_DEPTH`,
  `TILERL_CLOSE_BG_MAX_BYTES`); with the gate off nothing changes and no worker
  starts.

The queue itself never allocates the big blob — the 1PR close batch made the
frame blob and the draft snapshots unconditionally on every path — so deriving
a large depth does not increase the peak; it only stops forcing those
already-made bytes through the locked path.

## Gates (CPU, hermetic)

- `build_engine` with the gate on yields depth `(slots+1)*ceil(ctx/16)` and
  byte cap = cold budget; explicit env vars win for both.
- Tier gates: at a one-blob cap, a second hold offer degrades inline; a kv job
  whose draft `extra` would exceed the cap also degrades inline while a kv job
  with only small bounds is still admitted; the queued-byte counter returns to
  zero after commit. The gates separate the actually-queued bytes (small) from
  the theoretical full-context upper bound so the cap is not tightened against
  coverage that never happens.
- Full CPU suite 1027 passed / 22 skipped / 1 xfailed.

## Rule

Size a background work queue from the worst burst the engine shape permits
(concurrency × per-unit work + slack), not a round number measured on no
configuration; and bound it on BYTES for the one payload that is large and
accounted-late, separately from the object count that bounds the rest — let
over-budget work degrade to the synchronous path rather than grow memory.

## Results

| date | machine | target | config | result |
|---|---|---|---|---|
| 2026-09-20 | V100 sm70 (#743 review, perf1) | 37.6k request close, bg2 | depth 512 | 34.5% degraded; depth 8192 → 0 degraded, pub_cold_transfer 1783→9 ms |
| 2026-09-20 | CPU (hermetic) | derived depth + byte cap | 4 slots / 4096 ctx | depth=(4+1)·256; 1027 passed |

Raw artifacts: `tests/test_sparse_engine.py`, `tests/test_sparse_kv_tier.py`;
changes `src/tilerl/build.py`, `src/tilerl/kv_tiers.py`.
