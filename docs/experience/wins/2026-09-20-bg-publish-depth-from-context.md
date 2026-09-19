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
  ~11.7k slots, so the observed 2345-job close never degrades on count.
- **Payload byte cap.** Queue depth bounds object count, but a "hold" job
  carries the page-sized frame blob the 1PR batch already allocated; enqueue
  delays that blob entering the pinned cold budget (and its LRU spill), so a
  worker lag could briefly pin a burst of frame blobs the budget did not see.
  `TILERL_CLOSE_BG_MAX_BYTES` (build default = the cold RAM budget) caps the
  total queued hold-payload: worst-case pinned is ≤ 2x the cold budget and the
  LRU converges right after commit. "kv" jobs (blob already budgeted) and
  private-SSD jobs (bytes on disk) are uncharged, so a frame-heavy burst cannot
  starve the reference-heavy SSD lifts behind the same counter. Over-cap offers
  degrade to the inline commit, which spills immediately — never an OOM.
- Both stay overridable by env (`TILERL_CLOSE_BG_DEPTH`,
  `TILERL_CLOSE_BG_MAX_BYTES`); with the gate off nothing changes and no worker
  starts.

The queue itself never allocates the big blob — the 1PR close batch made it
unconditionally on every path — so deriving a large depth does not increase the
peak; it only stops forcing those already-made blobs through the locked path.

## Gates (CPU, hermetic)

- `build_engine` with the gate on yields depth `(slots+1)*ceil(ctx/16)` and
  byte cap = cold budget; explicit env vars win for both.
- Tier gate: at a one-blob cap, a second hold offer degrades inline while a kv
  job is still admitted (uncharged), and the queued-byte counter returns to zero
  after commit.
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
