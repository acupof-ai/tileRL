# 1-tick-delay async sparse refresh — plan and pre-registered quality gate (#805)

Status: PLAN, no src change yet. Branch probe/805-serve-sm70. Supersedes the
"whole-cycle A'" in the refresh analysis: selection churn measured Jaccard
median 0.438 (56 adjacent refresh pairs × 4 groups, min 0.058), so any
multi-tick staleness (R=16, or applying a selection one whole 8-tick cycle
late) is not free. The only staleness that might be free is ONE decode tick.

## Mechanism

Measured in the phase window (revision f7a93e5c/a6b6a511, V100 sm70,
`TILERL_REFRESH_PHASES=1`, CUDA-event spans): the every-8-tick synchronous
refresh step p50 is **249.312 ms** (n=15) vs a captured graph tick **45.732
ms** (n=108) — a **203.58 ms** delta. The eager sparse trunk forward
(`eager_trunk_forward`) is **198.663 ms = 97.6%** of that delta; the same
geometry replays in 45.732 ms, so the eager trunk runs **4.34x** its own
graph. Non-trunk work is 50.649 ms; host blocking is only 1.76 ms/tick
(0.87%: synchronize ~1.76, 4 `.item` + 7 `.tolist` per refresh tick;
select_quest ~17.3 ms, promote ~24.5 ms, select_pages ~3.7 ms). The refresh
tick does three serial things:

1. eager `_select` per source group — bounds `index_select` + chunked
   `quest_scores` (device), then `select_pages` whose width is forced to host
   via `.item()` (reference.py) and the chosen ids pulled back via
   `sel.tolist()` (sparse_engine.py `_select`). ~8 host readbacks for the 27B.
2. resolves newly named cold pages inline — batched private promote (one sync
   at batch exit), per-page `cuda.synchronize` on `shared_promote`, a blocking
   mmap→pinned read when the blob was spilled, and a blocking mid-forward
   evict (D2H + possible SSD write) when the pool is full.
3. runs the whole trunk eager (`reuse=False`, Python-rebuilt tables), not a
   graph replay.

1-tick delay: on a side stream, AFTER each graph tick, compute the next tick's
candidate selection (bounds/quest are pure device ops) and promote the named
cold pages. The following graph tick then attends the new set through the
normal captured replay; refresh never runs an eager trunk. The selection is at
most one decode tick behind the current query.

A graph replay masks non-resident picks to pad 0, so the promoted set MUST be
fully resident before the next replay: a stream wait on the promote batch
(including the per-page shared sync and mmap) is the core synchronization
point.

What it removes: the 198.663 ms eager-trunk penalty (the refresh tick becomes
a graph replay). The ~50.6 ms non-trunk select/promote work is moved off the
critical path but now runs concurrently with graph ticks and shares the H2D/
SSD bandwidth, so graph ticks may get slightly longer — the net gain is
measured, not assumed. This is the 1-tick design rather than a longer refresh
interval: the phase window already answered the A'-vs-B question (the gap is
the eager trunk, not host sync), and churn Jaccard 0.438 says staleness is
not free.

Staging: SHADOW MODE first — compute and promote in the background while the
tick still uses the synchronous selection, changing no tokens; measure whether
background prep keeps up and how much it lengthens the graph tick. Switch to
the real delay only if shadow shows it fits.

## Quality gate (pre-registered, 2026-09-24)

Churn is not quality. The delayed configuration is judged against the CURRENT
R=8 synchronous refresh on the same prompts, temp 0, same seed.

1. NOISE FLOOR — MEASURED 2026-09-24, revision 768c3f8c, V100 sm70, same
   machine, two back-to-back synchronous R=8 graph floor runs (no other job in
   between), 6 real 37.6k wikitext prompts, 1024 generated each:
   `compare_refresh_runs.py floorA_pp floorB_pp`:
   - min/mean/median per-prompt agreement all **1.0**; n_diverged_prompts 0;
     every prompt matched 1024/1024; first divergence null; mod8 null.
   So the temp-0 noise floor on this machine is exactly 0, not an assumption.
   Distinct inodes, 25 min apart; cross-prompt runs differ at position 0, so
   the identical result is run-to-run determinism, not a vacuous comparator.
   Consequences, locked with the measurement:
   - the −1.0pp / −0.5pp slacks now absorb ONLY the delayed side's drift, not
     measurement noise (measured noise is zero);
   - `median first-divergence >= 128` is the binding divergence criterion (an
     agreement of 0.9951 would still clear 0.99);
   - the mod8==0 clustering check has no object in the floor (n_diverged=0);
     it applies only to the delayed run.

2. PASS LINE, fixed BEFORE any delayed run and not adjusted afterward:
   - per-prompt agreement `>= floor_i - 1.0 percentage point` for every prompt
     (floor_i measured = 1.0, so effective line 0.99);
   - mean agreement across prompts `>= floor_mean - 0.5 percentage point`
     (effective 0.995);
   - median first-divergence position `>= 128` generated tokens (16 refresh
     periods; a divergence in the first 128 tokens fails) — the binding line;
   - divergences must not be monotonic-clustering at refresh boundaries
     (record divergence position mod 8; a run dominated by position 0 fails
     even if the aggregate agreement passes).
   Below any line: 1-tick staleness is rejected regardless of the speed gain.

Prompt set: the same 37.6k real wikitext prompts used in stage 1; report n
and every per-prompt value, not just the aggregate.

## Shadow v1 — go/no-go (pre-registered 2026-09-24, before the run)

Env-gated (`TILERL_SPARSE_SHADOW=quest|h2d|both`, `TILERL_SPARSE_SHADOW_PAGES`
for the H2D volume), probe branch only, default off. No residency side
effects: after each graph tick, on a side stream, run the quest compute with a
synthetic post-rope q (same shape/SM cost, result unused) and/or issue cold H2D
copies into scratch blocks carved out of `num_blocks` (popped off the pool's
free list, restored on shutdown — capacity leaves via free_blocks, not the
fixed `num_blocks`); nothing is mapped into l2p and no live tick reads the
scratch, so output is token-identical with shadow off (CPU gate asserts it).

- Budget: carve scratch out of `num_blocks` (no +k; V100 has only 2-3 GiB
  free), capped at half the current free pool. Verdict prints carved pages,
  num_blocks fixed, and free blocks before/after.
- The real per-refresh promotion volume is measured in the SAME window by
  delta-ing `HostKvPages.promotions` (`kv_cold_promotions`, the `.take` H2D
  counter) on each refresh tick — `offers_pages` is EVICTIONS and churn's 400
  is REPLACED-pages; neither is promotions, so neither sets the volume.
  Reported per run as refresh_promotions p50/p90/max.
- Synthetic H2D volumes run at three points: **107** (offers_pages p90,
  proxy), **206** (observed max eviction), **512** (4×k supremum). The gate is
  read at the point matching the measured real promotion p90; if that p90
  falls between two points, report both adjacent points. quest is
  informational (SM contention only; no H2D gate).
- Verdict (n ≥ 50 graph ticks per side, same process, alternating on/off in
  segments as a placement control; start-to-start intervals):
  - background device time p50/p90/p99;
  - the real budget denominator is graph-tick interval measured in OFF
    segments (ON-segment intervals include the background's own wait and
    would make the fit gate self-proving);
  - graph-tick p50 OFF vs ON;
  - tail: at each ON tick start, non-blocking event query of whether the
    previous tick's background finished — fraction NOT finished.
- GO line, locked before the run (h2d/both, at the measured-volume point):
  1. background **p90 ≤ OFF-segment graph-tick interval p50**;
  2. graph-tick p50 slowdown ON vs OFF **≤ 5%**;
  3. **fraction of refreshes whose background exceeds one interval ≤ 5%**
     (the tail: an occasional two-tick refresh is the real-delay failure).
  All three must hold; any fails → v2 (real delay) is not built.

## Shadow v1 — RESULT (measured 2026-09-24, revision 24898a26, V100 sm70)

Window `~/shadowwin-0924-054509`, `PROBE EXIT=1`, five configs each its own
subprocess; token gate OK (400 tokens, all configs identical to off).

| config | bg p50 | bg p90 | bg p99 | interval_off p50 | g_off | g_on | slow | exceed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | – | – | – | 45.91 | 45.76 | – | – | – |
| quest | 22.11 | 23.15 | 31.29 | 46.50 | 46.25 | 68.85 | 1.491 | 0.0 |
| h2d107 | 33.71 | 71.43 | 75.87 | 61.77 | 47.58 | 56.37 | 1.185 | 0.837 |
| both107 | 54.77 | 55.60 | 56.55 | 46.59 | 46.12 | 77.99 | 1.691 | 1.0 |
| h2d206 | 67.01 | 68.50 | 69.15 | 46.43 | 46.19 | 64.45 | 1.395 | 1.0 |

Real cold promotions this window: n=29, p50 69 / p90 138 / max 156 — p90
bracketed by the 107 and 206 points.

All three v1 gates failed (rc1) for h2d/both. quest — zero H2D, zero residency,
the minimal config — already failed gate 2 (slow x1.491), so the no-go does not
depend on any H2D-volume assumption.

**Criterion 2 was specified against the wrong object.** v1 fired the background
on EVERY graph tick, so every tick paid SM contention. The real design fires
once per 8-tick cycle and the work overlaps the following ticks. Re-derived
cycle estimate (8 ticks, current = 7×45.76 + 249.3 = 569.6 ms): if the once-per-
cycle wait is bg≈56–71 ms, cycle ≈ 7×46 + 56..71 = 389–407 ms (−28–32%, eff
~34–36 tok/s). That assumes the seven overlapping ticks stay near 46 under side-
stream contention — exactly what v2 must measure, not assume. The v1 every-tick
slowdown does not transfer one-for-one to the once-per-cycle design.

Open instrument note: h2d107 interval_off p50 = 61.77 vs 45.9–46.6 for the
other four configs. OFF segments emit no background, so by construction this
should be config-independent. Offline parser `probe_shadow_interval.py` splits
OFF gaps into plain / spans-eager-refresh / other intervening steps and by
segment, to localize it (non-steady first gaps vs an intervening-step class).

## v2 — real 1-tick delay (ruling 2026-09-24)

Shadow GO/NO-GO as pre-registered is mooted by the object error; build the real
delay and measure the cycle directly.

- At refresh cadence (every 8 ticks) start the background from the PREVIOUS
  tick's REAL q: store the per-layer post-rope q in the captured forward.
- Background: quest + promotion into the reserved blocks on the side stream.
- On the next graph tick: wait on the event, map the selection into l2p, replay.
  If promotion is not done yet (bg > interval) that tick WAITS on the event; it
  does NOT fall back to eager.
- Measure: whole 8-tick cycle p50/p90, eff tok/s, and the quality gate below.
- CPU gate: v2 with delay=0 must be token-identical to the synchronous path.

## Device measurement order

- ANSWERED by the phase window (f7a93e5c/a6b6a511, no nsys needed — nsys
  2022.4.2.1 export is broken): the gap is the eager sparse trunk (97.6%),
  not host sync (0.87%). This picked the 1-tick-delay design over a longer
  refresh interval; see Mechanism.
- DONE: shadow mode. v1 gates fired red under every-tick emission; criterion 2
  targeted the every-tick object, not the once-per-8-ticks design, so it does
  not decide the design — v2 measures the real cycle.
- NEXT: v2 real delay (above) against the pre-registered quality gate.

