# 1-tick-delay async sparse refresh — plan and pre-registered quality gate (#805)

Status: PLAN, no src change yet. Branch probe/805-serve-sm70. Supersedes the
"whole-cycle A'" in the refresh analysis: selection churn measured Jaccard
median 0.438 (56 adjacent refresh pairs × 4 groups, min 0.058), so any
multi-tick staleness (R=16, or applying a selection one whole 8-tick cycle
late) is not free. The only staleness that might be free is ONE decode tick.

## Mechanism

Today's every-8-tick synchronous refresh runs eager and costs p50 ~203-211 ms
vs ~46 ms for a captured graph tick; ~128 ms of the gap is inside the eager
sparse `model.forward` envelope, ~157 ms total. It does three serial things:

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

What it removes from the 157 ms:
- the eager-trunk launch gap (C3) — refresh tick returns to graph replay;
- the ~8 in-forward readback stalls (C1) — moved off the critical path;
- inline promote/shared/mmap/evict blocking (C2) — moved off the critical
  path, but the H2D/SSD bandwidth now contends with the graph replay, so the
  graph tick gets slightly longer. Net gain is measured, not assumed to be 46.

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

## Device measurement order

- task 2 breakdown (commit 9b5cd169, `TILERL_REFRESH_PHASES=1`) attributes the
  128 ms without nsys (nsys 2022.4.2.1 export is broken): GPU-event spans for
  quest/select_pages/promote/shared_promote/eager trunk, wall time for
  synchronize/evict, raw `.item/.tolist/.cpu` counts. Decides whether the gap
  is idle/sync-bound (A' wins) or real eager trunk GPU time (B wins).
- then shadow mode, then the real delay, then the pre-registered gate above.
