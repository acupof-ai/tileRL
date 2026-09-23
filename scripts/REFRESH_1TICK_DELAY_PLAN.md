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

## Shadow v1 — result (2026-09-24)

Window `shadowwin-0924-054509`, revision `24898a26`, V100 sm70,
`graph_w2048`, one real 37.6k wikitext prompt, 5 configs each in its own
process, 106 OFF / 100 ON graph ticks per config.

| config | bg p50 | bg p90 | bg p99 | OFF interval p50 | graph p50 OFF | ON | slowdown | over-1-interval | carved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | — | — | — | 45.91 | 45.76 | — | — | — | — |
| quest | 22.11 | 23.15 | **31.29** | 46.50 | 46.25 | 68.85 | **1.491** | **0.0** | 0 |
| h2d107 | 33.71 | 71.43 | 75.87 | 61.77 | 47.58 | 56.37 | 1.185 | 0.837 | 107 |
| both107 | 54.77 | 55.60 | 56.55 | 46.59 | 46.12 | 77.99 | **1.691** | **1.0** | 107 |
| h2d206 | 67.01 | 68.50 | 69.15 | 46.43 | 46.19 | 64.45 | 1.395 | **1.0** | 206 |

All three H2D configs failed the pre-registered GO line (rc1). **This
no-go is v1's verdict.** It is not a verdict on the 1-tick design, whose
binding gates are the measured 8-step cycle time and the quality gate
above. The token gate passed everywhere (`OK (400 tokens)`), which is the one v1 property
that was being tested and held: shadow on/off produce identical output, so
the background has no residency side effect. Measured cold-promotion
distribution: n=29, p50 69, **p90 138**, max 156 — the p90 sits between the
two synthetic volumes, so both are reported.

**The gate failed as pre-registered, and that stands.** The finding is that
the gate measured the wrong object. v1 fired background on *every* graph
tick (`n_bg == n_graph_on` was ruled as the invariant); the 1-tick design
launches it **once per refresh cycle**, i.e. once per 8 ticks, because only
the *next* tick needs the new selection. So v1 measured the worst case of
per-tick contention, and criterion 2 ("graph tick slowdown <= 5%") is not the
criterion this design must meet. This is stated plainly rather than as a
retroactive relaxation.

What the same data says about the design (the per-tick contention result is
still real and is the input): the background compute is **not hidden in the
graph tick's shadow — it adds on top nearly ms-for-ms**. The `quest` config
is the proof, since it moves no pages and carves nothing (`carved=0`): its
23.15 ms p90 fits the interval and its over-1-interval fraction is 0.0, yet
it pushes the graph tick from 46.25 to 68.85 ms. No copy volume explains
that. (Mechanism — no idle SM capacity on this card at this tick
occupancy — is an explanation, not a measurement.)

Re-derived cycle time, **estimated, not measured** (7 plain graph ticks +
1 background tick per cycle; current = `7 x 45.76 + 249.3 = 569.6 ms`):

| variant | cycle estimate | vs current |
|---|---:|---:|
| quest only (no page moves) | `7 x 45.76 + 68.85 = 389.2 ms` | −31.7% |
| both @107 (+9 ms measured wait) | `7 x 45.76 + 77.99 + 9.1 ≈ 407 ms` | −28.5% |

Effective throughput would go from ~24.9 to roughly 34–36 tok/s — a gain, but
not the 40+ the per-tick arithmetic suggested. These are arithmetic on v1
numbers and are labelled estimates; the delayed configuration's cycle time
must be measured, and that measurement is a binding gate, not this table.

### The `h2d107` OFF-interval anomaly — attributed

`h2d107`'s OFF interval p50 read 61.77 ms while all four other configs read
45.9–46.6 ms, and OFF segments emit no background, so that quantity should not
depend on the config. An offline parser over the `[step-timing]` log
(splitting each OFF gap into plain / spans-a-refresh / other) locates it: it is
**`h2d107`'s first OFF segment only**.

**Artifact.** Every number in this section is in
`scripts/shadow_v1_interval_summary.json`, produced by
`scripts/probe_shadow_interval.py` at probe-branch commit `96bc4fa0`
(pushed), run over the five raw `shadow_v1_*.dev.err` files in
`/home/chenkailun.c/shadowwin-0924-054509` on the V100. The summary carries the
seg0/seg2 table, the tick-80 raw line per config, the plain/mixed gap split,
and the OFF-segments-past-the-first denominator (n=55) for all five configs;
the raw directory is readable read-only over `ssh v100` for cross-checking.

| config | seg0 graph p50 | seg0 `stats` p50 | seg2 graph p50 | seg2 `stats` p50 |
|---|---:|---:|---:|---:|
| off | 42 | **3** | 42 | 3 |
| quest | 42 | 3 | 42 | 4 |
| **h2d107** | **84** | **41** | 42 | 3 |
| both107 | 45.5 | 5 | 42 | 3 |
| h2d206 | 42 | 4 | 42 | 3 |

`OTHER n=0` in all five files, so no extra class of gap exists. Raw lines at
the same tick show the extra is host-side: `h2d107` tick 80 is
`total=123 stats=43 graph=79` against `off` tick 80 `total=40 stats=3
graph=37`, at the same `free=3914 MiB`, `why=gpu_drain`, `d_malloc=0`, no
stall.

**Cause is startup non-stationarity in the first segment, not the segment's
contents.** Segment 0 is the first ~50 graph ticks after the last prefill —
first graph capture per bucket key, page-carve settling, allocator warmup — and
its gaps are both inflated and higher-variance; the effect is config-dependent
(`h2d107` worst, `both107` partially, the rest clean). The gap *structure* is
identical across configs, which rules out the segments mixing different kinds
of step: every config has exactly **43 gaps with nothing between the two graph
ticks and 7 with an intervening step**, yet `h2d107`'s 43 plain gaps read
p50 **127 ms** against `off`'s 46 ms, and its `stats` p50 is 41 ms against 3 ms.
By the second segment every config reads 42–46 ms graph and 3 ms `stats`,
including `h2d107` — so the carve itself costs nothing in steady state.

Corrected denominator (OFF segments past the first — the artifact's
`off_segments_past_first`, n=55 each): `h2d107` **46**, `off` 45, `both107`
45, `quest` 46, `h2d206` 46. Criterion 1 was `71.43 <= 61.77 → False`; at
46 it is more False. **All three gates fail under either denominator, so the
v1 no-go does not depend on this denominator.**

For the v2 probe the adopted fix is to **discard the first startup segment**
before taking cycle p50/p90, keeping segments cut at graph-tick boundaries:
the 8-tick cycle is defined by graph ticks and the one-tick wait at the cycle's
end is part of the design, so it belongs in the steady-state distribution.

## v2 pre-registration — gates, fixed before the v2 window (2026-09-24)

Locked by the coordinator before any v2 number exists, so the gates cannot be
set after seeing the result. v2 = real lagged-q + real selection mapping,
background launched **once per refresh cycle** (not once per tick, which is
what v1 measured).

**The prize, from the v1 numbers.** Current cycle `7 x 45.76 + 249.3 = 569.6 ms`
per 8 ticks (71.2 ms/tick). The 7 plain graph ticks are already 320.3 ms
(56.2%) of it; the whole prize is the 249.3 ms eager refresh tick (43.8%).
v2's cycle is `7 x plain + 1 x carrying tick`.

1. **CYCLE, measured — the shape gate.** Measure the 8-tick cycle directly,
   p50 and p90, **discarding the first startup segment** (v1 showed segment 0
   is non-stationary: `h2d107` read 84 ms graph / 41 ms `stats` against 42/3
   for every other config, settling by segment 2). Report p50/p90 on the
   directly measured cycles only.
   **Reconciliation is on MEANS, not a sum of percentiles** — a sum of p50s is
   not the p50 of the sum. Compare `7 x mean(plain tick) + mean(carrying tick)`
   against `mean(cycle)`; a gap over 5% means the segments still hold
   non-stationary ticks and the reading is not usable.

2. **THE CARRYING TICK, absolute — the primary gate.** The carrying tick's
   **p90 <= 135.4 ms**. Absolute ms, not a slowdown ratio: with only 1 tick in
   8 hosting background, a ratio gate would be diluted 7x into a false margin.
   135.4 ms is the 20% tier (`7 x 45.76 + 135.4 = 455.7 ms`, a 20% cycle cut
   against 569.6), and it implies background <= 89.6 ms on a 45.76 ms plain
   tick.

3. **`carry_fallback_frac`, per cycle — <= 0.05.** A *cycle* where the
   background missed its one-tick deadline and the carrying tick fell back to
   the synchronous path. Deliberately named apart from v1's
   `bg_exceed_fraction`, which counted **per tick**; the two are different
   quantities and must not be compared across versions.

4. **QUALITY — the pre-registered gate above stands.** Noise floor 1.0
   (measured); per-prompt agreement >= floor - 1.0pp; mean >= floor - 0.5pp;
   **median first-divergence >= 128** generated tokens (the binding line);
   mod8==0 clustering applies only to the delayed run, since the floor has no
   diverged prompt to cluster.

5. **RESULT — the gate that decides go.** Run **control -> v2 -> control**,
   each in its own process, on the SAME prompts, in the same window (the two
   controls are the placement control: **if they differ by > 5% the whole
   window is invalid**).

   The control is the current configuration (sparse graph + draft W=2 + W2048
   + `sparse_min_tokens=0`), i.e. the delayed design off. **The gate is the
   ratio:**

       v2 eff tok/s / same-window control eff tok/s >= 1.20

   **Both absolute values are reported**, alongside the historical reference
   24.9 tok/s. The ratio is the ruling: if the control deviates from 24.9 by
   more than 10%, then 29.9 is no longer used as the line, but the deviation
   **must be explained separately in the verdict and cannot be skipped** —
   otherwise a control that drifted for an unrelated reason silently re-sets
   the bar.

   Why the baseline is a separate committed artifact at all: it is the only
   cross-window anchor, and it is what the ratio is checked *against* for
   sanity, not what the gate divides by.

**The v1 background figures are a LOWER BOUND, not the v2 budget.** v1's
background was a synthetic load with zero side effects (no l2p writes, no
promotions, `carved` pages untouched); v2 writes the mapping and moves real
pages. So 23.15 / 55.60 / 68.50 ms are "at least this much", and a v2 carrying
tick above them is expected rather than a regression.

## Ceiling — even with refresh eliminated entirely, <= 39.8 tok/s (2026-09-24)

**The `45.5 tok/s` figure that circulated for "refresh excluded" is a
read-out error and is withdrawn.** It divided all 5957 warm tokens by the
*2862 graph ticks'* time alone (`2862 x 45.749 ms`) — keeping in the numerator
the tokens the 408 eager ticks produced while charging those ticks **zero ms**.
Eliminating refresh does not make those ticks free; it makes them graph ticks.

The correct figure charges them the graph price:

    5957 / (3270 x 45.749 ms) = 39.82 tok/s

**And that is itself an over-estimate**, because it multiplies a *p50*.
`2862 x 45.749 + 408 x 210.987 = 217016 ms` against a measured warm wall of
**239094 ms** — a **22078 ms residual, 9.2% of the measured wall** (10.2% only
if divided by the model itself, which is the wrong denominator). Two p50s do not
reproduce a
sum; the mean tick is above the p50 for both classes (the tick distribution has
a right tail: the 9.9 s and 10.3 s ticks recorded in
`errors/2026-09-23-b1024-order-b-outside-graph.md`). So the real all-graph
ceiling is **at or below 39.8**.

**Consequence for this plan: 40 tok/s is not reachable through the refresh
design.** Reaching 40 at the current tick time would need **45.54 ms/tick** —
0.21 ms (0.5%) *faster* than today's p50 — and the whole refresh prize is worth
at most 39.8. The lever is the trunk graph itself: `p2_replay` is **28.219 ms of
the 42.721 ms graph envelope (66.1%)** at 32.9k, against `p5_draft` 11.891 ms
(27.8%), `p0_fill` 0.907, `p4_verify` 0.834, `p_rows` 0.497, `p1_h2d` 0.109,
`p3_finalize` 0.049 (`wins/2026-09-23-sparse-w2-phase-timing.md`, the
`w2_graph_dw2048` column = production shape). Those seven sum to 42.506 ms
against the 42.721 ms envelope, so the phases account for all but **0.215 ms
inside** the envelope. The separate **3.955 ms is `step − envelope`** — host
work *outside* the captured graph, a different object from the phases and not
to be listed beside them. Cutting `p2_replay` 20% removes 5.64 ms from the
whole 46.676 ms step: 41.03 ms, i.e. **1.8217 / 0.04103 ≈ 44.4 tok/s**; v2's
own contribution is bounded well below that.

What v2 is still for: it converts the 249.3 ms eager refresh tick into a graph
tick, which is the difference between the measured 24.9 and the <=39.8 ceiling.
It is a real gain; it is just not a route to 40.

## Device measurement order

- ANSWERED by the phase window (f7a93e5c/a6b6a511, no nsys needed — nsys
  2022.4.2.1 export is broken): the gap is the eager sparse trunk (97.6%),
  not host sync (0.87%). This picked the 1-tick-delay design over a longer
  refresh interval; see Mechanism.
- DONE (shadow v1, 0195bcce..24898a26). See "Shadow v1 — result" above.
- NEXT: v2 (real lagged-q + mapping), background launched only on refresh
  ticks. Binding gates for v2 are the quality gate above (floor 1.0;
  median first-divergence >= 128) **plus a measured 8-step cycle time**
  from the delayed configuration itself. v1's own no-go is v1's verdict,
  not a verdict on the design.
