# b1024 / ORDER B costs ~38 ms outside the graph, reproducibly — V100 sm70, 2026-09-23

> Status: open

## Context

Reading a 19-cell phase-attribution window (`#805` W=2 captured sparse decode),
one cell pairing misbehaved in both run orders' second and third arms: cmax
bucket **1024**, **ORDER B**. The window is the third to see it, so it is a
defect to track rather than a noise note in a wins entry.

Measurand for this entry is **`step − graph`**: the whole `engine.step()` wall
minus the product `graph` envelope that brackets `run_decode_graph`
(`src/tilerl/engine.py`, `_run_fwd` graph branch). Both come from the same
`_StepTiming` object, `step` from `last_total` and `graph` from the per-tick
mark, so the difference is the host work outside the captured forward.

## What worked

Nothing yet — this is the evidence that it is real and not jitter.

In the phase window, two cells deviated, both at b1024/ORDER B, in **two
different arms**:

| order | arm | b1024 graph ms | step ms | step − graph ms | p_rows | p0_fill | p4_verify |
|---|---|---:|---:|---:|---:|---:|---:|
| A | w2_graph | 90.871 | 94.755 | 3.739 | 0.254 | 0.821 | 0.816 |
| **B** | w2_graph | **125.952** | **167.789** | **41.909** | 2.668 | 5.328 | 2.812 |
| A | w2_graph_dw2048 | 41.942 | 45.630 | 3.585 | 0.251 | 0.799 | 0.828 |
| **B** | w2_graph_dw2048 | **83.368** | **125.809** | **41.911** | 2.336 | 5.541 | 2.860 |

Three things make this a defect and not noise:

1. **Two different arms give 41.909 and 41.911 ms** — agreement to three
   decimals. Independent draws do not do that.
2. **16 of 18 cells agree between orders**; only these two do not.
3. **It reappears in three separate windows.** The earlier ones are recorded in
   `wins/2026-09-23-sparse-w2-graph-keep-steps.md` (control-B b1024 W=1 graph
   250.8 ms/tick vs 86.8/85.1 in the same arm; patch-B b1024 W=2 eager 974.7 vs
   237.7 in ORDER A) and in the parity window that preceded it.

Not the allocator: the per-tick allocator event counts are **identical** between
orders (12 `alloc_reclaim`, 53 `dev_malloc`, 1732 `gpu_drain`; minimum `free=`
2884 MiB in both). The extra time lands in host phases that scale together —
`p_rows`, `p0_fill` and `p4_verify` all rise by roughly 3–5x in the deviating
cells — not in one allocator-backed phase.

## Rule

**The b1024 / ORDER B combination can cost an extra ~38 ms per step outside the
graph. Do not quote b1024 latencies without re-measuring both orders, and do not
read the pairing as machine jitter: it has reproduced across three windows and
across measurands.** Mechanism undetermined, so no fix is named yet.

## Results

Phase window `1d106e6a`, base PATCH `ad2d0495` (no #807 guard), V100 sm70,
qwen38-27b, `sparse_min_tokens=0`, 50 steady graph ticks per cell, closure gate
`|Σp_* − graph| / graph ≤ 5%` green on all 18 cells (p90 gap ≤ 0.22%).

| date | commit | machine | bucket | order | step − graph p50 ms | n |
|---|---|---|---:|---|---:|---:|
| 2026-09-23 | `1d106e6a` | V100 sm70 | 1024 | A | 3.739 / 3.585 | 50 ×2 |
| 2026-09-23 | `1d106e6a` | V100 sm70 | 1024 | **B** | **41.909 / 41.911** | 50 ×2 |

Raw artifacts: `wins/sparse-w2-phase-timing-2026-09-23/arm_B_w2_graph.json`,
`…/arm_B_w2_graph_dw2048.json`, `…/arm_A_w2_graph.json`,
`…/arm_A_w2_graph_dw2048.json` (probe revision `1d106e6a`).

## Second sample, 2026-09-24 — a different measurand, recorded not concluded

The refresh-phase window (`probe/805-serve-sm70` @ `f7a93e5c`, V100 sm70,
qwen38-27b, `graph_w2048`, one real 37.6k wikitext prompt,
`TILERL_REFRESH_PHASES=1`, no nsys, dir `phasewin-0924-025040`) contains one
graph-path tick that stalled on the storage tier:

| tick | step ms | graph ms | `step − graph` ms | ssd_mmap ms | note |
|---:|---:|---:|---:|---:|---|
| 197 (last of run) | 9904 | 9901 | 3 | 8031 | `release_cold_forget=9870`, `pub_cold_transfer=7771`, `pub_bounds_d2h=309`, `pub_draft_clone=348`, `pub_frame_d2h=278`, `pub_share_hold=383` |

Every other graph tick in the same run reads 37–46 ms. The segments above are
that tick's own phase marks and overlap; they do not sum to the total.

**This is not a sample of the b1024 / ORDER B phenomenon.** Two differences, both
load-bearing:

1. **Opposite measurand.** b1024/ORDER B is defined by `step − graph` ≈ 41.9 ms
   with the graph itself short. Here `step − graph` is 3 ms: 9901 of the 9904 ms
   is inside the graph envelope, so the cost is on the captured path, not host
   work outside it.
2. **Different bucket and different position.** The window ran one 37.6k prompt
   at the steady bucket (`own_w=8`, `table_w=137`), not bucket 1024, and tick 197
   is the run's final tick — the request's departure-release path, not steady
   state. Its phase marks are dominated by `release_cold_forget` and
   `pub_cold_transfer`, which the b1024 cells do not show.

Recorded because an unexplained ~9.9 s sparse graph tick is worth knowing about,
not because it corroborates anything. One sample, no mechanism, no fix. A third
distinct measurement on this path would be worth its own entry.

### A second, independent data point (2026-09-24)

The shadow-v1 go/no-go window (`probe/805-serve-sm70` @ `0195bcce`, V100 sm70,
`graph_w2048`, `TILERL_SPARSE_SHADOW=quest`, one real 37.6k wikitext prompt,
dir `shadowwin-0924-051849`) hit the same shape on its last tick before the
window aborted on an unrelated CUDA OOM:

| tick | step ms | graph ms | `step − graph` ms | ssd_mmap ms | note |
|---:|---:|---:|---:|---:|---|
| 730 (last of run) | 10303 | — | — | 8365 | `path=eager`; `release_cold_forget=10109`, `pub_cold_transfer=8069`, `pub_share_hold=404`, `pub_draft_clone=333`, `pub_bounds_d2h=295`, `sample=10110`, `model=167` |

Same order of magnitude (10.3 s vs 9.9 s), the same segments dominating, and the
same position — the run's **last** tick, on the request's departure-release
path — and again at the tail of a long prompt. Every other tick in that run was
42–46 ms (graph) / ~190–200 ms (eager refresh), and the two runs differ in
branch, env, and failure context, so the two are independent draws.

**One difference, stated because it matters:** the first sample was a `path=graph`
tick and this one is `path=eager` (`draft_step=0`, `sample=10110`). So the two do
not share a path; what they share is the storage traffic and the tail position.

**Still no conclusion.** Two samples on one machine, at the same position, is
not a mechanism: the tail path is where a request's cold pages are forgotten
(`release_cold_forget`) and where a spill file's final writes land, so a
storage-tier stall there may be ordinary end-of-request work rather than a
defect. What would discriminate is the same measurement on a prompt that ends
while pages are still hot, and a run long enough to show whether non-tail ticks
ever do this. Not claimed here.
