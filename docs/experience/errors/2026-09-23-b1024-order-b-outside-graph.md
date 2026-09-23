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
