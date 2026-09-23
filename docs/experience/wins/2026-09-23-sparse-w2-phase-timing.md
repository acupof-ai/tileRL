# Draft read window 2048 removes 90% of the W=2 graph tick — V100 sm70, 2026-09-23

> Status: pending-remote

**First line of the conclusion: this window runs on `ad2d0495` (no #807 guard),
which is not the same configuration as the merged tree.** It answers the shape of
the phases only. The probe builds with `sparse_min_tokens=0`, so it measures the
**long-context graph**, not production's 8192 eager gate. The prime prompt is
synthetic (" Count aloud from one to forty" after filler `z` tokens) and therefore
**cannot price the draft window's acceptance cost** — see the Rule.

## Context

At W=2 the captured sparse decode graph was known to be slower than W=1 on the
same box (7–13 tok/s vs 11.6–14.9, `2026-09-23-sparse-w2-graph-keep-steps.md`),
but not *where* the step went. This window splits one steady graph tick into the
phases the product marks, to decide whether the cost is the trunk graph, the draft
forward, or host work.

Workload: qwen38-27b, V100 sm70, cmax buckets 512/1024/**2048** (**32887-token
prompt; 32.9–33.1k at measurement**), 50 steady graph ticks per cell, two run
orders (A and reversed B) so a position effect would show as an order split.

## What Worked

Three arms in one window, one process each:

- **w2_graph** — depth 1, draft read window 0 (full prefix).
- **w2_graph_dw2048** — same, but `load_draft(..., attn_window_tokens=2048)`.
- **w1_graph** — depth 0, no draft (the baseline; spec-free by construction).

**`p5_draft` is the whole story.** At bucket 2048, ORDER A:

| phase | w2_graph (W=0) | w2_graph_dw2048 | w1_graph |
|---|---:|---:|---:|
| p_rows | 0.513 | 0.497 | 0.471 |
| p0_fill | 0.934 | 0.907 | 0.856 |
| p1_h2d | 0.113 | 0.109 | 0.094 |
| **p2_replay** | **28.216** | **28.219** | 23.683 |
| p3_finalize | 0.053 | 0.049 | 0.045 |
| p4_verify | 0.850 | 0.834 | – (p_sample 0.353) |
| **p5_draft** | **119.526** | **11.891** | 0 |
| **graph envelope** | **150.501** | **42.721** | 25.558 |
| **whole step** | **154.456** | **46.676** | 28.993 |

Every phase moves ≤0.5 ms between the two w2 arms except `p5_draft`, which falls
**119.5 → 11.9 ms**. `p2_replay` is identical (28.216 vs 28.219), so the window
changes only what the draft reads, not the trunk graph.

Two checks that it is real:

- **`p5_draft` goes flat across buckets in the dw2048 arm** (11.734 / 11.650 /
  11.891) while `w2_graph`'s grows with context (30.733 / 60.651 / 119.526,
  ratio 3.89 against a 4x context ratio). A window that silently stayed at 0
  would look like the first column.
- **Closure holds on all 18 cells**: `|Σp_* − graph envelope| / envelope ≤ 5%`,
  p90 gap ≤ 0.22%. Every phase is host wall and `p2_replay`/`p5_draft` carry an
  in-phase `torch.cuda.synchronize()` fence (timing-only), so each phase drains
  its own async work and the sum returns to the envelope.

## Rule

**On the V100 sm70 27B long-context path, the W=2 draft forward's cost is its
unbounded read prefix: gating it to 2048 tokens takes `p5_draft` from 119.5 ms to
11.9 ms and the whole step from 154.5 ms to 46.7 ms at 32.9k.** Two caveats bind
this number:

1. **This window cannot price the acceptance cost.** Both w2 arms read
   `tok/fwd` = 1.977961 and 1.978022 — agreeing to four decimals, differing by
   6.1e-05 — because the synthetic counting prompt's next token is determined by
   the previous line and always sits inside 2048 tokens. That is a workload that
   cannot see the loss, not evidence the window is free. **`tok/fwd` is an
   accepted count, not a token identity**: under greedy verify speculation is
   lossless, so a window change moves *how many* drafts are accepted, not *which*
   tokens are committed. The registered sweep on real text measured ~2.4–3.4
   points of acceptance lost at W=2048; the end-to-end serving window with real
   37.6k prompts is what settles it.
2. **It is a steady-state step, not a server mean.** Prefill and refresh ticks
   (every `SPARSE_REFRESH_TICKS`) are excluded by construction.

## Results

Phase window probe `1d106e6a`, base PATCH `ad2d0495`, V100 sm70, qwen38-27b,
`sparse_min_tokens=0`, `sparse_k=128`, 50 steady graph ticks per cell, both run
orders, `TILERL_STEP_TIMING=1 TILERL_STEP_TIMING_SLOW_MS=0`.
`eff tok/s = tok/fwd × 1000 / step_p50`.

| arm | bucket | graph p50 ms | step p50 ms | step p90 ms | step−graph p50 | p2_replay | p5_draft | tok/fwd | eff tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w2_graph | 512 | 60.722 | 64.244 | 65.755 | 3.427 | 28.091 | 30.733 | 1.977961 | 30.79 |
| w2_graph | 1024 | 90.871 | 94.755 | 95.891 | 3.739 | 28.122 | 60.651 | 1.977961 | 20.87 |
| w2_graph | 2048 | 150.501 | 154.456 | 156.398 | 3.928 | 28.216 | 119.526 | 1.977961 | 12.81 |
| **w2_graph_dw2048** | 512 | 41.829 | 45.325 | 47.072 | 3.392 | 28.087 | **11.734** | 1.978022 | **43.64** |
| **w2_graph_dw2048** | 1024 | 41.942 | 45.630 | 46.571 | 3.585 | 28.108 | **11.650** | 1.978022 | **43.35** |
| **w2_graph_dw2048** | 2048 | 42.721 | 46.676 | 49.894 | 3.769 | 28.219 | **11.891** | 1.978022 | **42.38** |
| w1_graph | 512 | 24.880 | 28.163 | 29.734 | 3.251 | 23.505 | 0 | 1.000000 | 35.51 |
| w1_graph | 1024 | 25.035 | 28.364 | 29.134 | 3.306 | 23.506 | 0 | 1.000000 | 35.26 |
| w1_graph | 2048 | 25.558 | 28.993 | 30.320 | 3.377 | 23.683 | 0 | 1.000000 | 34.49 |

ORDER B reproduces every cell except b1024, where two arms deviate by ~42 ms of
outside-graph time — a separate reproducible defect, tracked in
[errors/2026-09-23-b1024-order-b-outside-graph.md](../errors/2026-09-23-b1024-order-b-outside-graph.md).
`tok/fwd` is identical between the two orders within each w2 arm (1.977961 for
`w2_graph`, 1.978022 for `w2_graph_dw2048`); the two arms differ from each other
by 6.1e-05.

`step − graph` is 3.3–3.9 ms across all arms (~8% of the dw2048 step). Reading it
against the per-tick log, `stats=3–4 ms` accounts for essentially all of it:
`_build_stats()` is called twice per tick.

Raw artifacts: `sparse-w2-phase-timing-2026-09-23/arm_A_w2_graph.json`,
`…/arm_A_w2_graph_dw2048.json`, `…/arm_A_w1_graph.json`, and the `arm_B_*`
counterparts, plus the aggregated `…/phase_w2_A.json` and `…/phase_w2_B.json`
(probe revision `1d106e6a`).
