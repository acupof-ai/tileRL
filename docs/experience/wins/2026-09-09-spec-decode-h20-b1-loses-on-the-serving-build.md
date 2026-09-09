# Speculative decode at B=1 on H20 — a small loss on the serving build, a win only against eager's host overhead

**Date:** 2026-09-09
**Arch:** H20 (sm90), 27B NVFP4 + MTP draft, B=1, `scripts/bench_batch_decode.py`, 30 timed ticks after 8 warmup
**Task:** fill the `pending-remote` gap in `2026-08-29-spec-decode-engine.md` — H20 B=1 spec had never been measured

> Status: Shipped

## Context

The README's headline 92.4 tok/s (B=1, H20, NVFP4+FP8) has no speculative decode, and every
depth/accept number in the CHANGELOG is V100 (sm70). The V100 result — depth 1 beats the
shipped depth 3 by 1.25x — turns on the verify-width ladder (`LADDER_WIDTHS = (1,2,4,8,32)`),
and H20's rung prices were unmeasured. The one metric: committed tokens per tick against the
tick's cost.

## What Worked

Four arms per build — no-draft baseline, depth 1, 2, 3 — same card, same minute, same prompts
(seed 7), each arm its own process and its own card claim. The baseline is each build's own
no-draft arm, not the README number, so every ratio is paired.

**Serving build (fused projections, decode graph on — the build that produces the README's 92.4):**

| arm | ms/tick | tok/tick | accept | tok/s | vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 10.569 | 1.00 | — | 94.6 | 1.00x |
| depth 1 | 20.718 | 1.93 | 93.3% | 93.3 | **0.986x** |
| depth 2 | 30.629 | 2.50 | 75.0% | 81.6 | 0.863x |
| depth 3 | 41.883 | 2.27 | 43.7% | 54.1 | 0.572x |

Spec loses at every depth. Depth 1's verify tick costs 1.96x the decode tick (20.72 vs 10.57 ms)
and buys 1.93x the tokens (93.3% acceptance) — a 1.4% net loss. Deeper depths pay the rung-4/8
verify prices for acceptance that falls with depth (93.3 / 75.0 / 43.7%), so the loss widens.
The ladder is the mechanism, same as the V100: at B=1, depth 1's W=2 verify sits on rung 2
alone, depth 2/3's W=3/4 land on rung 4. What does not transfer is the *price*: on the V100 the
rung step was +60% of a tick; on the H20 graph build the W=2 verify is ~2x the W=1 decode.

**Eager build (no fuse, no graph — the script's defaults, and the build the `pending-remote`
entry's command runs):**

| arm | ms/tick | tok/tick | accept | tok/s | vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 66.369 | 1.00 | — | 15.1 | 1.00x |
| depth 1 | 68.001 | 1.80 | 80.0% | 26.5 | **1.756x** |
| depth 2 | 79.994 | 2.20 | 61.0% | 27.5 | **1.824x** |
| depth 3 | 99.471 | 2.70 | 57.3% | 27.1 | **1.797x** |

The same engine, same model, same card, minutes apart: the eager tick is 6.3x the graph tick
(66.4 vs 10.57 ms). On that tick, depth is nearly free — depth 1 adds 1.6 ms, depth 2 13.6 ms,
and every depth wins 1.76-1.82x. The graph build removes that ~56 ms, exposes the GPU width
cost, and the win reverses. **A spec win measured on an eager tick is a win over host overhead,
not over the decode tick.**

The 14.7 tok/s eager baseline vs the README's 92.4 is the same gap: build, not contention, not
the model. The serving-build baseline (94.6) lands on the README's 92.4 and the 2026-09-06
snapshot's 94.3.

**Two things left open, recorded not explained:**

- Acceptance differs by build: 80.0% (eager) vs 93.3% (graph) at depth 1, each stable across
  re-runs (93.3% twice). Both paths call the same `_draft.step`; the cause is unread. It does
  not move the verdict — at 80% the serving-build depth 1 would be 0.90x, still a loss.
- The first pass compiled 4-5 kernels per fresh process, and the deeper arms' W=3/W=4 shapes
  compiled inside their timed windows: the cold eager depth-2/3 arms read 169/201 ms/tick
  against 80/99 ms warm, and the cold graph depth-3 arm read 137.5 against 41.9. The tables
  above are the warm-cache re-runs (0 compiles, verified per arm); the cold logs are kept for
  the record.

Prompts are 16 random token ids, not text: the paired comparison is fair, but the absolute
acceptance is not a serving-traffic claim. n=1 per arm, 30 ticks.

## Rule

On the H20 serving build, B=1 speculative decode is a net loss at every depth (depth 1: 0.986x;
depth 2: 0.863x; depth 3: 0.572x). Break-even needs the W=2 verify under 2x the W=1 decode
tick; H20 pays ~2x. Do not enable spec for B=1 serving on H20. The V100's depth-1 win does not
transfer — it was priced on a different ladder, and an eager-tick spec win is priced against
host overhead the serving build does not have.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-09 | 53d349a | H20 card 6 | cuda/sm90 fused+graph | Qwen3.8-27B-NVFP4 | — | 10.569 (B=1) | 94.6 baseline / 93.3 d1 / 81.6 d2 / 54.1 d3 |
| 2026-09-09 | 53d349a | H20 card 6 | cuda/sm90 eager | Qwen3.8-27B-NVFP4 | — | 66.369 (B=1) | 15.1 baseline / 26.5 d1 / 27.5 d2 / 27.1 d3 |

Raw artifacts: `/work/specg2{base,d1,d2,d3}.log` and `/work/spece{base,d1,d2,d3}.log` on the
pod (warm-cache runs); the cold first-pass logs (`/work/specfd3.log` et al.) are kept for the
record.
