# Draft decode trailing window W=2048 default — V100 sm70 sparse 27B, 2026-09-17

> Status: Shipped behind `DRAFT_ATTN_WINDOW_TOKENS_DEFAULT=2048`; live post-deploy 32k confirmation (~9 tok/s) pending.

## Context

On the V100 sm70 sparse 27B serve line, long-context decode was dominated by the
1-layer MTP draft head's dense full-prefix attention: it scaled linearly in
context (the #665 slopes 3.57/3.64 µs/token) and was the single largest 32k
decode term (~38%, ~101 ms GPU at 27.9k). The draft only needs recent context
to propose its next token. We gate its decode attention READ to a trailing
window of W tokens (#684) — write and full-prefix KV retention are untouched —
and choose a fixed production W from the W-sweep.

## What worked

Swept W = 0 (full prefix) / 1024 / 2048 / 4096 / 8192 in one process
(`scripts/probe_draft_window_sweep.py`), think-off spec_depth=1, disjoint
wikitext-103 prompts, NoPrefixStore. Absolute proof-engine tok/s at n=9 was
non-monotone and noisy on the long cells, so the decision uses the **on-#665
constant model** to translate measured per-stage costs into served tok/s:

| ctx | W=0 | W=1024 | **W=2048** | W=4096 | W=8192 |
|---|---:|---:|---:|---:|---:|
| 9k  | 9.18 | 10.31 (+12%) | **10.33 (+12.5%)** | 10.06 (+10%) | 9.28 (+1%) |
| 16k | 8.08 | 10.25 (+27%) | **10.19 (+26%)** | 10.02 (+24%) | 9.30 (+15%) |
| 32k | 5.93 | 9.05 (+53%) | **9.07 (+53%)** | 8.91 (+50%) | 8.28 (+40%) |

Draft step GPU ms (W=0 → W=2048): 33.7 → 11.2 (9k), 59 → 11.8 (16k),
117 → 12.1 (32k) — 0.20–0.33x. Spec acceptance at W=2048:
0.743 / 0.729 / 0.723 at 9k/16k/32k vs 0.755–0.757 at W=0, a drop of at most
~3.4 points.

**Why 2048 and not a claim that it is strictly best:** W=2048 and W=1024 are
**not statistically separable** on the 32k cell (n=9); their modeled speed is
within noise at all three lengths. 2048 is chosen as the **safety margin**, not
as a measured winner: acceptance is ~2.4 points higher than 1024, it is the best
arm at 9k, and one fixed value covers all three served lengths without per-length
tuning. A larger sample (below) decides whether 1024/2048 truly differ. 8192
leaves most of the gain on the table (the draft still reads too much); 4096 is
marginally slower. So: 1024≈2048>4096>8192 on speed, with 2048 the robust
single-value pick — not "2048 significantly beats 1024".

The default is a FIXED window, not adaptive. Set W=0 (serve flag
`--draft-attn-window-tokens 0` or env `TILERL_DRAFT_ATTN_WINDOW_TOKENS=0`) to
restore full-prefix behavior.

## Rule

A dense draft head's decode attention can be capped to a small trailing window
with near-full-prefix acceptance: on this 27B/MTP1 sm70 line a W in the
1024–2048 range cuts the draft step 3–10x and lifts long-context decode up to
~1.5x at ~3 points of acceptance. 1024 and 2048 were not separable at n=9, so
the fixed default takes 2048 as the safety margin (higher acceptance, one value
for all lengths), pending a larger sample. Decide W from acceptance + a cost
model, not absolute tok/s on a small-n long-context cell.

## Follow-up

- Re-run 32k at **n≥30** with the wikitext **train** split (the test split caps
  disjoint 32k spans at 9) to confirm W=1024 and W=2048 do not diverge before the
  fixed 2048 default is treated as settled. Tracked as post-merge device work; CLI
  (`--draft-attn-window-tokens`) and env (`TILERL_DRAFT_ATTN_WINDOW_TOKENS`) keep
  the window overridable, 0 = full prefix.
- **Rollout scope (OPD/GRPO).** The module default also applies to the draft head
  used in OPD/GRPO rollout, not just serving. This is read-only: the window
  changes what the draft attends to when proposing, but the trunk verifies every
  proposed token one at a time, so on-policy token distribution is not rewritten
  (a rejected draft is just a rejected draft) and no stored KV changes. If any
  training run needs the pre-window read, set the env to 0. A targeted rollout
  acceptance re-check is still cheap insurance and is left to the training line.

## Confidence / caveats

- The 9k cell had n=30 independent prompts; 16k n=18 and 32k **n=9** (the
  wikitext-103 test split is only ~297k tokens, so disjoint 32k spans cap at 9).
  Medians at n=9 are directionally strong (53%) but the 32k acceptance spread is
  about ±0.04 — that is precisely why W=1024 vs W=2048 is called inseparable and
  2048 is a margin, not a winner. Live post-deploy 32k (~9 tok/s) is the
  confirmation, not this table.
- Served-tok/s is **modeled** from the #665 constants; raw measured per-stage
  costs (draft ms, acceptance) are direct CUDA-event/stats measurements.
- Acceptance is encyclopedic prose (wikitext); chat/code workloads accept
  differently and may need their own W. v1 is fixed, not per-workload.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-17 | pending merge | V100 sm70 | sparse serve d1, W=2048 vs 0 | 27B + MTP1 | unchanged (read-only) | draft step 33.7/59/117→11.2/11.8/12.1 ms | modeled 10.33/10.19/9.07 (9.18/8.08/5.93 W=0) |

Raw artifacts (vendored this PR under the wins data dir, alongside
`bench-baseline.json`; the repo `.gitignore` excludes top-level `runs/`):
decision note and modeled table
`draft-window-2048-2026-09-17/decision-note.md`, probe aggregates
`draft-window-2048-2026-09-17/resume-16k-32k.json` (n=18/9, script JSON),
`draft-window-2048-2026-09-17/recovered-9k.json` (n=30; recovered from the
crashed full-table stdout — JSON was never written, provenance recorded in the
file); raw V100 log `/tmp/w_fulltable.log`; sweep `scripts/probe_draft_window_sweep.py`;
wiring #684, corpus fix #696.
