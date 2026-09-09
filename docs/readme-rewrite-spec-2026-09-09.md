# README rewrite spec — 2026-09-09

Deliverable: the table. README itself is NOT touched in this spec — 27 reviews
first. All numbers read off origin/main @ `5896bc7`.

**Status: approved by 27 on 2026-09-09, with three corrections (applied):**
the prefill-regression leg retracted (see below), the baseline-row disposition
is delete-not-fix, and the MMLU disposition is write-it-fully.

## Why the headline changes

`docs/bench-metrics.json` derives its weights from a real agent turn
(`wins/2026-09-07-a-claude-code-turn-is-314-seconds-of-prefill.md`): prefill is
294/314.33 s = **94%** of a turn (w=0.94), decode is 20/314.33 s = **6%**
(w=0.06). The README's headline was decode — the 6% quantity.

**The argument has one leg, and it needs no measurement newer than the
denominator:** prefill is 94% of a real agent turn, and the README led with the
6% quantity. (An earlier draft also claimed prefill had regressed 2.1x that
day. 27 retracted it on 2026-09-09: two independent instruments showed the
09657c0 reading undercounted its own prefill by half and HEAD is flat —
118.4 vs 123.8 ms/chunk, HEAD 4% faster. The real open regression is a
spec-arm-only ~4% wall, unlocated, and it does not belong in the README.)

Rewrite direction: headline = prefill. The sglang comparison stays and stays
honest: **we lose prefill** (2689.8 vs 4022, fp8 arm) and win decode (92.4 vs
54.2). A README that leads with the 6% it wins and buries the 94% it loses
loses more than a number when the reader recomputes it.

Every number gets a weight tag and a provenance. The table-interpretation
sentence: `decode is 6% of a real agent turn; prefill is 94% (314.33 s turn,
wins/2026-09-07-...)`.

## The table

| README number | Current value | Provenance today | Post-rewrite path |
|---|---|---|---|
| Headline decode | 92.4 tok/s | `bench-baseline.json` decode-kv family — **the d512-b1 row itself had `commit: "unknown"`** | store: `decode_tok_s` (w=0.06), demoted into the table; baseline row deleted (below) |
| Headline prefill | 2689.8 tok/s | `bench-baseline.json` prefill rows, commits pinned (9e3836b) | store: `prefill_tok_s` (w=0.94), **promoted to headline** |
| sglang decode | 54.2 / 39.9 | external, 2026-08-28, not reproduced | keep, external/dated |
| sglang prefill | 2512 / 4022 | same | keep, state plainly that we lose |
| Spec on/off | 126.5 / 79.5 | `wins/2026-09-09-accspec-b1-repro-w8-block-drafter.md` | store: `decode_tok_s` with the build column (`eager+draft` vs `eager`) |
| Spec ratio | 1.591x / 0.928x | derived (126.5/79.5; the B=8 arm) | view: `spec_goodput_ratio`, never stored |
| Spec reproduction | 126.5w/122.4c, 79.5w/78.2c, 6.19/6.14 | same entry (card 6) | store: `decode_tok_s` + `warm.state` column |
| 135.5 | — | already exiled to its dated entry by the README itself | no action — the model handling |
| MMLU | 74.6% | a one-off eval; `errors/2026-09-03` shows the score is build-dependent (74.6% fused vs 74.2% unfused); `bench-baseline.json` carries a *different* MMLU number (81.0, mmlu-200 slice) | **write it fully: value + n + build + date + provenance, or don't write it** — the rewrite writes it fully |
| V100 | 50.0 / 46.3 | sm70 wins series (13 entries) + sm70 arena scripts | store: `decode_tok_s` sm70 rows (pending-remote) |
| RL accuracy | 89.6 → 94.8 | `wins/2026-09-04-the-thinking-cap.md` + runs/ ledger | store: `gsm8k_pct` (cc filling tonight) |
| RL tokens | 157,601 → 121,642 | same | store: `rollout_tokens` (mean, cc filling tonight) |
| tokens/correct | 351.8 → 256.6 | derived | view: `rollout_tokens / gsm8k_pct`, never stored |
| p=0.002, McNemar p=0.18 | statistics | prose | stay prose — not metrics |
| Transfer 22.0/18.9/22.9% | n=100 one-off | prose | stays prose with n stated |
| 96.6 / 92 / 96.4 | train-prompt solve rate | prose | stays prose (rejected as a metric, YAGNI) |

## The "?" column — which numbers have no provenance today

Every number traces **except two**:

1. **Five `bench-baseline.json` rows carried `commit: "unknown"`** — not one:
   `decode-kv/d512-b1` (the 92.4 headline row itself), d2048-b1, d32768-b1,
   d8192-b1, d8192-b8, all dated 2026-08-28. The values appear nowhere else in
   the tree, so the commits cannot be recovered — only guessed. **Disposition
   (27): delete the rows, do not fix them.** A deleted row is an honest gap; a
   guessed sha is forged provenance. The ≥0.97× gate survives on the pinned
   rows; the single-stream decode numbers rest on their dated entries until
   the store has them.
2. **74.6% MMLU is worse than unprovenanced — it is build-dependent.** The tree
   holds two MMLU numbers (74.6% on the 1000-slice fused arm, 81.0 on
   mmlu-200), and the errors entry records the score moving with
   `fuse_projections`. **Disposition (27, upgraded): write it fully — value + n
   + build + date + provenance — or don't write it.** The rewrite writes it
   fully: 746/1000, sample-draw, `fuse_projections=True`, `scripts/mmlu.py`,
   2026-09-03, with the unfused 74.2% beside it.

## What waits for the store

`tilerl bench --readme` (`bench_harness.py:_view_readme`) already exists. Tonight
the store has `train_step_tok_s` rows only; after cc's filling, `gsm8k_pct` and
`rollout_tokens` join. `decode_tok_s` / `prefill_tok_s` rows land when their
collectors run under B1 — until then the README table carries pending-remote
markers, not silent blanks.

## Two metrics never generated

`mmlu_pct` and `kernel_ms` have zero collectors. Both are weight 0.0, so
weighted coverage is unaffected; the README cites one of them (74.6%). Disposition:
write it fully with its population (above), do not build the collector — a
weight-0 metric does not earn an instrument.
