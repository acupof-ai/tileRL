# The index-2 divergence is a near-tie argmax flip — V100 sm70, 2026-09-24

> Status: measured; mechanism a hypothesis with n=2

## Context

The #805 production cutover window compared three arms on eight prompts and its
correctness gate (⑤) came back red: the whole config change was blocked. This
entry is the diagnostic 94 ordered to find out whether the divergence was noise
or a real computational difference, and it changes what the gate can be.

Setup: prompt 0 of the cutover set (37600 real tokens), 5 verify ticks, three
arms carrying the cutover build's knobs, per-generated-index top-5 logits for the
trunk's verify output plus the draft chain. Artifacts:
`near-tie-flip-2026-09-24/step_{baseline,ref_eager_w2048,graph_w2048}.json`
(written by `scripts/probe_step_margin.py`, which reads the engine's existing
`_keep_draft_logits` / `_trunk_logits` / `_verify_chains` seam).

Arms: `baseline` = min8192, no graph; `ref_eager_w2048` = min0, forced eager, no
graph; `graph_w2048` = min0, captured replay.

## What worked

The diagnostic separates the two explanations on its first reading.

**Margin at the position that diverged** (generated index 2, second chain
position; the cutover window's prompt 0 gave graph=`9593` / ref=`19312`):

| arm | top1 | top2 | margin |
|---|---|---:|---:|
| baseline (no graph) | 9593 (15.1985) | 19312 (15.1654) | **0.0331** |
| ref (no graph) | 19312 (15.3730) | 9593 (15.2405) | **0.1325** |
| graph (replayed) | 9593 (15.5521) | 19312 (15.1437) | **0.4084** |

All three arms put the decision inside the **same two tokens**, a few tenths of
a logit apart, and order them differently. The first chain position at the same
index repeats it: margins 0.1995 / 0.0587 / 0.0169, same pair (1048 vs 551).

**The control that makes it a reading.** At generated index 1 the same three arms
agree — all pick token 279 — and their margins are 0.5357 / 0.5288 / 0.6289, the
largest recorded. Cross-arm spread at index 1 is 0.2530, i.e. *comparable to
index 2's*, so the spread is not smaller where they agree. What differs is the
margin. Large margin → agreement; small margin → split, on one prompt in one run.

Consequence: the two non-graph arms flip on the same pair, so **the graph arm's
red is not attributable to the captured graph**, and ⑤'s criterion is not
reachable — see below.

## What the numbers do and do not support

- **Supported:** same-token cross-arm logit spread at index 2 is 0.23–0.60, while
  the within-arm top1−top2 margin there is 0.02–0.41. A flip happens when spread
  exceeds margin, which these numbers do.
- **Hypothesis, not measurement (n=2 indexes, one prompt):** 0.23–0.60 is far
  above f32 reduction-order variation (~1e-3), so it most likely reflects the
  **sparse page selection** differing between configs — prefill chunking (hybrid
  caps sparse prefill at 192 tokens/tick, so a 37600-token prompt takes 196
  prefill ticks against 74 when the cap is absent and the general
  `max_num_batched_tokens` of 512 applies) and refresh cadence change which
  k pages are resident. That would make it the variance of the sparse
  approximation itself rather than numerical noise. Stated as a hypothesis
  because there are only two pre-divergence indexes available from one prompt.
- **Not supported, and easy to get wrong:** indexes ≥3 are **post-divergence**.
  From there the arms generate from different contexts, so "the logit at index 5
  in arm A vs arm B" compares two different prefixes. The larger numbers from
  those indexes (0.85, 2.37, 7.77) are not measurements of the same quantity and
  must not be quoted as spread.
- Also: an earlier version of this table averaged over the **union** of each
  arm's top-5, which differs per arm; part of that spread was two arms ranking
  different tokens rather than one token's logit moving. The strict version
  (tokens present in both arms' top-5) is what is quoted above.

## ⑤'s criterion is not achievable on this stack

The cutover gate required the graph arm's free-run output to be **identical** to
the forced-eager reference for ≥128 generated tokens at temp 0. With cross-arm
spreads around a third of a logit and margins dipping to 0.02, a first
divergence inside the first few tokens is the expected outcome at any level of
kernel correctness. Exact token identity is therefore not a gate this stack can
pass, and a gate of that shape will report red on a correct implementation.
What it would need: a tolerance on the margin (compare argmax only where the
margin exceeds the spread), or a comparison that holds the selection path fixed.

The same reading applies to the **09-17 first-replay value gate**, cited in
`_sparse_capture_allowed`'s comment as the verification that the sm70 fix is
correct. That gate is also exact token equality at temp 0. It passed 6/6 there,
which is evidence of something, but this diagnostic shows such a gate can flip on
a 0.03-logit margin, so **6/6 should not be read as bit-identity**.

## Next

A teacher-forced window (pending perf1's utility, in rev) with three arms —
graph(min0) and baseline(min8192) each teacher-forcing ref(min0 eager)'s
free-run output. That gives, per position on the **same token**:
`graph vs ref` (does the graph add error) and `baseline vs ref` (the sparse
approximation variance from prefill chunking/cadence). Binding gate stays
`graph vs ref` top1 ≥ 0.99, with ref teacher-forcing itself = 1.0 as the floor.
It doubles as the measurement that would promote the hypothesis above.
