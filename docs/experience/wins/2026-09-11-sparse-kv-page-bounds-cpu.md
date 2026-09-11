# Sparse KV day 1: Quest page-bounds scorer and page selector — CPU, 2026-09-11

> Status: **CPU ops + gates green** (`tests/test_sparse_kv.py`). The sm70 scorer/
> selector kernels, `serve --sparse-k` wiring, and the 128k/256k V100/H20
> measurements are the card half (design #486).

## Context

Long-context inference on one small card needs attention over a *selected*
subset of KV: the device holds weights, a per-page index, and the selected hot
pages; the cold pages live in host RAM. The day-1 selector is training-free
Quest: bound each page by the elementwise min/max of its K, score pages by the
upper bound the bound implies for the query, and attend to the top-k pages.
`paged_attention` already gathers through a block table, so selecting is
producing a shorter table — the attention kernel does not change.

## What worked

Three pure f32 ops in `tilerl_kernels.reference`, exposed on `RefBackend` as
the CPU parity oracle:

- `page_bounds(k)` — a layer/row's gathered pages `[P,Hkv,16,D]` →
  `[P,Hkv,2,D]` (stacked kmin, kmax), elementwise over the 16 tokens. fp16 on
  the cards, f32 on the CPU cell (input dtype passes through).
- `page_bound_scores(q, bounds)` — `sum_d max(q·kmin, q·kmax)` per page per
  KV head, summed over query positions: the Quest upper bound on the logit.
- `select_pages(block_table, n_pages, scores, k_pages)` — top-`k_pages` set
  per row per layer, **returned in sequence order**.

The sequence-order point is load-bearing and easy to get wrong: the score
chooses the SET, but `paged_attention` names a token's absolute position from
the table's order and applies a causal mask. A score-descended table would
mis-position every page. The op takes `torch.topk` indices and sorts them back
to original page position before gathering ids.

## Gates (`tests/test_sparse_kv.py`, CPU)

- `test_select_at_full_k_is_the_dense_table_in_sequence_order` — k ≥ pages
  returns the dense table in its original order, not score order.
- `test_sparse_attention_equals_dense_at_full_k` — the REAL
  `RefBackend.paged_attention` fed the full-k selected table returns the exact
  same output and argmax as the dense table (a single-layer paged plane with
  pages at block ids 1..P). This is the `k >= context` correctness gate.
- `test_recall_mass_of_quarter_selected_pages_is_stated` — at k = pages/4 the
  Quest-selected pages hold **more** dense causal-attention mass (last decode
  query, summed/averaged over heads) than both a uniform quarter and the
  bottom-scored quarter. Measured on the random-Gaussian fixture (seed 7):
  selected **0.269** vs bottom quarter **0.238**, uniform **0.250**.

## What the recall number does and does not claim

Random Gaussian KV with one decode query has diffuse attention mass, so the
lift over uniform is small (0.269 vs 0.25). The gate asserts the DIRECTION —
the scorer beats uniform and an anti-scorer — which is what pins that the op
actually uses q and the bounds. The design's acceptance of **X ≥ 0.9 recall at
k=2048 on 128k held-out prompts** is a property of the real 27B's peaked
attention and is a card measurement, not something diffuse random KV can
license. Asserting 0.9 here would be the tolerance-chosen-to-pass failure.

## Rule

Page selection chooses a SET by score but must present pages in SEQUENCE order
to a kernel that derives causal positions from table order. Gate sparse
attention at full k for byte-equality with the dense paged path (the kernel is
unchanged), and gate the scorer's direction on diffuse random KV while leaving
the absolute recall number to the peaked real model on a card.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-11 | local | cpu | 3 gates green; quarter recall 0.269 > 0.250/0.238 |
| pending | V100/H20 | sm70/sm90 | scorer+selector kernels, serve --sparse-k, 128k/256k tokens/s + bytes |
