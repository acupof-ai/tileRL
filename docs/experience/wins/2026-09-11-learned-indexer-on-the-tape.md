# Learned page indexer learns on the hand-written tape — CPU tiny, 2026-09-11

> Status: **pending-remote** — the KL warm-up backward and recipe run on the
> tiny CPU model; the 27B card run that produces a recall number has not run.

## Context

Sparse-KV unit D part 2 (after the V4.1 scorer math, #496): the two indexer
projection weights have to be trained, and the project rule is no
`torch.autograd` / `torch.optim` in framework code — every backward is a
hand-written reverse on the tape. The warm-up objective fits the indexer to
the frozen base's dense attention:

    iq = H @ iq_weight ; ik = page-K(head-grouped) @ ik_weight
    loss = KL(dense causal attention mass, pooled per page
              || softmax(sum_h ReLU(iq·ik)/sqrt(di)))

Only `iq_weight` and `ik_weight` learn; H, page-K and the dense-mass target are
frozen-base inputs and receive no gradient.

## What worked

One tape op, `indexer_warmup`, with a pure-f32 torch reverse living beside the
scorer in `sparse_index.py` (not a backend kernel): softmax-cross-entropy delta
on the page axis → the pre-ReLU sign gate → the two projection einsums. The
selector's true `-inf` mask stays; the KL softmax keeps the part-1 `-1e9`
sentinel because `log_softmax(-inf)` has no finite central-difference
derivative. The reverse recomputes its activations from saved inputs, matching
the tape's recompute convention.

The frozen-base teacher is captured by an opt-in `Model.index_capture` list
(None on serving/training forwards) that records each source layer's input H
and post-rope Q/K on the dense path. Dense causal mass is averaged over query
heads (GQA repeat), pooled per 16-token page with the window excluded by
`page_mass_target`.

Gates (`tests/test_sparse_index.py`):
- f64 `gradcheck` through the loss AND an independent 24-coordinate central-
  difference check of the hand-written reverse, for BOTH weights
  (`test_warmup_bwd_gradcheck_on_both_projection_weights`);
- the op records exactly one tape entry and the resolved leaves are only the two
  weights — H/page-K/target produce no grad;
- one tape-driven `AdamW` step strictly lowers the frozen-batch loss;
- `tilerl train --recipe indexer-warmup` runs a step on tiny through the real
  CLI and the manifest gate passes.

On a RANDOM tiny model the dense teacher is near-uniform over indexable pages
(late-query target entropy 2.071 vs ln 8 = 2.079; random QK has no locality),
so the CPU gate is that the chain runs one finite step, NOT that KL halves —
learnability against a sharp, aligned teacher is the part-1
`test_warmup_drives_kl_down_on_a_fixed_batch` gate (0.40 → 0.00). Conflating the
two would tune the recipe to a teacher with no signal.

## Rule

A learned auxiliary head lands on the same hand-written tape as the base: one
recorded op, a reverse whose weight grads are f64-checked independently of
`gradcheck`, and frozen activations explicitly absent from the leaf set. A
near-uniform teacher cannot demonstrate learning; gate "the step runs" and
"the head can learn a sharp target" separately.

## Results

| date | commit | machine | target | model | result |
|---|---|---|---|---|---|
| 2026-09-11 | (PR head) | Mac CPU | cpu f32/f64 | tiny | 11/11 sparse gates; warm-up recipe one step finite, gate PASS; analytic bwd f64 gradcheck + FD rel err < 1e-5 |

Raw artifacts: `tests/test_sparse_index.py`; recipe manifest via
`tilerl train --recipe indexer-warmup`.

## Pre-registered 27B science run (H20, cards 6/7) — before launch

Fixed before the run; a miss is recorded with the token count, not retuned.

**Scope (corpus).** Prompts are 8192/16384/32768-token spans cut from the
pod's **Chinese Wikipedia** parquet (`/work/newdata/wiki/zh.parquet`). The
corpus has almost no single 8k+ document (token p50 ≈405, p99 ≈9.4k), so
consecutive articles are concatenated into a token stream and cut into
disjoint fixed-length spans (`scripts/prepare_indexer_corpus.py`, the
`scripts.corpus.long_doc_spans` construction). Articles are hashed into a
seeded train/held split BEFORE tokenising (10% held: 22,819 of 230,792
articles), so no held-out article token appears in a training span; span
counts held 32/16/8 and train 160/80/40 at 8k/16k/32k, held-out article ids
kept in the corpus manifest. Chinese is the relevant distribution (Qwen3.8 is
bilingual and ckl's traffic is Chinese-heavy), not a weaker claim; a 32k span
crosses article boundaries, so dense attention within a span concentrates on
the current article — the ≥0.9 number is the indexer on this mixture, not on
one coherent 32k document. Recall is reported separately per span length.
(An English-cosmo 8k cross-corpus control was planned but its raw source was
removed in the same /work cleanup that took the checkpoint; recorded, not
silently substituted.)

- Base Qwen3.8-27B-NVFP4 fully frozen (checkpoint at the tileRL-owned
  `/work/tilerl-ckpt`, sha256 c473512c…); trained tensors are ONLY the four
  source layers' two indexer projection weights (V4.1 form: page indexer-K,
  `sum_h ReLU(q·k)/sqrt(di)`, 128-token/8-page window excluded).
- A few hundred warm-up steps over the prepared train spans.
- Teacher: dense attention mass pooled per 16-token page at each source layer,
  streamed in O(T·block) (`dense_causal_page_mass`), never a [T,T] matrix.
  **Amendment 2026-09-11 (a3):** the full-position teacher is ~T² over 8k–32k
  spans and did not finish the held-before pass in an hour. The teacher now
  evaluates **256 seeded query positions per span** (`--q-samples 256
  --q-min-pos 2048`), uniform over positions ≥ 2048 so the selector has
  missable pages, positions fixed per span by `seed + 100003*group + index` so
  before/after and warm-up reuse the same rows. The sampled teacher equals the
  full teacher's rows at those positions (CPU gate, atol 1e-6); cost is
  O(256·T) per span. Threshold unchanged.
- Metric: `topk_page_recall` (this PR, tested f32) at `k_pages=128`, measured
  BEFORE warm-up and AFTER, on held-out prompts, per span length; plus the KL
  curve and total tokens seen. Reported per length: the mean over spans AND the
  per-span min (manifest `recall_detail`); the verdict gate stays the mean.
- **Accept: mean recall@128 after warm-up >= 0.9.** Below 0.9 is a science
  result, written down with the token count — no tolerance change, no rerun with
  a moved gate.
