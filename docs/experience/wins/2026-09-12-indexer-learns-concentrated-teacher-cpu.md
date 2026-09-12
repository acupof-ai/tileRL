# Page-indexer warm-up learns a concentrated teacher (CPU tiny) — cpu, 2026-09-12

> Status: Shipped (card-free; answers whether the V4.1 indexer warm-up recipe
> is learnable at all before another 27B card)

## Context

The 27B indexer warm-up made oracle recall WORSE (0.28 → 0.10) while KL moved
49.9 → 6.9 toward a diffuse, near-uniform dense page-mass teacher. Two
explanations were open: the recipe (KL-to-page-mass loss / heads-pooled
target / tape backward / AdamW) is broken, or the 27B teacher carries no
concentrated structure to learn (diffuse mass). a3 ordered a card-free test:
give the SAME recipe a CONCENTRATED teacher it can represent, warm up 200
steps at lr 0.02 and lr/10, and read held recall@k before/after vs oracle.

Synthetic "fixed random projection teacher" (`scripts/indexer_learnability.py`)
that drives the REAL `indexer_warmup_loss`, the real hand-written tape
backward, the real `AdamW` and the real `topk_page_recall`:

- one fixed teacher `(iq_star,ik_star)` shared across batches = the frozen
  base's fixed H/K → dense-mass map;
- each step: fresh random H `[1,L,q,hidden]` and page-K `[1,L,p,ih,dkv]`;
- teacher per-query scores are the real `page_index_scores` (sum_h ReLU(q·k) /
  √di) with iq_star/ik_star, concentrated to a sharp softmax (sharpness 6) on
  each query's top-1 page. `h_attn=ih` so `ik_weight=ik_star` reproduces every
  projected key exactly — the target is representable, no model-error confound;
- train on fresh batches; evaluate on n=8 HELD unseen batches × 4 layers × 32
  queries = 1024 held (layer,query) points per seed, which can only be fit by
  recovering the shared teacher.

Tiny dims: hidden 64, ih 4, dkv 16, di 16, 64 pages (56 indexable past the
8-page window), 200 AdamW steps, init scale 0.1 (the production init).

## What Worked

The recipe learns a concentrated teacher and generalizes. Held-set, 3 seeds:

| seed | lr | recall@1 before→after | recall@4 before→after | KL before→after |
|---|---|---|---|---|
| 0 | 0.02   | 0.0127 → 0.0635 | 0.0830 → 0.2412 | 4.024 → 0.785 |
| 0 | 0.002  | 0.0127 → 0.0596 | 0.0830 → 0.1846 | 4.024 → 2.430 |
| 1 | 0.02   | 0.0234 → 0.0664 | 0.0791 → 0.2324 | 4.037 → 0.731 |
| 1 | 0.002  | 0.0234 → 0.0654 | 0.0791 → 0.2012 | 4.037 → 2.375 |
| 2 | 0.02   | 0.0293 → 0.0625 | 0.1035 → 0.2080 | 4.027 → 0.816 |
| 2 | 0.002  | 0.0293 → 0.0615 | 0.1035 → 0.1836 | 4.027 → 2.392 |

At lr=0.02 held KL falls 4.02 (≈ ln 56, uniform) to ~0.77 and held recall@4
roughly triples; lr/10 moves the same way but stops at KL ~2.4, as expected.
Direction is identical across all three seeds; absolute recall is low because
each layer's 32 queries point at different top-1 pages while the selector
picks k pages SHARED across queries (recall@k's ceiling is the per-layer modal
page, ~k/56), so recall rises but cannot approach 1 here — KL is the
unconfounded fit statistic and it moves decisively.

A planted shared-needle variant (one page K aligned to a common query
direction) was tried and rejected as a metric: no magnitude regime gives both
chance-level init recall AND a consistent teacher argmax — a dominant teacher
needle is also selected by K-norm at the random init (before-recall already
0.9), so there is no head-room to attribute to learning. The per-query
concentrated teacher is the clean measurement and is what the production KL
target actually pools.

## Rule

The V4.1 warm-up recipe (page-mass KL target, heads-pooled bilinear indexer,
tape backward, AdamW lr 0.02) is learnable: on a concentrated, representable
teacher it fits and generalizes on held batches at both lr and lr/10. The 27B
recall getting worse is therefore not a broken loss/target — it is the teacher
(diffuse full-attention mass has little per-page signal to recover). The
indexer unit parks until k and output fidelity are settled; do not spend a
card re-tuning warm-up lr/loss against diffuse mass. Related:
[[attention-mass-is-diffuse-on-the-27b-full-attn-planes]].

Raw artifacts: `/tmp/final0.json`, `/tmp/fin1.json`, `/tmp/fin2.json`
(`scripts/indexer_learnability.py --seed {0,1,2} --steps 200`).
