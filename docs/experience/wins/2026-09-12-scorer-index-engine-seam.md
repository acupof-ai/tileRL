# scorer="index" through the engine sparse seam — CPU tiny, 2026-09-12

> Status: **pending-remote** — full-k==dense equivalence and the fp8 index-key
> ledger are CPU-tested on tiny; the sm90 cell and the served 27B recall of the
> live selection are the card follow-ups.

## Context

Unit F (#518) wired the training-free Quest **bounds** scorer into the engine:
per-page kmin/kmax stored at append, `select_pages` over the scores, a packed
`[selected; own]` block table. This PR adds the learned **index** scorer
(`scorer="index"`) through the SAME seam — V4.1 page indexer keys projected
from each complete page's mean K, scored against the source layer's live
indexer-Q — so the trained indexer from unit D selects in serving. `select_pages`
and the attention kernel are unchanged.

## What worked

- `SparseTracker(scorer="index")` keeps per-page fp8 keys `[n_src, ih, di]` plus
  one f32 scale per key (the ledger's `index_keys` face), projected once at
  finalize from the page's mean K via `ik_weight`; `iq`/`ik` move to the backend
  device after materialize. Keys, like bounds, live independently of the KV
  pool and survive demotion, so scoring a cold page never reads its K back.
- `SparseForward._select` branches on scorer: bounds scores post-rope Q against
  kmin/kmax; index scores the source plane's layer input H (`h` is now passed to
  `attention_args`) against the dequantized stored keys with the V4.1
  ReLU(q·k)/√di score. Group-mates reuse the cached decision.
- **full-k == dense, token for token for token, through the real engine**, for
  the index scorer (untrained weights): `test_index_scorer_equals_dense_token_for_token_at_full_k`.
  Equality is only at full k with an untrained indexer — the meaningful shipped
  claim is the trained indexer's recall@128 (the #512/card run).
- The live served selection is retained per req/group after each forward, and
  `Engine.sparse_selection_recall(rid, target_mass)` reports recall of what the
  engine ACTUALLY selected against an offline dense top-k — the card run can
  report served selection recall, not only the offline teacher. Full-k gate is 1.0.
- The `index_keys` ledger row reconciles **derived == measured on tiny**
  (di=16 → 20 B/key), via `index_keys_bytes(cfg, pages, di)`; the shipped di=128
  face stays 132 B/key, 33.0 MiB at 256k (the existing 27B ledger gate).

ih is `min(4, num_kv_heads)` so the 27B (4 KV heads) runs the shipped 4-head
form while the tiny cell (2 KV heads) runs ih=2.

## Rule

A learned scorer must reuse the bounds scorer's whole seam — same append-time
per-page store, same cold-survival, same `select_pages`, same packed table —
changing only what scores a page and where its queries come from. The
full-k==dense token gate and a derived==measured scorer-storage row apply to
every scorer, and live-selection recall is exposed on the Engine so the card
run measures the served selection, not a recomputation.

## Results

| date | commit | machine | target | model | result |
|---|---|---|---|---|---|
| 2026-09-12 | (PR head) | Mac CPU | cpu f32 | tiny | index full-k == dense token-for-token prefill+decode; live-selection recall 1.0 at full k; index_keys derived==measured (di=16, 20 B/key); 6/6 sparse-engine gates, 44+1x sparse/memory/docs |

Raw artifacts: `src/tilerl/sparse_engine.py`, `src/tilerl/engine.py`,
`src/tilerl/memory.py`, `tests/test_sparse_engine.py`. sm90 cell and 27B served
recall pending cards.
