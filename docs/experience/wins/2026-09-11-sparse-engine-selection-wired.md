# Sparse-KV selection wired through the engine (Unit F) — 2026-09-11

CPU first cut: `build_engine(sparse_k=K, scorer="bounds")` selects pages per
tick, demotes the rest to a pinned host tier, and serves 256k-class contexts on
the same paged-attention kernel with **no kernel change**. The learned indexer
scorer is the later PR; this is the training-free Quest path.

## What runs

- every complete 16-token page keeps Quest bounds (per full-attn plane, per KV
  head, a kmin/kmax pair in fp16), computed once from the K the pool holds after
  each tick and keyed by **logical** page index. Bounds live independently of
  the KV pool, so they stay device-resident when the page demotes — scoring a
  cold page never reads its K back;
- at every tick the row's OWN span is dense: on decode the trailing 8-page
  window, on a prefill chunk the pages the chunk writes (from
  `floor(prefill_from/16)`, so a chunk overlapping the prior chunk's last page
  keeps that overlap own);
- each full-attn source layer scores the span's own queries against earlier
  complete pages (Quest upper bound, max-pooled over the chunk's queries so a
  page hot for ANY query is selectable), `select_pages` takes the top-K union
  the forced 8 pre-chunk pages, and the group's other layers reuse that one
  selection;
- selected cold pages promote through `PagedKvPool.promote_page` before
  attention and every resident private page demotes after the forward via the
  #500 `demote_page`/host-tier seam;
- the device pool is sized per slot as `k_pages + 8 window + one chunk of own
  pages`, not the full context — that is what makes a 256k context fit.

## No kernel change: the packed-table observation

cc ported this to sm70 and found the existing kernels are purely
**slot-causal** (sm70 split computes the mask from
`SeqLens-SeqQLens+tt+1`; the same slot test covers the sm90 MMA history and C
cells, but sm90 has not been run yet): attention therefore gets one packed block table

```
[ selected earlier pages (complete, n_sel of them) ; own span pages ]
seq_lens = n_sel*16 + own_len
```

Every selected page is a complete earlier page whose 16 keys precede every
query, so its slot columns are unmasked; the own span's own causal mask folds
into the slot test because `own_len - sq == q_start - own_offsets` (decode:
trailing window; prefill: own starts at the overlap boundary). One softmax over
the packed keys gives the V3.2/V4.1 `[selected ; window]` form. cc measured
3.1e-7 / 2.8e-7 on the CPU probe (`scripts/probe_sparse_attn_kwargs.py`) and
3.2e-4 on the V100 sm70; the sm90 parity run is pending a free card. No new
paged_attention kwargs were needed — an
earlier five-kwarg contract on #507 was superseded by this simpler form.

The block table otherwise keeps the existing semantics: ids in logical
sequence order, own-only table with a per-row `page_base` so `write_tokens`
indexes `pos//16 - page_base`, partial last own page sliced by `own_len`.

## Gates (`tests/test_sparse_engine.py`, real Engine, RefBackend)

1. **`k_pages >= pages` is token-for-token dense** across BOTH prefill and
   decode: a 6-page prompt + 8 generated tokens through the sparse engine
   (k=6) equals a dense engine byte-for-byte in sampled tokens, through the
   unmodified `paged_attention`.
2. **a real sparse run (k=2) cycles the tier**: pages demote and promote every
   tick (33 demotions / 27 promotions over a short run), bounds for every
   complete page survive demotion, finite tokens come out.
3. the live ledger reconciles: `page_bounds` and host `kv_cold` derived ==
   measured mid-run; the dense engine's manual-`sparse_retier` cold row (#500)
   is unchanged.

First-cut limits, marked in code: every selected page re-fetches next tick
(the cross-tick hot pin is the perf PR the card bench justifies); eager only
(no captured decode graph); bf16 pool (fp8 bounds raise to the card PR); no
prefix-store reuse or spec draft with sparse.

The 27B V100/H20 tokens-per-token and the "delta is a few pages" prediction
remain card measurements, pending-remote.
