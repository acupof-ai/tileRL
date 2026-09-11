#!/usr/bin/env python3
"""Interpretation controls for the recall FAIL, no training, one frozen forward
per span. On the SAME 256 sampled positions report captured dense attention
MASS for four page sets — random, bounds top-128, learned-index top-128, oracle
dense-mass top-128 — under two denominators:

- window EXCLUDED (the gate metric; target renormalized over indexable pages);
- window INCLUDED (pre-renorm mass rows sum to 1, selected set + the 8 forced
  window pages). This is the mass the model's output actually attends; it is the
  design-doc >=0.9 number.

Bounds is scored both pooled-GQA (the unit-F scorer) and per-head. Read-only."""
import json
import sys
from pathlib import Path

import torch
from tilerl_kernels.backend import get_backend

from tilerl.cli import _build_model
from tilerl.sparse_index import (
    WINDOW_PAGES,
    page_scores_for_selector,
    project_indexer_queries,
    project_page_keys,
)
from tilerl.train import (
    indexer_capture,
    init_indexer_weights,
    quest_bounds_scores,
    quest_bounds_scores_per_head,
    sample_query_positions,
)

CORPUS, K, SEED, NQ, MINP = Path(sys.argv[1]), 128, 0, 256, 2048
GROUP = {8192: 2, 16384: 0, 32768: 1}
NSPAN = {8192: 5, 16384: 6, 32768: 5}


def cap(mass, pages, include_window):
    """Mean captured mass [L,nq] over page list; window included/excluded."""
    idx = torch.as_tensor(pages, device=mass.device)
    return float(mass.index_select(-1, idx).sum(-1).mean())


def main():
    backend = get_backend()
    _cfg, model = _build_model("qwen38-27b", seed=SEED, keep_master=False,
                               backend=backend)
    gen = torch.Generator(device=backend.device).manual_seed(SEED)
    w = init_indexer_weights(model.cfg, gen, backend.device, 128)
    out = {}
    for ctx in (8192, 16384, 32768):
        with open(CORPUS / f"held_{ctx}.jsonl") as fh:
            rows = [json.loads(l) for l in fh][:NSPAN[ctx]]
        agg = {f"{s}_{win}": [] for s in
               ("random", "bounds", "bounds_per_head", "index", "oracle")
               for win in ("xwin", "iwin")}
        for j, row in enumerate(rows):
            ids = torch.tensor([row["ids"]], device=backend.device)
            qp = sample_query_positions(ctx, NQ, SEED + 100003 * GROUP[ctx] + j,
                                       MINP, backend.device)
            cap_ = indexer_capture(model, ids, backend, 16, WINDOW_PAGES, qp,
                                   return_raw=True)
            H, k_pages, target, n_pages, q_eval, bounds, raw = cap_
            L, P = target.shape[1], target.shape[-1]
            indexable = list(range(P - WINDOW_PAGES))
            winpages = list(range(P - WINDOW_PAGES, P))
            tgt_x = target[0]                                # [L,nq,p] renorm, win 0
            tgt_i = raw[0]                                   # [L,nq,p] sums to 1
            with torch.no_grad():
                iq = project_indexer_queries(H, w["iq"])
                ik = project_page_keys(k_pages, w["ik"])
                sel_idx = page_scores_for_selector(iq, ik, n_pages, WINDOW_PAGES)
                sel_bnd = quest_bounds_scores(q_eval, bounds)
                sel_bh = quest_bounds_scores_per_head(q_eval, bounds)
            rng = torch.Generator().manual_seed(SEED * 7919 + j)
            for l in range(L):
                k_eff = min(K, len(indexable))
                rnd = torch.tensor(indexable)[
                    torch.randperm(len(indexable), generator=rng)[:k_eff]].tolist()
                orc = [indexable[int(i)] for i in
                       tgt_x[l].sum(0)[torch.tensor(indexable)].topk(k_eff).indices]
                sets = {
                    "random": rnd,
                    "bounds": [int(i) for i in sel_bnd[0, l, :P - WINDOW_PAGES].topk(k_eff).indices],
                    "bounds_per_head": [int(i) for i in sel_bh[0, l, :P - WINDOW_PAGES].topk(k_eff).indices],
                    "index": [int(i) for i in sel_idx[0, l, :P - WINDOW_PAGES].topk(k_eff).indices],
                    "oracle": orc,
                }
                for name, pages in sets.items():
                    agg[f"{name}_xwin"].append(cap(tgt_x[l][None], pages, False))
                    agg[f"{name}_iwin"].append(
                        cap(tgt_i[l][None], pages + winpages, True))
        out[str(ctx)] = {k: round(sum(v) / len(v), 4) for k, v in agg.items()}
        print(ctx, out[str(ctx)], flush=True)
    print("RESULT", json.dumps(out))


if __name__ == "__main__":
    main()
