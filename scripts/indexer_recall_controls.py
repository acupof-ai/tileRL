#!/usr/bin/env python3
"""Interpretation controls for the recall FAIL, no training: on the SAME 256
sampled positions, score four page sets against the dense page-mass teacher:
random indexable pages, the bounds top-128, the learned-index top-128, and the
oracle dense-mass top-128. Reports mean captured MASS per length. A
window-only set is 0 by construction (the metric renormalizes the 8-page
window out of the target), so it is not a row. Read-only eval."""
import json
import sys
from pathlib import Path

import torch

from tilerl import config as cfg_mod
from tilerl.cli import _build_model
from tilerl.train import (indexer_capture, init_indexer_weights,
                          quest_bounds_scores, sample_query_positions)
from tilerl_kernels.backend import get_backend
from tilerl.sparse_index import (WINDOW_PAGES, page_scores_for_selector,
                                 project_indexer_queries, project_page_keys)

CORPUS, K, SEED, NQ, MINP = Path(sys.argv[1]), 128, 0, 256, 2048
GROUP = {8192: 2, 16384: 0, 32768: 5}
NSPAN = {8192: 5, 16384: 6, 32768: 5}


def mass_of(chosen, target):
    # chosen [p] bool/index over indexable pages; target [L,nq,p] L1 page mass
    idx = torch.as_tensor(chosen, device=target.device)
    return float(target.index_select(-1, idx).sum(-1).mean())


def main():
    backend = get_backend()
    cfg = cfg_mod.qwen38_27b()
    model = _build_model("qwen38-27b", seed=SEED, keep_master=False,
                         backend=backend)
    gen = torch.Generator(device=backend.device).manual_seed(SEED)
    w = init_indexer_weights(model.cfg, gen, backend.device, 128)
    out = {}
    for ctx in (8192, 16384, 32768):
        rows = [json.loads(l) for l in open(CORPUS / f"held_{ctx}.jsonl")][:NSPAN[ctx]]
        agg = {k: [] for k in ("random", "bounds", "index", "oracle")}
        for j, row in enumerate(rows):
            ids = torch.tensor([row["ids"]], device=backend.device)
            qp = sample_query_positions(ctx, NQ, SEED + 100003 * {8192: 2, 16384: 0, 32768: 1}[ctx] + j,
                                       MINP, backend.device)
            H, k_pages, target, n_pages, q_eval, bounds = indexer_capture(
                model, ids, backend, 16, WINDOW_PAGES, qp)
            # target [1,L,nq,p], window pages already zero and renormalised
            L, P = target.shape[1], target.shape[-1]
            indexable = [p for p in range(P - WINDOW_PAGES)]
            per_q_mass = target[0]                                   # [L,nq,p]
            with torch.no_grad():
                iq = project_indexer_queries(H, w["iq"])
                ik = project_page_keys(k_pages, w["ik"])
                index_sel = page_scores_for_selector(iq, ik, n_pages, WINDOW_PAGES)
                bounds_sel = quest_bounds_scores(q_eval, bounds)
            rng = torch.Generator().manual_seed(SEED * 7919 + j)
            for l in range(L):
                tgt = per_q_mass[l]                                 # [nq,p]
                k_eff = min(K, len(indexable))
                rnd = torch.tensor(indexable)[torch.randperm(
                    len(indexable), generator=rng)[:k_eff]]
                agg["random"].append(mass_of(rnd.tolist(), tgt[None]))
                orc = tgt.sum(0)[torch.tensor(indexable)].topk(k_eff).indices
                agg["oracle"].append(mass_of([indexable[int(i)] for i in orc],
                                             tgt[None]))
                s_idx = index_sel[0, l, :P - WINDOW_PAGES].topk(k_eff).indices
                s_bnd = bounds_sel[0, l, :P - WINDOW_PAGES].topk(k_eff).indices
                agg["index"].append(mass_of([int(i) for i in s_idx], tgt[None]))
                agg["bounds"].append(mass_of([int(i) for i in s_bnd], tgt[None]))
        out[str(ctx)] = {k: sum(v) / len(v) for k, v in agg.items()}
        print(ctx, {k: round(x, 4) for k, x in out[str(ctx)].items()}, flush=True)
    print("RESULT", json.dumps(out))


if __name__ == "__main__":
    main()
