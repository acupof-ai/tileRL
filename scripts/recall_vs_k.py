#!/usr/bin/env python3
"""Recall vs k and window/sink mass from one frozen forward per span.

1. captured dense page mass (pre-renorm, window in) for random/bounds/oracle at
   k in {128,256,512,1024,2048}+8 window, at 8k and 32k — finds the k where
   oracle and bounds cross 0.9 and the gap between them.
2. mass on page 0 (attention sink) and the last 8 window pages, one span, to
   explain why window inclusion moves recall ~0.

No training. Read-only eval. Usage: recall_vs_k.py CORPUS [card_ctx ...]."""
import json
import sys
from pathlib import Path

import torch
from tilerl_kernels.backend import get_backend

from tilerl.cli import _build_model
from tilerl.train import indexer_capture, quest_bounds_scores, sample_query_positions

KS = (128, 256, 512, 1024, 2048)
SEED, NQ, MINP, W = 0, 256, 2048, 8
GROUP = {8192: 2, 16384: 0, 32768: 1}
NSPAN = {8192: 5, 16384: 6, 32768: 5}


def cap(raw, pages, winpages):
    """captured fraction of TOTAL pre-renorm mass from selected pages + window."""
    idx = torch.as_tensor(pages + winpages, device=raw.device)
    return float(raw.index_select(-1, idx).sum(-1).mean())


def main():
    corpus = Path(sys.argv[1])
    ctxs = tuple(int(x) for x in sys.argv[2:]) or (8192, 32768)
    backend = get_backend()
    _cfg, model = _build_model("qwen38-27b", seed=SEED, keep_master=False,
                               backend=backend)
    result = {}
    for ctx in ctxs:
        with open(corpus / f"held_{ctx}.jsonl") as fh:
            rows = [json.loads(l) for l in fh][:NSPAN[ctx]]
        agg = {k: {"random": [], "bounds": [], "oracle": []} for k in KS}
        sink = win = total = None
        for j, row in enumerate(rows):
            ids = torch.tensor([row["ids"]], device=backend.device)
            qp = sample_query_positions(ctx, NQ, SEED + 100003 * GROUP.get(ctx, 0) + j,
                                       MINP, backend.device)
            captured = indexer_capture(model, ids, backend, 16, W, qp, return_raw=True)
            _H, _kp, target, _np, _qe, bounds, raw = captured
            L, P = raw.shape[1], raw.shape[-1]
            indexable = list(range(P - W))
            winpages = list(range(P - W, P))
            rng = torch.Generator().manual_seed(SEED * 7919 + j)
            with torch.no_grad():
                bsel = quest_bounds_scores(_qe, bounds)
            for l in range(L):
                m = raw[0, l]                                 # [nq,p], sums~1
                # window/sink sanity on the first source layer of the first span
                if j == 0 and l == 0:
                    sink = float(m[:, 0].mean())
                    win = float(m[:, P - W:].sum(-1).mean())
                    total = float(m.sum(-1).mean())
                for k in KS:
                    ke = min(k, len(indexable))
                    rnd = torch.tensor(indexable)[
                        torch.randperm(len(indexable), generator=rng)[:ke]].tolist()
                    orc = [indexable[int(i)] for i in
                           target[0, l].sum(0)[torch.tensor(indexable)].topk(ke).indices]
                    bnd = [int(i) for i in bsel[0, l, :P - W].topk(ke).indices]
                    agg[k]["random"].append(cap(m, rnd, winpages))
                    agg[k]["bounds"].append(cap(m, bnd, winpages))
                    agg[k]["oracle"].append(cap(m, orc, winpages))
            del captured, raw, target, bounds, _H, _kp, _qe
            if backend.device.type == "cuda":
                torch.cuda.empty_cache()
        result[str(ctx)] = {
            "curve": {str(k): {s: round(sum(v) / len(v), 4)
                               for s, v in agg[k].items()} for k in KS},
            "window_mass_first_span": {"page0_sink": round(sink, 4),
                                       "last8_window": round(win, 4),
                                       "row_total": round(total, 4)}}
        print("CTX", ctx, json.dumps(result[str(ctx)]), flush=True)
    print("RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
