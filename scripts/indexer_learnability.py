#!/usr/bin/env python3
"""Card-free learnability test for the V4.1 page-indexer warm-up recipe.

Question (a3 2026-09-12): the 27B warm-up made recall WORSE (0.28->0.10) toward
a diffuse ~uniform page-mass teacher. Before spending another card, does the
recipe learn AT ALL when the dense page-mass teacher is CONCENTRATED and in the
model's reach?

Design (a3's "fixed random projection teacher"), driving the REAL loss, the
REAL hand-written tape backward, the REAL AdamW and the REAL topk recall:

  - one FIXED teacher (iq_star, ik_star) shared by every batch — the analogue
    of the frozen base's fixed H/K -> dense-mass map;
  - each step draws fresh random H [1,L,q,hidden] and page-K [1,L,p,ih,dkv];
  - teacher per-query scores are the REAL page_index_scores (sum_h ReLU(q.k) /
    sqrt(di)) with iq_star/ik_star, concentrated to a sharp softmax on each
    query's top-1 page (h_attn=ih so ik_weight=ik_star reproduces the key
    exactly -> the target is representable, no model error confound);
  - trained on fresh batches; evaluated on HELD unseen batches from the SAME
    fixed teacher, so the metric can only rise by recovering the shared teacher.

Pass criterion: held recall@1/@4 rises and KL falls at lr=0.02, with lr/10
moving the same way but slower. If it does NOT, the KL loss or pooled-head
target is broken and that is the finding. All numbers carry n (held batches x
layers x queries), seed, pages, steps, lr.
"""

from __future__ import annotations

import argparse
import json

import torch

from tilerl.autograd import AdamW, Tape
from tilerl.sparse_index import (
    WINDOW_PAGES,
    _indexable_mask,
    indexer_warmup_loss,
    page_index_scores,
    page_scores_for_selector,
    project_indexer_queries,
    project_page_keys,
    topk_page_recall,
)

torch.set_num_threads(4)


def make_teacher(seed, *, hidden, ih, dkv, di):
    """One FIXED random-projection teacher shared by every batch (the analogue
    of the frozen base model's fixed H/K->mass map). Returns iq_star, ik_star."""
    g = torch.Generator().manual_seed(seed)
    iq_star = torch.randn(ih, hidden, di, generator=g) * (1.0 / hidden**0.5)
    ik_star = torch.randn(ih, dkv, di, generator=g) * (1.0 / dkv**0.5)
    return iq_star, ik_star


def make_batch(
    seed,
    teacher,
    *,
    n_pages,
    n_queries,
    n_layers,
    hidden,
    ih,
    dkv,
    di,
    sharpness,
    n_peak,
    win=WINDOW_PAGES,
):
    """Fresh random inputs through the FIXED teacher + a CONCENTRATED target.

    Copy/needle proxy a3 specified as "a fixed random projection teacher": one
    shared teacher (iq_star,ik_star) is the analogue of the frozen base's fixed
    H/K->mass map; each batch draws fresh random H and page-K. The teacher score
    is the REAL page_index_scores (ReLU dots summed over heads) — the identical
    bilinear the learner computes — concentrated per query to a sharp softmax on
    its top n_peak pages (a needle target the V4.1 head can represent exactly:
    ik_weight=ik_star reproduces every projected key). The train target is thus
    the SAME per-(layer,query) page-KL distribution the production warm-up pools
    to. Generalizing to held batches requires recovering the shared teacher, not
    memorizing pages. Returns (H, k_pages [h_attn=ih, grouping identity],
    target[1,L,q,pages] L1 over indexable, n_pages)."""
    iq_star, ik_star = teacher
    g = torch.Generator().manual_seed(seed)
    H = torch.randn(1, n_layers, n_queries, hidden, generator=g) * 0.5
    k_pages = torch.randn(1, n_layers, n_pages, ih, dkv, generator=g)
    iq_proj = torch.einsum("rlqd,hde->rlqhe", H, iq_star)
    ik_proj = torch.einsum("rlphd,hde->rlphe", k_pages, ik_star)
    raw = page_index_scores(iq_proj, ik_proj)  # [1,L,q,p], heads summed
    indexable = n_pages - win
    raw_idx = raw[..., :indexable]
    topv, topi = raw_idx.topk(n_peak, dim=-1)
    logits = torch.full_like(raw_idx, -1e9)
    logits.scatter_(-1, topi, topv * sharpness)
    target = torch.zeros(1, n_layers, n_queries, n_pages)
    target[..., :indexable] = torch.softmax(logits, dim=-1)
    return H, k_pages, target, torch.tensor([n_pages])


def init_weights(seed, hidden, ih, dkv, di, scale=0.1):
    g = torch.Generator().manual_seed(seed)
    return {
        "iq": (torch.randn(ih, hidden, di, generator=g) * scale),
        "ik": (torch.randn(ih, dkv, di, generator=g) * scale),
    }


def eval_recall(H, k_pages, target, n_pages_t, w, ks=(1, 2, 4)):
    with torch.no_grad():
        iq = project_indexer_queries(H, w["iq"])
        ik = project_page_keys(k_pages, w["ik"])
        sel = page_scores_for_selector(iq, ik, n_pages_t, WINDOW_PAGES)
        out = {
            f"recall@{k}": float(topk_page_recall(sel, target, n_pages_t, k, WINDOW_PAGES))
            for k in ks
        }
        pages = ik.shape[2]
        m4 = _indexable_mask(n_pages_t, pages, WINDOW_PAGES, H.device)[:, None, None, :]
        lg = torch.log_softmax(page_index_scores(iq, ik).masked_fill(~m4, -1e9), -1)
        out["kl"] = float(-(target * lg).sum(-1).mean())
    return out


def mean_metrics(teacher, seeds, args, w):
    out = [None, None, None, None]
    for sd in seeds:
        H, k_pages, target, n_pages_t = make_batch(
            sd,
            teacher,
            n_pages=args.pages,
            n_queries=args.queries,
            n_layers=args.layers,
            hidden=args.hidden,
            ih=args.ih,
            dkv=args.dkv,
            di=args.di,
            sharpness=args.sharpness,
            n_peak=args.peak,
        )
        m = eval_recall(H, k_pages, target, n_pages_t, w)
        for j, k in enumerate(("recall@1", "recall@2", "recall@4", "kl")):
            out[j] = m[k] if out[j] is None else out[j] + m[k]
    return {
        "recall@1": out[0] / len(seeds),
        "recall@2": out[1] / len(seeds),
        "recall@4": out[2] / len(seeds),
        "kl": out[3] / len(seeds),
    }


def run(lr, steps, seed, args):
    teacher = make_teacher(seed + 12345, hidden=args.hidden, ih=args.ih, dkv=args.dkv, di=args.di)
    train_seeds = range(seed + 1, seed + 1 + steps)  # fresh batch/step
    held_seeds = [seed + 9000 + j for j in range(args.held)]  # unseen
    w = init_weights(seed + 7, args.hidden, args.ih, args.dkv, args.di)
    opt = AdamW(lr=lr)
    before = mean_metrics(teacher, held_seeds, args, w)
    for st, sd in enumerate(train_seeds):
        H, k_pages, target, n_pages_t = make_batch(
            sd,
            teacher,
            n_pages=args.pages,
            n_queries=args.queries,
            n_layers=args.layers,
            hidden=args.hidden,
            ih=args.ih,
            dkv=args.dkv,
            di=args.di,
            sharpness=args.sharpness,
            n_peak=args.peak,
        )
        with Tape() as tape:
            loss = indexer_warmup_loss(
                H, k_pages, w["iq"], w["ik"], target, n_pages_t, WINDOW_PAGES
            )
        grads = tape.backward(torch.ones(()), needs={id(w["iq"]), id(w["ik"])})
        opt.step([w["iq"], w["ik"]], grads)
        if st % 25 == 0 or st == steps - 1:
            print(f"lr={lr:<7} step {st:>3} train_kl {float(loss):.4f}", flush=True)
    after = mean_metrics(teacher, held_seeds, args, w)
    return {
        "lr": lr,
        "steps": steps,
        "seed": seed,
        "n_held": args.held,
        "before_held": before,
        "after_held": after,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--pages", type=int, default=64)
    ap.add_argument("--queries", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ih", type=int, default=4)
    ap.add_argument("--dkv", type=int, default=16)
    ap.add_argument("--di", type=int, default=16)
    ap.add_argument("--sharpness", type=float, default=6.0)
    ap.add_argument("--peak", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--out", default="/tmp/indexer_learnability.json")
    a = ap.parse_args()
    res = {
        "config": vars(a),
        "runs": [run(0.02, a.steps, a.seed, a), run(0.002, a.steps, a.seed, a)],
    }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("RESULT", json.dumps(res["runs"], indent=2))


if __name__ == "__main__":
    main()
