#!/usr/bin/env python3
"""First-diff fingerprint for sparse+spec: dump the token stream of the first N
identical MMLU questions under dense spec-off and sparse spec-on, report the
first differing absolute output index and the 20 tokens around it. Greedy,
thinking on, same engine build as mmlu_thinking_spec.py."""
from __future__ import annotations

import argparse
import os

from tilerl.engine import SamplingParams
from tilerl.eval import mmlu_questions

CONCURRENCY = 8
MAX_NEW = 2048
MAX_THINK = 512


def thinking_prompts(raw_prompts):
    from tilerl.prompt import render_chat

    out = []
    for p in raw_prompts:
        body = p.split("Answer:")[0].rstrip()
        out.append(render_chat([("user", body + "\nAnswer with one letter.")], True))
    return out


def gen(source, prompts, k, draft_path, tok, backend):
    from tilerl.cli import _build_model
    from tilerl.engine import build_engine

    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True,
                              backend=backend)
    draft = None
    if draft_path:
        from tilerl.spec import load_draft

        draft = load_draft(model, draft_path)
    kw = dict(num_blocks=0, num_slots=CONCURRENCY + 2, max_batch=CONCURRENCY,
              max_total_tokens=8192, max_num_batched_tokens=512,
              sparse_k=k, scorer="bounds", draft=draft, spec_depth=1)
    if k:
        kw["kv_cold_bytes"] = 1 << 34
    eng = build_engine(cfg, model, backend, **kw)
    sp = SamplingParams(
        temperature=0.0, seed=0, max_new_tokens=MAX_NEW,
        max_think_tokens=MAX_THINK,
        end_think_ids=tuple(tok.encode("</think>\n\n")),
        stop_token_ids=tuple(getattr(tok, "stop_token_ids", ())))
    try:
        # Submit all, then step until each request lands in poll(); read its ids
        # the once it is returned (poll does not retain finished requests).
        wids = [eng.submit(tok.encode(p), sp) for p in prompts]
        out = {}
        pending = set(wids)
        while pending:
            eng.step()
            for wid, ids in list(eng.poll().items()):
                if wid in pending:
                    out[wids.index(wid)] = ids
                    pending.discard(wid)
    finally:
        eng.shutdown()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--draft", default="model_mtp.safetensors")
    ap.add_argument("--out", default="/work/mmlu_firstdiff_5.json")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")
    from tilerl_kernels.backend import get_backend

    from tilerl.tokenizer import get_tokenizer

    raw, golds, _ = mmlu_questions(2000, 0)
    prompts = thinking_prompts(raw[: a.n])
    tok = get_tokenizer(a.source)
    backend = get_backend()
    dpath = None if not a.draft else (
        a.draft if os.path.isabs(a.draft) else os.path.join(a.source, a.draft))
    print("dense spec-off...", flush=True)
    dense = gen(a.source, prompts, 0, None, tok, backend)
    print("sparse spec-on...", flush=True)
    sparse = gen(a.source, prompts, a.k, dpath, tok, backend)
    result = {"gold": golds[: a.n], "questions": []}
    for i in range(a.n):
        dt, st = dense[i], sparse[i]
        dtxt, stxt = tok.decode(dt), tok.decode(st)
        fd = next((j for j in range(min(len(dt), len(st))) if dt[j] != st[j]),
                  min(len(dt), len(st)))
        lo = max(0, fd - 3)
        result["questions"].append({
            "i": i, "gold": golds[i],
            "dense_len": len(dt), "sparse_len": len(st),
            "first_diff": fd,
            "dense_ids_around": dt[lo:fd + 20],
            "sparse_ids_around": st[lo:fd + 20],
            "dense_tok_around": [tok.decode([t]) for t in dt[lo:fd + 20]],
            "sparse_tok_around": [tok.decode([t]) for t in st[lo:fd + 20]],
            "dense_tail": dtxt[-160:], "sparse_tail": stxt[-160:]})
        print(f"q{i} gold={golds[i]} dense_len={len(dt)} sparse_len={len(st)} "
              f"first_diff={fd}", flush=True)
        print(f"  dense : {[tok.decode([t]) for t in dt[lo:fd+8]]}", flush=True)
        print(f"  sparse: {[tok.decode([t]) for t in st[lo:fd+8]]}", flush=True)
    import json
    with open(a.out, "w") as fh:
        json.dump(result, fh)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
