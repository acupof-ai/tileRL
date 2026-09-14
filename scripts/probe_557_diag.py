"""#557 diagnostic: which decode path each tick takes + first-diff logit gap.

Sparse-cap arm wraps _run_decode_graph / _run_sparse_decode_graph to count which
returned True, and saves per-step full logits. Same for sparse-eager. Reports
dense_graph ticks vs sparse_graph ticks vs eager, and logits max_abs at the first
token-diff step. ONLYS env picks ctx/B. Buckets>0 plus sparse_graph_ticks>0 is the
only real capture proof.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine, build_model
from tilerl.engine import Engine, SamplingParams
from tilerl.tokenizer import get_tokenizer

SRC = os.environ.get("SRC", "/work/tilerl-ckpt/Qwen3.8-27B-NVFP4")
STEPS = int(os.environ.get("STEPS", "64"))
WARM = 8
K = int(os.environ.get("K", "128"))
CTX = int(os.environ.get("CTX", "32768"))
B = int(os.environ.get("B", "1"))
QWEN38_SOURCE = SRC
backend = get_backend()
tok = get_tokenizer(SRC)
V = 248068


def make_prompts(ctx, b, gen):
    out = []
    for i in range(b):
        ids = torch.randint(10, V - 100, (ctx,), generator=gen).tolist()
        ids[-6:] = [11, 22, 33, 44, 66 + i]
        out.append(np.asarray(ids, dtype=np.int64))
    return out


def run(ctx, b, sparse: bool, captured="default"):
    """captured: "default" = build the engine exactly like serve (no explicit
    sparse_device_select, decode_graph auto); True/False force it for the
    cap/eager comparison arms."""
    cfg, model = build_model("qwen38-27b", seed=0, fuse_projections=True,
                              backend=backend)
    kw = dict(num_blocks=0, num_slots=b + 2, max_batch=b,
              max_total_tokens=ctx + STEPS + 64, max_num_batched_tokens=512)
    if sparse:
        kw.update(sparse_k=K, scorer="bounds", kv_cold_bytes=1 << 34)
        if captured != "default":
            kw["sparse_device_select"] = captured
    e = build_engine(cfg, model, backend, **kw)
    cnt = {"dense_graph": 0, "sparse_graph": 0, "eager_forward": 0, "decode": 0}
    od = e._run_decode_graph
    osg = e._run_sparse_decode_graph

    def wd(reqs, chains=None):
        r = od(reqs, chains)
        if r:
            cnt["dense_graph"] += 1
        return r

    def ws(reqs, chains):
        r = osg(reqs, chains)
        if r:
            cnt["sparse_graph"] += 1
        return r
    e._run_decode_graph = wd
    e._run_sparse_decode_graph = ws

    lg = {}
    o_sample = Engine._sample_batch

    def hook(self, rows):
        for r, l, g in rows:
            if len(r.output) < STEPS:
                lg.setdefault(r.req_id, {})[len(r.output)] = (
                    int(l.argmax()), l.detach().float().clone().cpu())
        return o_sample(self, rows)
    Engine._sample_batch = hook

    sp = SamplingParams(temperature=0.0, seed=0, max_new_tokens=STEPS)
    gen = torch.Generator().manual_seed(1234)
    rids = [e.submit(pp, sp) for pp in make_prompts(ctx, b, gen)]
    toks = {r: [] for r in rids}
    decode_ticks = 0
    deadline = time.perf_counter() + 2400
    while True:
        pre = e._decode_forwards
        e.step()
        post = e._decode_forwards
        if post > pre:
            decode_ticks += 1
        done = e.poll()
        for r in rids:
            for t in done.get(r, ()):
                if len(toks[r]) < STEPS:
                    toks[r].append(int(t))
        if all(len(toks[r]) >= STEPS for r in rids):
            break
        if time.perf_counter() > deadline:
            print("TIMEOUT", {r: len(toks[r]) for r in rids}, flush=True)
            break
    cnt["eager_forward"] = decode_ticks - cnt["dense_graph"] - cnt["sparse_graph"]
    cnt["decode"] = decode_ticks
    out = {
        "cnt": dict(cnt),
        "buckets": len(getattr(e, "_sparse_graphs", {})),
        "gon": bool(getattr(e, "_sparse_graph_on", False)),
        "dgon": bool(getattr(e, "_decode_graph_on", False)),
        "toks": [toks[r][:STEPS] for r in rids],
        "lg": {i: lg[r] for i, r in enumerate(rids)},
    }
    Engine._sample_batch = o_sample
    e.shutdown()
    return out


print(f"DIAG ctx={CTX} B={B} default(captured-on) then cap then eager", flush=True)
u = run(CTX, B, True, "default")
c = run(CTX, B, True, True)
x = run(CTX, B, True, False)
print(f"PATH default {u['cnt']} gon={u['gon']} dgon={u['dgon']} buckets={u['buckets']}",
      flush=True)
print(f"PATH cap     {c['cnt']} gon={c['gon']} dgon={c['dgon']} buckets={c['buckets']}",
      flush=True)
print(f"PATH eager   {x['cnt']} gon={x['gon']} dgon={x['dgon']} buckets={x['buckets']}",
      flush=True)
for i in range(B):
    for j, (a, z) in enumerate(zip(u["toks"][i], c["toks"][i])):
        if a != z:
            lu = u["lg"][i].get(j)
            lc = c["lg"][i].get(j)
            gap = float((lu[1] - lc[1]).abs().max()) if lu and lc else float("nan")
            print(f"FIRSTDIFF default-vs-cap row{i} step{j} logit_maxabs={gap:.4g}",
                  flush=True)
            break
    else:
        print(f"FIRSTDIFF default-vs-cap row{i} none (token-equal)", flush=True)
for i in range(B):
    for j, (a, z) in enumerate(zip(c["toks"][i], x["toks"][i])):
        if a != z:
            lc = c["lg"][i].get(j)
            lx = x["lg"][i].get(j)
            gap = float((lc[1] - lx[1]).abs().max()) if lc and lx else float("nan")
            print(f"FIRSTDIFF cap-vs-eager row{i} step{j} cap_tok={a} eager_tok={z} "
                  f"logit_maxabs={gap:.4g} "
                  f"cap_am={lc[0] if lc else None} eager_am={lx[0] if lx else None}",
                  flush=True)
torch.save({"cap": {k: v for k, v in c.items() if k != "lg"},
            "eager": {k: v for k, v in x.items() if k != "lg"}}, "/work/d557_paths.pt")
print("DIAG_DONE", flush=True)
