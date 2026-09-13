"""#557 card-2 validation: captured sparse decode/verify tick.

For each (ctx, B): sparse captured vs sparse eager, same greedy prompts/seeds,
64 decode tokens spanning the 8-tick refresh boundary (SPARSE_REFRESH_TICKS=8) ->
8 refreshes. Token equality + decode ms/tick (CUDA-synced, warm ticks dropped),
captured bucket count, and dense captured ms/tick for reference.

Capture confirmed by a nonempty engine._sparse_graphs.
Markers: CELL ... / V557_DONE.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.tokenizer import get_tokenizer

SRC = os.environ.get("SRC", "/work/tilerl-ckpt/Qwen3.8-27B-NVFP4")
STEPS = int(os.environ.get("STEPS", "64"))
WARM = 8
K = int(os.environ.get("K", "128"))
cli._QWEN38_SOURCE = SRC
backend = get_backend()
tok = get_tokenizer(SRC)
V = 248068


def make_prompts(ctx, b):
    # Fresh generator PER ARM: the old probe shared one module-level generator,
    # so the cap/eager/dense arms saw different 32k prompts and every
    # token_equal=False comparison was void (#557 CHANGE-REQ probe bug).
    g = torch.Generator().manual_seed(1234)
    out = []
    for i in range(b):
        ids = torch.randint(10, V - 100, (ctx,), generator=g).tolist()
        ids[-6:] = [11, 22, 33, 44, 55, 66 + i]
        out.append(np.asarray(ids, dtype=np.int64))
    return out


def run(ctx, b, sparse: bool, captured: bool):
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True,
                              backend=backend)
    kw = dict(num_blocks=0, num_slots=b + 2, max_batch=b,
              max_total_tokens=ctx + STEPS + 64, max_num_batched_tokens=512)
    if sparse:
        kw.update(sparse_k=K, scorer="bounds", kv_cold_bytes=1 << 34,
                  sparse_device_select=captured)
    e = build_engine(cfg, model, backend, **kw)
    sp = SamplingParams(temperature=0.0, seed=0, max_new_tokens=STEPS)
    rids = [e.submit(p, sp) for p in make_prompts(ctx, b)]
    toks = {r: [] for r in rids}
    ms = []
    seen_decode = 0
    deadline = time.perf_counter() + 2400
    while True:
        pre = e._decode_forwards
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        post = e._decode_forwards
        done = e.poll()
        for r in rids:
            for t in done.get(r, ()):
                if len(toks[r]) < STEPS:
                    toks[r].append(int(t))
        if post > pre:  # this tick ran a decode forward
            seen_decode += 1
            if seen_decode > WARM:
                ms.append(dt)
        if all(len(toks[r]) >= STEPS for r in rids):
            break
        if time.perf_counter() > deadline:
            print("TIMEOUT", ctx, b, sparse, captured,
                  {r: len(toks[r]) for r in rids}, flush=True)
            break
    buckets = len(getattr(e, "_sparse_graphs", {}))
    graph_on = bool(getattr(e, "_sparse_graph_on", False))
    e.shutdown()
    return [toks[r][:STEPS] for r in rids], graph_on, buckets, ms


ONLYS = set(os.environ.get("ONLYS", "32k1,32k8,128k1").split(","))
for ctx, bs in ((32768, (1, 8)), (131072, (1,))):
    for b in bs:
        tag = ("128k1" if ctx == 131072 else f"32k{b}")
        if tag not in ONLYS:
            continue
        tc, gon_c, bk_c, ms_c = run(ctx, b, True, True)
        te, gon_e, bk_e, ms_e = run(ctx, b, True, False)
        _, _, _, ms_d = run(ctx, b, False, True)
        eq = all(tc[i] == te[i] for i in range(b))
        mc = float(np.median(ms_c)) if ms_c else float("nan")
        me = float(np.median(ms_e)) if ms_e else float("nan")
        md = float(np.median(ms_d)) if ms_d else float("nan")
        print(f"CELL ctx={ctx} B={b} token_equal={eq} "
              f"sparse_graph_on={gon_c} buckets={bk_c} "
              f"eager_buckets={bk_e} nrows={[len(x) for x in tc]} "
              f"sparse_cap_ms={mc:.3f} sparse_eager_ms={me:.3f} "
              f"dense_cap_ms={md:.3f} cap/eager={me/mc:.3f} "
              f"dense/sparse_cap={mc/md:.3f}", flush=True)
        if not eq:
            for i in range(b):
                for j, (a, z) in enumerate(zip(tc[i], te[i])):
                    if a != z:
                        print(f"  row{i} first_diff@{j} cap={a} eager={z}", flush=True)
                        break
print("V557_DONE", flush=True)
