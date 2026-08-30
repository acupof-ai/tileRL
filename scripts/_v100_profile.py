"""Segment-profile one decode tick: wrap backend ops with CUDA events to see
where the 48ms goes (paged_attention vs fp4 GEMV vs norms vs gdn). Runs EAGER
(decode_graph=False) so the wrappers actually execute per call — the relative
split holds; absolute ms is higher than the captured 48ms tick.

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70e \
    TILERL_TARGET=cuda TILERL_QWEN38_SOURCE=... \
    PYTHONPATH=packages/tilerl-kernels/src:src python scripts/_v100_profile.py
"""
import collections
import os
import time

import torch

from tilerl import config as config_mod
from tilerl import model as model_mod
from tilerl.engine import SamplingParams, build_engine
from tilerl.server import get_tokenizer
from tilerl_kernels.backend import get_backend

TIMES = collections.defaultdict(float)
COUNTS = collections.defaultdict(int)


def _wrap(be, name):
    fn = getattr(be, name)

    def timed(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        TIMES[name] += (time.perf_counter() - t0) * 1000
        COUNTS[name] += 1
        return r

    setattr(be, name, timed)


def main():
    src = os.environ["TILERL_QWEN38_SOURCE"]
    cfg = config_mod.qwen38_27b()
    model = model_mod.load_hf(cfg, src, fuse_projections=True)
    be = get_backend()
    for name in ("paged_attention", "linear_fp4", "linear", "rmsnorm", "rope",
                 "silu_mul", "gdn_decode", "linear_attn_chunk", "embedding",
                 "add", "attn_prep", "flip_window_parity"):
        if hasattr(be, name):
            _wrap(be, name)
    # eager decode: no graph, so the wrappers run every tick
    eng = build_engine(cfg, model, be, num_blocks=256, num_slots=16,
                       max_total_tokens=8192, decode_graph=False)
    tok = get_tokenizer(src)
    ids = tok.encode("The capital of France is")

    # warm one request (compile), reset timers, then profile 8 decode ticks
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=4))
    out = None
    while out is None:
        out = eng.take(rid)
        if out is None:
            eng.step()
    TIMES.clear(); COUNTS.clear()

    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=8))
    t0 = time.perf_counter()
    out = None
    n = 0
    while out is None:
        out = eng.take(rid)
        if out is not None:
            break
        eng.step()
        n += 1
    wall = (time.perf_counter() - t0) * 1000

    print(f"eager wall for {n} ticks = {wall:.0f}ms ({wall/max(n,1):.1f}ms/tick)", flush=True)
    print(f"{'op':22} {'total_ms':>10} {'calls':>7} {'ms/call':>9} {'%wall':>7}", flush=True)
    for name in sorted(TIMES, key=lambda x: -TIMES[x]):
        t, c = TIMES[name], COUNTS[name]
        print(f"{name:22} {t:10.1f} {c:7d} {t/max(c,1):9.3f} {100*t/wall:6.1f}%", flush=True)
    print("PROFILE OK", flush=True)


if __name__ == "__main__":
    main()
