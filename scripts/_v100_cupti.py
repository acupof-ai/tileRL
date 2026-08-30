"""Kernel-level profile via CUPTI (torch.profiler) — works through CUDA graph
replay because it reads the CUPTI kernel records, not the Python op layer. The
eager op-timing profile was invalid (202ms/call was per-call JIT/dispatch, not
kernel time); self_device_time_total here is real GPU time, dispatch excluded.

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70e \
    TILERL_TARGET=cuda TILERL_QWEN38_SOURCE=... \
    PYTHONPATH=packages/tilerl-kernels/src:src python scripts/_v100_cupti.py
"""
import os

import torch
from torch.profiler import ProfilerActivity, profile

from tilerl import config as config_mod
from tilerl import model as model_mod
from tilerl.engine import SamplingParams, build_engine
from tilerl.server import get_tokenizer
from tilerl_kernels.backend import get_backend


def main():
    src = os.environ["TILERL_QWEN38_SOURCE"]
    cfg = config_mod.qwen38_27b()
    model = model_mod.load_hf(cfg, src, fuse_projections=True)
    be = get_backend()
    # graph ON (default) — profile the real replayed tick
    eng = build_engine(cfg, model, be, num_blocks=256, num_slots=16, max_total_tokens=8192)
    tok = get_tokenizer(src)
    ids = tok.encode("The capital of France is")

    # warm fully: JIT + graph capture off the profiled window
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=8))
    out = None
    while out is None:
        out = eng.take(rid)
        if out is None:
            eng.step()
    print("warm done:", repr(tok.decode(out)[:60]), flush=True)

    # profile a run of steady decode ticks
    N = 32
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=N + 2))
    # skip the first tick (prefill+capture); profile the steady ones
    eng.step()  # prefill
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(N):
            eng.step()
        torch.cuda.synchronize()
    while eng.take(rid) is None:
        eng.step()

    rows = []
    for e in p.key_averages():
        dt = getattr(e, "self_device_time_total", 0)
        if dt > 0:
            rows.append((e.key, dt / 1000.0 / N, e.count / N))  # ms/tick, launches/tick
    rows.sort(key=lambda r: -r[1])
    tot = sum(r[1] for r in rows)
    print(f"total device ms/tick = {tot:.2f}  ({1000/tot:.1f} tok/s if perfectly packed)", flush=True)
    print(f"{'kernel':52} {'ms/tick':>9} {'launch/tick':>12} {'%':>6}", flush=True)
    for key, mspt, cpt in rows[:25]:
        print(f"{key[:52]:52} {mspt:9.3f} {cpt:12.1f} {100*mspt/tot:5.1f}%", flush=True)
    print("CUPTI OK", flush=True)


if __name__ == "__main__":
    main()
