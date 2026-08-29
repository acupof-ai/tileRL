"""V100 end-to-end: load the fp4 27B, decode, measure B=1 tok/s.

Runs the sm70 fp4 path through the real engine — the acceptance gate that the
Volta cell serves the 27B, not just that a GEMV matches a reference.

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70 \
    TILERL_TARGET=cuda TILERL_QWEN38_SOURCE=/data00/.../Qwen3.8-27B-NVFP4 \
    PYTHONPATH=packages/tilerl-kernels/src:src python scripts/_v100_e2e.py
"""
import os
import sys
import time

import torch

from tilerl import config as config_mod
from tilerl import model as model_mod
from tilerl.engine import SamplingParams, build_engine
from tilerl_kernels.backend import get_backend


def main():
    src = os.environ["TILERL_QWEN38_SOURCE"]
    cfg = config_mod.qwen38_27b()
    print(f"loading fp4 27B from {src} ...", flush=True)
    t0 = time.time()
    model = model_mod.load_hf(cfg, src, fuse_projections=True)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    backend = get_backend()
    print("device", torch.cuda.get_device_name(0), backend.arch, flush=True)
    print("HBM used", f"{torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    eng = build_engine(cfg, model, backend, num_blocks=256, num_slots=16, max_total_tokens=8192)

    # A short real prompt; greedy decode 64 tokens.
    from tilerl.server import get_tokenizer

    tok = get_tokenizer(src)
    prompt = "The capital of France is"
    ids = tok.encode(prompt)
    print("PROMPT ids:", ids, flush=True)

    # Diagnostic: no stop, no thinking budget, no id restriction — force 8 raw
    # tokens so an empty decode separates (a) logits (identical ids / EOS at 0),
    # (b) tokenizer (varied ids -> ""), (c) the commit rewrite (thinking_budget).
    params = SamplingParams(temperature=0.0, max_new_tokens=8)
    rid = eng.submit(ids, params)
    while eng.take(rid) is None:
        eng.step()
    out = eng.take(rid) or []
    print("RAW OUT ids:", out, flush=True)
    print("PROMPT:", prompt)
    print("OUTPUT:", repr(tok.decode(out)[:300]), flush=True)
    print("per-id decode:", [(i, repr(tok.decode([i]))) for i in out[:8]], flush=True)

    # timed run: fresh request, steady-state tok/s.
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=128))
    t0 = time.time()
    n = 0
    while eng.take(rid) is None:
        eng.step()
        n += 1
    dt = time.time() - t0
    out = eng.take(rid) or []
    tps = len(out) / dt if dt > 0 else 0
    print(f"DECODE B=1: {len(out)} tokens in {dt:.2f}s = {tps:.1f} tok/s", flush=True)
    print("E2E OK")


if __name__ == "__main__":
    main()
