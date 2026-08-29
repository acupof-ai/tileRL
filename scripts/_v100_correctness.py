"""sm70 correctness: truncate the 27B to N layers and check decode is sane.

8 layers (full-attn at idx 3,7 + 6 GDN) exercises both the attention and the
GDN path in seconds, isolating a wrong-op from the eager-GDN slowness of the
full 64-layer stack.

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70 \
    TILERL_TARGET=cuda TILERL_QWEN38_SOURCE=/data00/.../Qwen3.8-27B-NVFP4 \
    PYTHONPATH=packages/tilerl-kernels/src:src python scripts/_v100_correctness.py [N]
"""
import os
import sys
import time

import torch

from tilerl import config as config_mod
from tilerl import model as model_mod
from tilerl.engine import SamplingParams, build_engine
from tilerl.server import get_tokenizer
from tilerl_kernels.backend import get_backend


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    src = os.environ["TILERL_QWEN38_SOURCE"]
    cfg = config_mod.qwen38_27b()
    t0 = time.time()
    model = model_mod.load_hf(cfg, src, num_layers=n, fuse_projections=True)
    print(f"loaded {n} layers in {time.time() - t0:.1f}s", flush=True)
    backend = get_backend()
    print("device", torch.cuda.get_device_name(0), backend.arch, flush=True)

    # The truncated model has a real cfg with num_layers=n; build the engine
    # against model.cfg so the pools size to the truncated layer set.
    eng = build_engine(model.cfg, model, backend, num_blocks=64, num_slots=4, max_total_tokens=512)
    tok = get_tokenizer(src)
    ids = tok.encode("The capital of France is")
    print("PROMPT ids:", ids, flush=True)

    nnew = int(os.environ.get("PROBE_NEW", "8"))
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=nnew))
    t0 = time.time()
    steps = 0
    out = None
    while out is None:
        out = eng.take(rid)  # capture the result — take() POPS, don't call it twice
        if out is not None:
            break
        eng.step()
        steps += 1
        if steps > 60:
            print("ABORT: >60 steps (stuck)", flush=True)
            out = []
            break
    dt = time.time() - t0
    print(f"RAW OUT ids ({len(out)} in {dt:.1f}s, {steps} steps):", out, flush=True)
    print("decoded:", repr(tok.decode(out)), flush=True)
    print("per-id:", [(i, repr(tok.decode([i]))) for i in out], flush=True)
    # Sanity: ids must vary (not all EOS/identical) and lie in-vocab.
    ok = len(out) > 0 and len(set(out)) > 1
    print("CORRECTNESS", "PLAUSIBLE" if ok else "DEGENERATE (all-same or empty)")


if __name__ == "__main__":
    main()
