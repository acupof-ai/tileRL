"""tiny-model cpu-vs-sm70 decode parity — the sm70 end-to-end correctness gate.

Same seeded random weights, both targets, greedy decode 8 tokens: identical
token ids means the sm70 kernel cell (GEMV + reused CPU attn/GDN/norm) is
end-to-end correct, independent of the 27B's real weights. Runs in seconds.
Set TILERL_TARGET per invocation and compare the printed ids across two runs.

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70 \
    TILERL_TARGET=cuda PYTHONPATH=packages/tilerl-kernels/src:src \
    python scripts/_tiny_parity.py
"""
import os
import sys

import torch

from tilerl import config as config_mod
from tilerl import model as model_mod
from tilerl.engine import SamplingParams, build_engine
from tilerl_kernels.backend import get_backend


def main():
    cfg = config_mod.tiny()
    model = model_mod.build_random(cfg, seed=0, fuse_projections=True)
    backend = get_backend()
    print("target", os.environ.get("TILERL_TARGET"), "arch", backend.arch, flush=True)
    eng = build_engine(cfg, model, backend, num_blocks=64, num_slots=4, max_total_tokens=256)
    ids = [1, 2, 3, 4, 5]
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=8))
    out = None
    steps = 0
    while out is None:
        out = eng.take(rid)
        if out is not None:
            break
        eng.step()
        steps += 1
        if steps > 40:
            out = []
            break
    print("TINY OUT ids:", out, flush=True)


if __name__ == "__main__":
    main()
