"""tiny with PARTIAL RoPE + large head_dim (rotary_dim 16 of head_dim 32) — the
27B's partial-rope-on-full-attn axis the default tiny (rotary_dim==head_dim)
never exercised. cpu vs sm70; divergence isolates partial-rope handling."""
import dataclasses
import os

import torch

from tilerl import config as config_mod
from tilerl import model as model_mod
from tilerl.engine import SamplingParams, build_engine
from tilerl_kernels.backend import get_backend


def main():
    base = config_mod.tiny()
    # head_dim 32, rotary_dim 16 (partial, like 27B's 64 of 256), fp4 on.
    cfg = dataclasses.replace(base, fp4=True, head_dim=32, rotary_dim=16)
    model = model_mod.build_random(cfg, seed=0, fuse_projections=True)
    backend = get_backend()
    print("target", os.environ.get("TILERL_TARGET"), "arch", backend.arch,
          "head_dim", cfg.head_dim, "rotary_dim", cfg.rotary_dim, flush=True)
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
    print("PARTIAL-ROPE OUT ids:", out, flush=True)


if __name__ == "__main__":
    main()
