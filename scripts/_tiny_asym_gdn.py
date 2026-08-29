"""tiny with ASYMMETRIC GDN heads (nkh=2, nvh=6, 3:1 broadcast like the 27B's
16/48) + fp4 — the one structural axis the symmetric tiny never exercised.
Compare cpu vs sm70 ids; divergence isolates the asymmetric-head GDN path."""
import dataclasses
import os

import torch

from tilerl import config as config_mod
from tilerl import model as model_mod
from tilerl.engine import SamplingParams, build_engine
from tilerl_kernels.backend import get_backend


def main():
    base = config_mod.tiny()
    # 2 key heads, 6 value heads (3:1), fp4 on — mirrors the 27B's GQA-GDN.
    cfg = dataclasses.replace(
        base, fp4=True,
        linear_num_key_heads=2, linear_key_head_dim=16,
        linear_num_value_heads=6, linear_value_head_dim=16,
    )
    model = model_mod.build_random(cfg, seed=0, fuse_projections=True)
    backend = get_backend()
    print("target", os.environ.get("TILERL_TARGET"), "arch", backend.arch,
          "nkh", cfg.linear_num_key_heads, "nvh", cfg.linear_num_value_heads, flush=True)
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
    print("ASYM-GDN OUT ids:", out, flush=True)


if __name__ == "__main__":
    main()
