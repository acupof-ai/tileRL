"""tiny WITH fp4=True: the fp4 GEMV end-to-end path tiny normally skips.

build_random(fp4) packs the random weights to fp4 (pack_fp4 + renorm), so this
runs linear_fp4_gemv_sm70 in a real forward — the exact path the 27B exercises
and the bf16 tiny does not. Compare ids across cpu and sm70 runs: identical =>
fp4 e2e correct (the 27B's 220 is elsewhere); sm70 degenerate => reproduced.
"""
import dataclasses
import os

import torch

from tilerl import config as config_mod
from tilerl import model as model_mod
from tilerl.engine import SamplingParams, build_engine
from tilerl_kernels.backend import get_backend


def main():
    cfg = dataclasses.replace(config_mod.tiny(), fp4=True)
    model = model_mod.build_random(cfg, seed=0, fuse_projections=True)
    backend = get_backend()
    print("target", os.environ.get("TILERL_TARGET"), "arch", backend.arch, "fp4", cfg.fp4, flush=True)
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
    print("FP4 TINY OUT ids:", out, flush=True)


if __name__ == "__main__":
    main()
