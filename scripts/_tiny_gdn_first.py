"""tiny with GDN as the FIRST layer (full_attn_layers=(1,), so idx0 is GDN like
the 27B) + fp4. The default tiny puts full-attn at idx0, so GDN never processed
the raw embedding directly — the 27B does. cpu vs sm70; a divergence isolates
the GDN-first path (conv_window=None prefill on layer 0)."""
import dataclasses
import os

import torch

from tilerl import config as config_mod
from tilerl import model as model_mod
from tilerl.engine import SamplingParams, build_engine
from tilerl_kernels.backend import get_backend


def main():
    base = config_mod.tiny()
    # idx0 = GDN, idx1 = full-attn (the 27B has GDN at 0,1,2 then full-attn at 3).
    cfg = dataclasses.replace(base, fp4=True, full_attn_layers=(1,))
    model = model_mod.build_random(cfg, seed=0, fuse_projections=True)
    backend = get_backend()
    print("target", os.environ.get("TILERL_TARGET"), "arch", backend.arch,
          "full_attn_layers", cfg.full_attn_layers, flush=True)
    eng = build_engine(cfg, model, backend, num_blocks=64, num_slots=4, max_total_tokens=256)
    ids = [1, 2, 3, 4, 5]
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=8, logprobs=True))
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
    print("GDN-FIRST OUT ids:", out, flush=True)


if __name__ == "__main__":
    main()
