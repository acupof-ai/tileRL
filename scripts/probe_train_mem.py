"""Peak train-step allocation per T for the 27B LoRA, plus the GDN backward's per-time-step
state size. b=2 x t=256 OOMs on a 95 GB H20, capping training at ~256 tokens per step.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_train_mem.py /data00/Qwen3.8-27B-NVFP4
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from tilerl.autograd import AdamW
from tilerl.config import qwen38_27b
from tilerl.model import add_lora, load_hf
from tilerl.train import train_step
from tilerl_kernels.backend import get_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--lens", default="64,128,256")
    args = ap.parse_args()
    b = get_backend()
    cfg = qwen38_27b()
    model = load_hf(cfg, args.source, fuse_projections=False, num_layers=args.layers)
    model.params = b.materialize(model.params)
    trainable = add_lora(model, rank=16)
    opt = AdamW(lr=1e-3)
    base = torch.cuda.memory_allocated() / 2**30
    print(f"weights + adapters resident: {base:.1f} GiB")
    nl = sum(1 for i in range(cfg.num_layers) if i not in cfg.full_attn_layers)
    print(f"{'T':>6} {'peak GiB':>10} {'over base':>10} {'GiB/token':>10} "
          f"{'gdn states':>11}")
    for t in (int(x) for x in args.lens.split(",")):
        ids = np.arange(1, t + 1, dtype=np.int64).reshape(1, t) % cfg.vocab_size
        train_step(model, ids, b, opt, trainable=trainable)  # warm
        torch.cuda.reset_peak_memory_stats()
        train_step(model, ids, b, opt, trainable=trainable)
        peak = torch.cuda.max_memory_allocated() / 2**30
        # states[b, t+1, nvh, key_dim, val_dim] f32, held for every GDN layer
        st = (t + 1) * cfg.linear_num_value_heads * cfg.linear_key_head_dim \
            * cfg.linear_value_head_dim * 4 * nl / 2**30
        print(f"{t:>6} {peak:>10.1f} {peak - base:>10.1f} {(peak - base) / t:>10.3f} "
              f"{st:>10.1f}G")


if __name__ == "__main__":
    main()
