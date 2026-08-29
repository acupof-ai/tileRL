"""Per-kernel GPU time in one train_step (forward + tape backward).

The 27B LoRA row measures 67 SECONDS per step at 1x64 tokens. Kernel-ing the
frozen backward's dequant moved it only 1.18x, so the cost is somewhere else —
and one wrong guess is enough. This says where.

  python scripts/profile_train.py /data00/Qwen3.8-27B-NVFP4 --gpu 7 --len 64
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--len", type=int, default=64)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile

    from tilerl.autograd import AdamW
    from tilerl.config import qwen38_27b
    from tilerl.model import add_lora, load_hf
    from tilerl_kernels.backend import get_backend
    from tilerl.train import train_step

    backend = get_backend()
    cfg = qwen38_27b()
    model = load_hf(cfg, args.source, fuse_projections=False, num_layers=args.layers)
    model.params = backend.materialize(model.params)
    trainable = add_lora(model, rank=16)
    opt = AdamW(lr=1e-3)
    ids = np.arange(1, args.len + 1, dtype=np.int64).reshape(1, args.len) % cfg.vocab_size

    train_step(model, ids, backend, opt, trainable=trainable)  # warm (JIT)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        train_step(model, ids, backend, opt, trainable=trainable)
        torch.cuda.synchronize()

    by = defaultdict(lambda: [0, 0.0])
    total = 0.0
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        us = e.time_range.elapsed_us()
        by[e.name[:60]][0] += 1
        by[e.name[:60]][1] += us
        total += us
    print(f"\n{args.layers} layers, 1x{args.len}: GPU-busy {total / 1e3:.1f} ms, "
          f"{sum(c for c, _ in by.values())} kernels")
    print(f"{'kernel':<60} {'count':>7} {'us each':>9} {'ms':>9} {'%':>5}")
    for name, (c, us) in sorted(by.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{name:<60} {c:>7} {us / c:>9.1f} {us / 1e3:>9.2f} {100 * us / total:>5.1f}")


if __name__ == "__main__":
    main()
