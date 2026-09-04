"""Per-kernel GPU time inside the captured decode graph (torch.profiler over replays);
the eager-event benches price launch+sync, not graph cost.

  python scripts/profile_graph_kernels.py /data00/Qwen3.8-27B-NVFP4 --gpu 6 --layers 8
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
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ticks", type=int, default=10)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    import benchkit as bk
    import torch
    from tilerl_kernels.backend import get_backend
    from torch.profiler import ProfilerActivity, profile

    from tilerl.config import qwen38_27b
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.model import load_hf

    backend = get_backend()
    model = load_hf(qwen38_27b(), args.source, fuse_projections=True, num_layers=args.layers)
    cfg = model.cfg
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=16, max_batch=8,
                          max_total_tokens=8192)
    b = args.batch
    for i in range(b):
        engine.submit(bk.rand_prompt(cfg.vocab_size, 512, seed=i),
                      SamplingParams(temperature=0.0, max_new_tokens=4096, seed=i))
    assert bk.settle_decode(engine, b, 8)
    for _ in range(8):  # graph captured + warm
        engine.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.ticks):
            engine.step()
        torch.cuda.synchronize()
    by = defaultdict(lambda: [0, 0.0])
    total = 0.0
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        us = e.time_range.elapsed_us()
        by[e.name[:70]][0] += 1
        by[e.name[:70]][1] += us
        total += us
    n = args.ticks
    print(f"\n{args.layers} layers, B={b}, {n} ticks: GPU-busy {total / n / 1e3:.3f} ms/tick, "
          f"{sum(c for c, _ in by.values()) // n} kernels/tick")
    print(f"{'kernel':<70} {'n/tick':>6} {'us each':>8} {'ms/tick':>8} {'%':>5}")
    for name, (c, us) in sorted(by.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{name:<70} {c // n:>6} {us / c:>8.1f} {us / n / 1e3:>8.3f} {100 * us / total:>5.1f}")


if __name__ == "__main__":
    main()
