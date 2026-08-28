"""Per-kernel GPU time in a PREFILL forward — the twin of
profile_graph_kernels.py, which only sees the captured decode graph. Prefill is
the biggest measured gap against sglang (B=1: 1836 vs 4022 tok/s), and nothing in the
tree said where its time goes.

Prefill is not graph-captured (shapes vary), so this profiles the real
engine.step() chunks that carry the prompt.

  python scripts/profile_prefill.py /data00/Qwen3.8-27B-NVFP4 --gpu 7 --len 2048
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
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=512,
                    help="prefill chunk width; 512 is the engine default, so that is "
                         "what this profiles unless you ask for another")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    import torch
    from torch.profiler import ProfilerActivity, profile

    import benchkit as bk
    from tilerl.config import qwen38_27b
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.model import load_hf
    from tilerl.ops.backend import get_backend

    backend = get_backend()
    model = load_hf(qwen38_27b(), args.source, fuse_projections=True, num_layers=args.layers)
    cfg = model.cfg
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=16, max_batch=8,
                          max_total_tokens=8192,
                          max_num_batched_tokens=args.chunk)
    b = args.batch

    def one_prefill():
        """Submit b prompts and step until they all reach decode; the steps in
        between are exactly the prefill chunks."""
        for i in range(b):
            engine.submit(bk.rand_prompt(cfg.vocab_size, args.len, seed=i),
                          SamplingParams(temperature=0.0, max_new_tokens=1, seed=i))
        done, steps = set(), 0
        while len(done) < b and steps < 10000:
            engine.step()
            done |= set(engine.poll())
            steps += 1
        return steps

    one_prefill()  # JIT + cache warm (every prefill shape compiles once)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        one_prefill()
        torch.cuda.synchronize()
    # Key by (kernel, input shapes): one kernel name covers several GEMM shapes
    # and only the per-shape cost says which one is slow.
    by = defaultdict(lambda: [0, 0.0])
    total = 0.0
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        us = e.time_range.elapsed_us()
        shapes = getattr(e, "input_shapes", None) or []
        sig = ",".join(str(tuple(x)) for x in shapes if x)[:44]
        key = f"{e.name[:34]} {sig}"
        by[key][0] += 1
        by[key][1] += us
        total += us
    toks = args.len * b
    print(f"\n{args.layers} layers, B={b}, prompt {args.len}: GPU-busy "
          f"{total / 1e3:.1f} ms, {sum(c for c, _ in by.values())} kernels, "
          f"{toks / (total / 1e6):.0f} tok/s GPU-bound (this layer count)")
    print(f"{'kernel  (input shapes)':<80} {'count':>6} {'us each':>8} {'ms':>8} {'%':>5}")
    for name, (c, us) in sorted(by.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{name:<80} {c:>6} {us / c:>8.1f} {us / 1e3:>8.3f} {100 * us / total:>5.1f}")


if __name__ == "__main__":
    main()
