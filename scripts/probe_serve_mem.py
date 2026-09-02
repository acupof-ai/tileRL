"""Serving memory after each construction step, then the largest live tensors by shape.
The arithmetic says 34 GiB at B=64 depth 512; the card fills at 94.

    CUDA_VISIBLE_DEVICES=6 python3 scripts/probe_serve_mem.py /data00/... --batches 32,64
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from tilerl.config import qwen36_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS
from tilerl.model import load_hf
from tilerl_kernels.backend import get_backend

GB = 1 << 30


def used() -> float:
    return torch.cuda.memory_allocated() / GB


def biggest(top: int = 12) -> None:
    seen, by = set(), defaultdict(lambda: [0, 0.0])
    for o in __import__("gc").get_objects():
        if not isinstance(o, torch.Tensor) or not o.is_cuda or id(o) in seen:
            continue
        seen.add(id(o))
        k = f"{tuple(o.shape)} {str(o.dtype).replace('torch.', '')}"
        by[k][0] += 1
        by[k][1] += o.numel() * o.element_size() / GB
    for k, (n, g) in sorted(by.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"    {g:8.2f} GiB  x{n:<5} {k}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--depth", type=int, default=512)
    ap.add_argument("--batches", default="32,64")
    args = ap.parse_args()

    backend = get_backend()
    cfg = qwen36_27b()
    print(f"start {used():.2f} GiB")
    model = load_hf(cfg, args.source, num_layers=args.layers, fuse_projections=True)
    print(f"after load_hf {used():.2f} GiB")

    for b in (int(x) for x in args.batches.split(",")):
        gen = 40
        need = 2 * (-(-(args.depth + gen) * b // BLOCK_TOKENS) + b)
        torch.cuda.empty_cache()
        base = used()
        engine = build_engine(
            cfg, model, backend, num_blocks=max(256, need), num_slots=max(16, b),
            max_batch=max(8, b), max_total_tokens=max(8192, args.depth + gen + 64),
        )
        print(f"\nB={b}  blocks={max(256, need)}  after build_engine "
              f"{used():.2f} GiB (+{used() - base:.2f})")
        toks = list(range(3, 3 + args.depth))
        for i in range(b):
            engine.submit(toks, SamplingParams(temperature=0.0, seed=i, max_new_tokens=gen))
        for t in range(60):
            engine.step()
            if t > 8:
                break
        torch.cuda.synchronize()
        print(f"B={b}  after {t + 1} ticks  {used():.2f} GiB  "
              f"reserved {torch.cuda.memory_reserved() / GB:.2f}  "
              f"peak {torch.cuda.max_memory_allocated() / GB:.2f}")
        biggest()
        del engine
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
