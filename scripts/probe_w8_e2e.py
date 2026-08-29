"""Paired W8 off/on decode, ONE process.

The wide-load fp4 mma is 1.18x on the kernel in isolation. Whether that reaches
the tick is a different question, and comparing against a baseline from another
run is the mistake this repo keeps making — so both arms are built and timed
here, back to back, on the same weights.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

import tilerl_kernels.kernels_linear as KL
import tilerl_kernels.registry as REG
from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import load_hf
from tilerl_kernels.backend import get_backend

WARMUP = 8


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--layers", type=int, default=64)
    p.add_argument("--ticks", type=int, default=30)
    p.add_argument("--batches", default="4,8")
    args = p.parse_args()
    backend = get_backend()
    base = qwen38_27b()
    cfg = replace(base, num_layers=args.layers,
                  full_attn_layers=tuple(i for i in base.full_attn_layers if i < args.layers))
    model = load_hf(cfg, args.source)
    gen = torch.Generator().manual_seed(7)

    print(f"  {'B':>3} {'W8':>3} {'ms/tick':>9} {'agg tok/s':>10} {'vs W8=0':>8}")
    for b in (int(x) for x in args.batches.split(",")):
        agg = {}
        for w8 in (0, 1):
            # The registry binds the factory BY REFERENCE at import, so patching
            # the module attribute does nothing — the dict entry is what resolves.
            REG._SM90_KERNELS["linear_fp4_mma8"] = (
                lambda t, _w=w8: KL.make_linear_fp4_mma8(t, W8=_w)
            )
            backend._kernels.clear()  # drop the compiled variant from the last arm
            engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=max(16, b),
                                  max_batch=max(8, b), decode_graph=True)
            prompts = [torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist()
                       for _ in range(b)]
            for i, pr in enumerate(prompts):
                engine.submit(pr, SamplingParams(temperature=0.0, seed=i,
                                                 max_new_tokens=WARMUP + args.ticks + 8))
            for _ in range(WARMUP):
                engine.step()
            torch.cuda.synchronize()
            s0, t0 = engine.stats(), time.perf_counter()
            for _ in range(args.ticks):
                engine.step()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / args.ticks * 1e3
            n = (engine.stats()["tokens_generated"] - s0["tokens_generated"]) / args.ticks
            agg[w8] = 1000.0 * n / ms
            print(f"  {b:>3} {w8:>3} {ms:>9.3f} {agg[w8]:>10.1f}"
                  + (f" {agg[1] / agg[0]:>7.3f}x" if w8 else ""))


if __name__ == "__main__":
    main()
