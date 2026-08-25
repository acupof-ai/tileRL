"""Decode throughput vs batch size on the NVFP4 slice (eager path).

Submits B concurrent requests, warms past their prefills, then times ticks
where all B are decoding. The decode graph is M=1-only (day-1), so every B
runs the eager path here — the scaling with B is the point (weights are read
once per tick regardless of B; the dequant issue bottleneck is an M=1 disease
that amortizes over the batch), not the absolute B=1 number (graph capture
gives 49 tok/s at B=1).

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src TILERL_TARGET=cuda \\
        python3 scripts/bench_batch_decode.py /host/tc27-nvfp4-slice4 --layers 4
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from tilerl.config import qwen36_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import load_hf
from tilerl.ops.backend import get_backend

WARMUP = 8  # ticks: flushes every 16-token prompt's prefill, leaves decode headroom


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ticks", type=int, default=30)
    p.add_argument("--batches", type=str, default="1,2,4,8")
    args = p.parse_args()

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = replace(
        qwen36_27b(),
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in qwen36_27b().full_attn_layers if i < args.layers),
    )
    model = load_hf(cfg, args.source)
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=8, decode_graph=False)

    gen = torch.Generator().manual_seed(7)
    print(f"\n=== decode throughput vs batch (slice {args.layers} layers, eager) ===")
    print(f"  {'B':>3} {'ms/tick':>9} {'per-request tok/s':>18} {'aggregate tok/s':>17}")
    for B in (int(x) for x in args.batches.split(",")):
        prompts = [
            torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist() for _ in range(B)
        ]
        wids = [
            engine.submit(
                p, SamplingParams(temperature=0.0, max_new_tokens=WARMUP + args.ticks + 4, seed=i)
            )
            for i, p in enumerate(prompts)
        ]
        for _ in range(WARMUP):  # flush prefills; all requests decoding afterwards
            engine.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.ticks):
            engine.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / args.ticks * 1e3
        print(f"  {B:>3} {ms:>9.3f} {1000 / ms:>18.1f} {1000 * B / ms:>17.1f}")
        done = {}
        for _ in range(WARMUP + args.ticks + 8):  # drain to completion
            done.update(engine.poll())  # poll clears per call; accumulate
            if all(w in done for w in wids):
                break
            engine.step()


if __name__ == "__main__":
    main()
