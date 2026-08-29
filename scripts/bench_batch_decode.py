"""Decode throughput vs batch size on the NVFP4 slice (eager path).

Submits B concurrent requests, warms past their prefills, then times ticks
where all B are decoding. Use --decode-graph to capture per-bucket decode
graphs (pure-decode ticks replay instead of dispatching ~900 kernels);
without it every B runs the eager path. --draft turns on speculative decode
and reports its acceptance and the tokens actually committed per tick.

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
from tilerl_kernels.backend import get_backend
from tilerl.spec import load_draft

WARMUP = 8  # ticks: flushes every 16-token prompt's prefill, leaves decode headroom


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ticks", type=int, default=30)
    p.add_argument("--batches", type=str, default="1,2,4,8")
    p.add_argument("--fuse", action="store_true", help="fuse same-input fp4 projections")
    p.add_argument(
        "--decode-graph", action="store_true", help="capture decode graphs per batch bucket"
    )
    p.add_argument("--slots", type=int, help="state slots / max_batch (default: max batch swept)")
    p.add_argument("--draft", help="draft head safetensors: speculative decode")
    p.add_argument("--depth", type=int, default=4, help="drafts per row per tick")
    args = p.parse_args()

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = replace(
        qwen36_27b(),
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in qwen36_27b().full_attn_layers if i < args.layers),
    )
    model = load_hf(cfg, args.source, fuse_projections=args.fuse)
    batches = [int(x) for x in args.batches.split(",")]
    bmax = args.slots or max(batches)
    engine = build_engine(
        # 16-token prompts generating ~170: 16 blocks/request covers it. Sizing
        # off bmax alone filled the card and the allocator thrashed at 94 GiB.
        cfg, model, backend, num_blocks=16 * bmax, num_slots=bmax, max_batch=bmax,
        max_total_tokens=256 * bmax, decode_graph=args.decode_graph,
        draft=load_draft(model, args.draft) if args.draft else None, spec_depth=args.depth,
    )

    gen = torch.Generator().manual_seed(7)
    print(
        f"\n=== decode throughput vs batch "
        f"(slice {args.layers} layers, {'graph' if args.decode_graph else 'eager'}) ==="
    )
    print(f"  {'B':>3} {'ms/tick':>9} {'tok/tick':>9} {'accept':>7} "
          f"{'per-request tok/s':>18} {'aggregate tok/s':>17}")
    for B in batches:
        prompts = [
            torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist() for _ in range(B)
        ]
        wids = [
            engine.submit(
                p, SamplingParams(
                    temperature=0.0, seed=i,
                    max_new_tokens=(WARMUP + args.ticks + 4) * (1 + args.depth),
                )
            )
            for i, p in enumerate(prompts)
        ]
        for _ in range(WARMUP):  # flush prefills; all requests decoding afterwards
            engine.step()
        torch.cuda.synchronize()
        s0 = engine.stats()
        t0 = time.perf_counter()
        for _ in range(args.ticks):
            engine.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / args.ticks * 1e3
        s1 = engine.stats()
        per_tick = (s1["tokens_generated"] - s0["tokens_generated"]) / args.ticks / B
        drafted = s1["spec_drafted"] - s0["spec_drafted"]
        acc = (s1["spec_accepted"] - s0["spec_accepted"]) / max(drafted, 1)
        print(f"  {B:>3} {ms:>9.3f} {per_tick:>9.2f} {100 * acc:>6.1f}% "
              f"{1000 * per_tick / ms:>18.1f} {1000 * B * per_tick / ms:>17.1f}")
        # Drain to completion: a slot is freed at finish, and the next B reuses
        # the same pool — an under-sized drain exhausts it on the following arm.
        done: dict = {}
        while not all(w in done for w in wids):
            engine.step()
            done.update(engine.poll())  # poll clears per call; accumulate


if __name__ == "__main__":
    main()
