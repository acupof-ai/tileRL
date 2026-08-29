"""TP decode throughput on the 27B: tp=N ranks vs the same engine at tp=1.

Each rank loads the checkpoint on CPU, keeps only its shard, and builds the
engine from the sharded config, so no rank ever holds the whole model on the
device.

  torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29540 \
      scripts/bench_tp.py /data00/Qwen3.8-27B-NVFP4 --batches 1,8,16

CUDA-graph capture is OFF here on purpose: a captured tick would have to
capture the NCCL collectives too, and graph capture already has an open
failure on this pod. Both arms run eager so the comparison is like for like -
comparing a TP eager tick against the shipped graph tick is what made the
speculation entry wrong the first time.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

from tilerl.config import qwen36_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import _fuse_projections, load_hf
from tilerl.tensor_parallel import shard_params, tp_config
from tilerl_kernels.backend import Backend, resolve_target


def time_decode(engine, cfg, B, ticks, windows=3):
    gen = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist() for _ in range(B)]
    wids = [
        engine.submit(p, SamplingParams(temperature=0.0, seed=i,
                                        max_new_tokens=(8 + ticks * windows + 8)))
        for i, p in enumerate(prompts)
    ]
    for _ in range(8):  # flush every prompt's prefill: time pure decode only
        engine.step()
    torch.cuda.synchronize()
    out = []
    for _ in range(windows):
        t0 = time.perf_counter()
        for _ in range(ticks):
            engine.step()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / ticks * 1e3)
    done: dict = {}
    while not all(w in done for w in wids):
        engine.step()
        done.update(engine.poll())
    return statistics.median(out), (max(out) - min(out)) / statistics.median(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--batches", default="1,8,16")
    ap.add_argument("--ticks", type=int, default=20)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        torch.cuda.set_device(rank)

    full = qwen36_27b()
    backend = Backend(resolve_target())
    backend.init_tp(world, rank)

    # Sharding runs on UNFUSED params: fusing the already-local q/k/v shards
    # afterwards gives the layout the fused kernels want, and needs no
    # segment bookkeeping. Fusing first would make every rank slice across
    # the q|k|v boundary.
    model = load_hf(full, args.source, num_layers=args.layers, fuse_projections=(world == 1))
    cfg = tp_config(full, world)
    if world > 1:
        model.params = shard_params(model.params, full, rank, world)
        _fuse_projections(cfg, model.params)
    if args.layers != full.num_layers:
        from dataclasses import replace
        cfg = replace(cfg, num_layers=args.layers,
                      full_attn_layers=tuple(i for i in cfg.full_attn_layers if i < args.layers))
    model.cfg = cfg

    batches = [int(x) for x in args.batches.split(",")]
    engine = build_engine(cfg, model, backend, num_blocks=16 * max(batches),
                          num_slots=max(batches), max_batch=max(batches),
                          max_total_tokens=256 * max(batches), decode_graph=False)
    if rank == 0:
        print(f"\n=== decode, tp={world}, {args.layers} layers, eager ===")
        print(f"  {'B':>3} {'ms/tick':>9} {'tok/s/req':>10} {'agg tok/s':>10} {'spread':>8}")
    for B in batches:
        ms, spread = time_decode(engine, cfg, B, args.ticks)
        if rank == 0:
            print(f"  {B:>3} {ms:>9.2f} {1000 / ms:>10.1f} {1000 * B / ms:>10.1f}"
                  f" {100 * spread:>7.1f}%")
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
