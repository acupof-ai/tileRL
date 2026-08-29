"""All-reduce bandwidth at the size tensor parallelism actually uses.

A TP=N decode tick does 2 all-reduces per layer over [B, hidden] f32 — 2.6 MB
at B=1, hidden 5120, 64 layers, so ~336 MB per tick. Peak NVLink numbers are
quoted for hundreds of MB; what decides TP here is the SMALL-message rate.

  torchrun --nproc_per_node=8 scripts/nccl_probe.py
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"world {world}")
        print(f"  {'bytes':>10} {'us/allreduce':>13} {'GB/s':>8} {'x64 layers ms':>14}")
    for elems in (5120, 5120 * 8, 5120 * 64, 5120 * 512):
        x = torch.ones(elems, dtype=torch.float32, device="cuda")
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(50):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 50 * 1e6
        nbytes = elems * 4
        if rank == 0:
            # a tick is 2 per layer x 64 layers
            print(f"  {nbytes:>10} {us:>13.1f} {nbytes / us / 1e3:>8.1f} {128 * us / 1e3:>14.2f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
