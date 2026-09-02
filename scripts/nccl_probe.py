"""All-reduce rate at TP's message size: 2 per layer over [B, hidden] f32,
so the small-message rate decides TP, not peak NVLink.
  torchrun --nproc_per_node=8 scripts/nccl_probe.py
"""

from __future__ import annotations

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
    # NCCL sets up channels on the first collective; without this the first size reads 3x slow
    warm = torch.ones(5120 * 512, dtype=torch.float32, device="cuda")
    for _ in range(20):
        dist.all_reduce(warm)
    torch.cuda.synchronize()
    del warm
    for elems in (5120 * 512, 5120 * 64, 5120 * 8, 5120):
        x = torch.ones(elems, dtype=torch.float32, device="cuda")
        for _ in range(20):
            dist.all_reduce(x)
        # floor across windows: small-message rows jitter 3-5x from other tenants
        wins = []
        for _ in range(7):
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(50):
                dist.all_reduce(x)
            torch.cuda.synchronize()
            wins.append((time.perf_counter() - t0) / 50 * 1e6)
        us = min(wins)
        nbytes = elems * 4
        if rank == 0:
            print(f"  {nbytes:>10} {us:>13.1f} {nbytes / us / 1e3:>8.1f} {128 * us / 1e3:>14.2f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
