"""The chunked GDN's matmuls need 9.9 ms; the torch path spends ~290. Times the glue (gates,
triangular solve, permutes) at real shapes."""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from tilerl_kernels.backend import get_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--heads", type=int, default=48)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    b = get_backend()
    assert b.device.type == "cuda", "needs TILERL_TARGET=cuda"
    dev, n, h, d = b.device, args.chunk, args.heads, args.dk

    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        return sum(e.time_range.elapsed_us() for e in prof.events()
                   if e.device_type.name == "CUDA") / args.iters

    q = torch.randn(1, n, h, d, device=dev)
    k = torch.nn.functional.normalize(torch.randn(1, n, h, d, device=dev), dim=-1)
    gt = -torch.rand(1, n, h, device=dev) * 0.5
    bt = torch.rand(1, n, h, device=dev)
    eye = torch.eye(n, device=dev)
    KK = torch.einsum("bihd,bjhd->bhij", k, k)
    low = torch.tril(torch.ones(n, n, device=dev), -1)

    def gates():
        gc = gt.cumsum(1)
        e = torch.exp(gc)
        gp = gc.permute(0, 2, 1)
        D = torch.exp((gp.unsqueeze(-1) - gp.unsqueeze(-2)).clamp(max=0.0))
        return e, D

    e, D = gates()
    L = bt.permute(0, 2, 1).unsqueeze(-1) * KK * D * low

    def solve():
        return torch.linalg.solve_triangular(eye + L, eye.expand_as(L), upper=False,
                                             unitriangular=True)

    def permutes():
        return (k.permute(0, 2, 1, 3).contiguous(), q.permute(0, 2, 1, 3).contiguous(),
                bt.permute(0, 2, 1).unsqueeze(-1) * D)

    parts = [("gates (cumsum/exp/clamp)", gates), ("triangular solve", solve),
             ("permutes + masks", permutes)]
    print(f"chunk={n} heads={h} DK={d}   matmuls (roofline, measured): 25.7 us/chunk/layer")
    print(f"{'part':>26} {'us':>8} {'x matmuls':>10}")
    tot = 0.0
    for name, fn in parts:
        us = timed(fn)
        tot += us
        print(f"{name:>26} {us:>8.1f} {us / 25.7:>9.2f}x")
    print(f"{'glue total':>26} {tot:>8.1f} {tot / 25.7:>9.2f}x")
    print(f"\nT=512 (8 chunks x 48 layers): glue {tot * 8 * 48 / 1e3:.1f} ms, "
          f"matmuls 9.9 ms, serial kernel today 63 ms")


if __name__ == "__main__":
    main()
