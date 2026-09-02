"""Roofline for the chunked GDN's matmuls (chunk 64, DK=DV=128, batched over 48 value heads)
via torch.bmm; no correctness. The serial kernel does 63 ms at 1.2% of bf16 tensor peak."""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from tilerl_kernels.backend import get_backend

# (name, M, N, K, count per chunk) — the matmuls of _gdn_chunk_fwd, per head.
SHAPES = [
    ("KK^T   [n,n,DK]", 64, 64, 128, 1),
    ("M@bV   [n,n,DV]", 64, 128, 64, 1),
    ("M@beK  [n,n,DK]", 64, 128, 64, 1),
    ("W@S    [n,DK,DV]", 64, 128, 128, 1),
    ("QK^T   [n,n,DK]", 64, 64, 128, 1),
    ("P@S    [n,DK,DV]", 64, 128, 128, 1),
    ("A@d    [n,n,DV]", 64, 128, 64, 1),
    ("R^T@d  [DK,n,DV]", 128, 128, 64, 1),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heads", type=int, default=48, help="value heads (the batch dim)")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    b = get_backend()
    assert b.device.type == "cuda", "needs TILERL_TARGET=cuda"
    dev = b.device

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

    print(f"batched over {args.heads} heads; peak bf16 ~148 TFLOP/s on H20")
    print(f"{'matmul':>18} {'us':>8} {'GFLOP':>8} {'TFLOP/s':>9} {'% peak':>7}")
    tot_us = tot_fl = 0.0
    for name, m, n, k, _ in SHAPES:
        A = torch.randn(args.heads, m, k, device=dev, dtype=torch.bfloat16)
        B = torch.randn(args.heads, k, n, device=dev, dtype=torch.bfloat16)
        us = timed(lambda: torch.bmm(A, B))
        fl = 2.0 * args.heads * m * n * k
        tot_us += us
        tot_fl += fl
        print(f"{name:>18} {us:>8.1f} {fl / 1e9:>8.3f} {fl / us / 1e6:>9.2f} "
              f"{100 * fl / us / 1e6 / 148:>6.1f}%")
    print(f"{'per chunk per layer':>18} {tot_us:>8.1f} {tot_fl / 1e9:>8.3f} "
          f"{tot_fl / tot_us / 1e6:>9.2f} {100 * tot_fl / tot_us / 1e6 / 148:>6.1f}%")
    # T=512 -> 8 chunks, 48 GDN layers.
    ms = tot_us * 8 * 48 / 1e3
    print(f"\nT=512, 8 chunks x 48 GDN layers: {ms:.1f} ms  "
          f"(serial gdn_chunk_fused today: 63 ms)")


if __name__ == "__main__":
    main()
