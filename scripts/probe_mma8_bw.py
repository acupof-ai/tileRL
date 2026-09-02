"""GEMV vs mma8 on the same weight bytes, back to back. Bare for the timing, or under
``ncu --set full --kernel-name-base function`` for the memory counters.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_mma8_bw.py
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # the 27B's decode shapes (hidden 5120); the in-model gap averages over all of them
    ap.add_argument("--shapes", default="8192x5120,5120x6144,17408x5120,5120x17408,"
                                        "10240x5120,6144x5120,248320x5120")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    b = get_backend()
    assert b.device.type == "cuda", "needs TILERL_TARGET=cuda"

    torch.manual_seed(0)

    def timed(fn):
        # kernel time from the profiler: wall adds ~50 us of dispatch on a ~20 us kernel
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        us = 0.0
        for e in prof.events():
            if e.device_type.name == "CUDA" and "linear" in e.name:
                us += e.time_range.elapsed_us()
        return us / args.iters / 1e3

    import tilerl_kernels.backend as B

    print(f"{'N':>7} {'K':>6} {'MB':>6} {'blocks':>7} "
          f"{'gemv M1':>9} {'mma8 M8':>9} {'ratio':>6} {'mma8 GB/s':>10}")
    for shape in args.shapes.split(","):
        n, k = (int(v) for v in shape.split("x"))
        wq, s = reference.pack_fp4(torch.randn(n, k) * 0.1)
        wq, s = wq.to(b.device), s.float().to(b.device)
        bytes_w = n * k // 2 + s.numel() * 4  # e2m1 nibbles + f32 block scales

        def run(m, mgemv):
            x = torch.randn(m, k, device=b.device, dtype=torch.bfloat16)
            old, B._MGEMV = B._MGEMV, mgemv
            try:
                return timed(lambda: b.linear_fp4(x, wq, s))
            finally:
                B._MGEMV = old

        g1, m8 = run(1, 8), run(8, 0)
        print(f"{n:>7} {k:>6} {bytes_w / 1e6:>6.1f} {-(-n // 32):>7} "
              f"{g1 * 1e3:>8.1f}u {m8 * 1e3:>8.1f}u {m8 / g1:>6.2f} "
              f"{bytes_w / m8 / 1e6:>10.0f}")


if __name__ == "__main__":
    main()
