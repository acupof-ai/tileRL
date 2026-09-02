"""mma8 vs the unreachable `linear_fp4_fp8_decode` (w4a8) at M=8. Both go through
`Backend.linear_fp4`, so the A/B is `_MX`: at 7, M=8 falls through to the w4a8 path.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_fp4_decode_paths.py
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

import tilerl_kernels.backend as B
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend

SHAPES = "8192x5120,5120x6144,17408x5120,5120x17408,10240x5120,6144x5120"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default=SHAPES)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    b = get_backend()
    assert b.device.type == "cuda", "needs TILERL_TARGET=cuda"

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

    torch.manual_seed(0)
    print(f"M={args.m}   (rel is vs the f32 dequant reference; the w4a8 path also "
          f"quantizes the ACTIVATION to e4m3, so its error is expected to be larger)")
    print(f"{'N':>7} {'K':>6} {'mma8':>9} {'w4a8':>9} {'ratio':>6} "
          f"{'rel mma8':>10} {'rel w4a8':>10}")
    for shape in args.shapes.split(","):
        n, k = (int(v) for v in shape.split("x"))
        wq, sc = reference.pack_fp4(torch.randn(n, k) * 0.1)
        wq_nat, sc_nat = wq.clone(), sc.clone().float()
        wq, sc = wq.to(b.device), sc.float().to(b.device)
        x = torch.randn(args.m, k, device=b.device, dtype=torch.bfloat16)
        ref = reference.linear_fp4(x.float().cpu(), wq_nat, sc_nat)
        out = {}
        for name, mx in (("mma8", 8), ("w4a8", 7)):
            old, B._MX = B._MX, mx
            try:
                us = timed(lambda: b.linear_fp4(x, wq, sc))
                got = b.linear_fp4(x, wq, sc).float().cpu()
            finally:
                B._MX = old
            rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
            out[name] = (us, rel)
        print(f"{n:>7} {k:>6} {out['mma8'][0]:>8.1f}u {out['w4a8'][0]:>8.1f}u "
              f"{out['w4a8'][0] / out['mma8'][0]:>6.2f} "
              f"{out['mma8'][1]:>10.1e} {out['w4a8'][1]:>10.1e}")


if __name__ == "__main__":
    main()
