"""mma8 moves weight bytes at ~0.68 TB/s vs the GEMV's ~1.95. Sweep (NG, KW, G, W8) on the
factory and time the kernel, not the Python call.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_mma8_tiles.py
"""

from __future__ import annotations

import argparse

import torch
from tilerl_kernels import kernels_linear, reference
from tilerl_kernels.backend import _pad2d, _round_up, get_backend
from torch.profiler import ProfilerActivity, profile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=17408)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--iters", type=int, default=50)
    # NGxKWxGxW8 — W8=1 loads 8 bytes of weight per lane instead of 4
    ap.add_argument("--combos", default="4x4x4x0,4x4x4x1,4x2x4x1,2x4x4x1,4x4x2x1")
    args = ap.parse_args()
    b = get_backend()
    assert b.device.type == "cuda", "needs TILERL_TARGET=cuda"

    torch.manual_seed(0)
    wq, sc = reference.pack_fp4(torch.randn(args.n, args.k) * 0.1)
    wq_nat = wq.clone()  # _served_fp4 twiddles in place; the reference wants natural order
    wq = b._served_fp4(wq.to(b.device))
    sc = sc.float().to(b.device)
    blk = args.k // sc.shape[1]
    x = torch.randn(8, args.k, device=b.device, dtype=torch.bfloat16)
    mb = (args.n * args.k // 2 + sc.numel() * 4) / 1e6

    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        us = sum(e.time_range.elapsed_us() for e in prof.events()
                 if e.device_type.name == "CUDA" and "linear" in e.name)
        return us / args.iters

    print(f"N={args.n} K={args.k}  {mb:.1f} MB, M=8")
    for combo in args.combos.split(","):
        ng, kw, g, w8 = (int(v) for v in combo.split("x"))
        nb = ng * 8
        np_ = _round_up(args.n, nb)
        w2, s2 = _pad2d(wq, np_, args.k // 2), _pad2d(sc, np_, sc.shape[1])
        osc = torch.ones(np_, dtype=torch.float32, device=b.device)
        res = torch.zeros(8, np_, dtype=torch.float32, device=b.device)
        try:
            k = kernels_linear.make_linear_fp4_mma8(b.target, NG=ng, KW=kw, G=g, W8=w8)
            us = timed(lambda: k(x, w2, s2, osc, res, blk))
        except Exception as exc:  # a combo the schedule cannot express
            print(f"  NG={ng} KW={kw} G={g} W8={w8}: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        ref = reference.linear_fp4(x.float().cpu(), wq_nat, sc.cpu())
        got = k(x, w2, s2, osc, res, blk)[:, :args.n].float().cpu()
        rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        print(f"  NG={ng} KW={kw} G={g} W8={w8}: {us:7.1f} us  {mb / us * 1e3:7.0f} GB/s  "
              f"{np_ // nb:>5} blocks  rel {rel:.2e}")


if __name__ == "__main__":
    main()
