"""mma8 is register-bound: 128 regs/thread, 21.8% occupancy, 18.5% of DRAM peak,
while issuing at only 44.6%. Neither bytes nor instructions bind it — resident
warps do. Force ptxas to fit more blocks per SM and see whether the spill costs
less than the occupancy buys.

TileLang's `register_cuda_postproc` rewrites the generated source, which is how
`__launch_bounds__` gets onto a kernel it emits.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_mma8_regs.py
"""

from __future__ import annotations

import argparse
import re

import torch
from torch.profiler import ProfilerActivity, profile

import tilelang
from tilerl_kernels import kernels_linear, reference
from tilerl_kernels.backend import _pad2d, _round_up, get_backend

KERNEL = "linear_fp4_mma8_kernel"
_want = [0]


def _postproc(src: str, target) -> str:
    if _want[0] and KERNEL in src:
        # Only this kernel: the callback is global, and the GEMV next door is
        # already at 64 regs and 45% occupancy.
        src = re.sub(
            r"(__global__\s+void\s+)(__launch_bounds__\(\s*\d+\s*(?:,\s*\d+\s*)?\)\s*)?"
            + KERNEL,
            rf"\1__launch_bounds__(128, {_want[0]}) " + KERNEL,
            src,
        )
    return src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=17408)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--iters", type=int, default=50)
    # ONE value per process: tilelang caches compiled kernels by signature, and
    # the postproc is not part of it, so a second arm in the same process (or
    # against the same TILELANG_CACHE_DIR) silently reuses the first's binary.
    ap.add_argument("--blocks", type=int, default=0, help="minBlocksPerMultiprocessor; 0 = as-is")
    ap.add_argument("--dump", action="store_true", help="print the kernel's __global__ line")
    args = ap.parse_args()
    b = get_backend()
    assert b.device.type == "cuda", "needs TILERL_TARGET=cuda"

    torch.manual_seed(0)
    wq, sc = reference.pack_fp4(torch.randn(args.n, args.k) * 0.1)
    wq_nat = wq.clone()
    wq = b._served_fp4(wq.to(b.device))
    sc = sc.float().to(b.device)
    blk = args.k // sc.shape[1]
    x = torch.randn(8, args.k, device=b.device, dtype=torch.bfloat16)
    mb = (args.n * args.k // 2 + sc.numel() * 4) / 1e6
    np_ = _round_up(args.n, 32)
    w2, s2 = _pad2d(wq, np_, args.k // 2), _pad2d(sc, np_, sc.shape[1])
    osc = torch.ones(np_, dtype=torch.float32, device=b.device)
    res = torch.zeros(8, np_, dtype=torch.float32, device=b.device)
    ref = reference.linear_fp4(x.float().cpu(), wq_nat, sc.cpu())

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
    for nb in (args.blocks,):
        _want[0] = nb
        tilelang.engine.callback.register_cuda_postproc(_postproc, override=True)
        try:
            k = kernels_linear.make_linear_fp4_mma8(b.target, NG=4, KW=4, G=4)
            us = timed(lambda: k(x, w2, s2, osc, res, blk))
        except Exception as exc:
            print(f"  blocks/SM {nb or '-'}: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        got = k(x, w2, s2, osc, res, blk)[:, :args.n].float().cpu()
        rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        print(f"  blocks/SM {nb or '-':>2}: {us:7.1f} us  {mb / us * 1e3:7.0f} GB/s  rel {rel:.2e}")


if __name__ == "__main__":
    main()
