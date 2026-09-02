"""Does an f16 scale stream actually speed up the sm70 GEMV?

The scale plane is 3.22 GB of the checkpoint's measured 20.35 GB, so halving it
takes per-token weight traffic to 18.7 GB and the roofline from 44.2 to 48.1
tok/s — the largest structural item left. The conversion is NOT bit-exact (66%
of 805M values move, worst relative error 3.2e-04 — 31x inside the 1e-2 parity
gate), so this measures whether the traffic saving is real before anything in
the production path changes.

One kernel, both scale dtypes, at the shapes that carry ~90% of the weight
bytes. Also checks the two agree.

  scripts/v100.sh run sh '/usr/bin/python3 -u scripts/ab_scale_f16.py'
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

os.environ.setdefault("TILERL_TARGET", "cuda")
from tilerl_kernels import kernels_linear  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

BLK = 16  # NVFP4 block; the served checkpoint is native block-16
#: (N, K, label) — gate/up/down carry 67% of all elements, qkv 4%, lm_head 5%.
SHAPES = [
    (17408, 5120, "gate/up"),
    (5120, 17408, "down"),
    (12288, 5120, "qkv"),
    (248320, 5120, "lm_head"),
]


def ms(fn, iters=30) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def main() -> None:
    be = get_backend()
    dev = be.device
    M = 1
    print(f"# sm70 fp4 GEMV, M={M}, f32 vs f16 block scales (block={BLK})")
    print(f"{'shape':>18} {'MB f32':>8} {'MB f16':>8} {'f32 us':>8} {'f16 us':>8} "
          f"{'gain':>6} {'GB/s':>7} {'relerr':>9}")
    tot32 = tot16 = 0.0
    for N, K, label in SHAPES:
        g = torch.Generator(device="cpu").manual_seed(N)
        wq = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, generator=g).to(dev)
        s32 = (torch.rand(N, K // BLK, generator=g) * 0.5 + 0.25).to(dev)
        s16 = s32.half()
        osc = torch.ones(N, device=dev)
        res = torch.zeros(M, N, device=dev)
        x = torch.randn(M, K, generator=g).to(dev).half()

        k32 = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, xh=True)
        k16 = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, xh=True, sh=True)
        f32 = lambda: k32(x, wq, s32, osc, res, 32, 4, BLK)  # noqa: E731
        f16 = lambda: k16(x, wq, s16, osc, res, 32, 4, BLK)  # noqa: E731

        y32, y16 = f32(), f16()
        rel = ((y32 - y16).abs() / y32.abs().clamp(min=1e-3)).max().item()
        t32, t16 = ms(f32), ms(f16)
        # Weight bytes this kernel streams: nibbles + its scale plane.
        nib = N * K / 2
        mb32 = (nib + N * (K // BLK) * 4) / 1e6
        mb16 = (nib + N * (K // BLK) * 2) / 1e6
        tot32 += t32
        tot16 += t16
        print(f"{label:>10} {N:>6}x{K:<5}"[:18].rjust(18)
              + f" {mb32:>8.1f} {mb16:>8.1f} {t32*1000:>8.1f} {t16*1000:>8.1f} "
                f"{t32/t16:>6.2f}x {mb16/t16/1000:>7.0f} {rel:>9.2e}")
    print(f"\nsum {tot32*1000:.0f} -> {tot16*1000:.0f} us, {tot32/tot16:.2f}x")
    print("Checkpoint-wide: scales 3.22 -> 1.61 GB of 20.35 -> 18.74 GB/token,")
    print("roofline 44.2 -> 48.1 tok/s if the kernel converts the traffic to time.")


if __name__ == "__main__":
    main()
