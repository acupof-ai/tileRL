"""A/B the sm70 M-row GEMV against its X-as-f16 twin at the real projections.

The f32 rungs re-read X from global in every block and convert it to f16 inside
the tile loop. Packing X once should remove 32 of ~49 per-row instructions and
half the X bytes; this measures whether it does.

  python3 scripts/ab_gemv_xh.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src"))

import torch
from tilerl_kernels import kernels_linear, reference
from tilerl_kernels.backend import _pad2d, get_backend

SHAPES = [(17408, 5120), (12288, 5120), (5120, 17408)]
bk = get_backend()
g = torch.Generator().manual_seed(1)
# The f32 rungs are no longer registered (the ladder is f16-X only), so the
# baseline arm is built straight from the factory.
_F32 = {M: kernels_linear.make_linear_fp4_gemv_sm70_m(bk.target, M=M) for M in (2, 4, 8)}


def timeit(k, args, n=20):
    for _ in range(3):
        y = k(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        y = k(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6, y


def run(name, x, wq, sc, osc, res, bN, blk, n=20, *fargs):
    return timeit(bk._kernel(name, *fargs), (x, wq, sc, osc, res, 32, bN, blk), n)


for N, K in SHAPES:
    w = torch.randn(N, K, generator=g) * 0.1
    wq, sc = reference.pack_fp4(w, block=16)
    wq = reference.twiddle_fp4_f16(wq)
    _, _, Np, Kp, _, bN = bk._plan("linear_fp4", 1, N, K)
    wq1 = _pad2d(wq.to(bk.device), Np, Kp // 2)
    sc1 = _pad2d(sc.to(bk.device), Np, Kp // 16)
    osc1 = bk._ones(Np)
    print(f"\nN={N} K={K}")
    x1 = _pad2d(torch.randn(1, K, generator=g).to(bk.device), 1, Kp)
    r1 = bk._zeros2(1, Np)
    t_g, y_g = run("linear_fp4_gemv", x1, wq1, sc1, osc1, r1, bN, 16)
    t_1h, y_1h = run("linear_fp4_gemv_sm70_m", x1.half(), wq1, sc1, osc1, r1, bN, 16,
                     20, 1, 4, True)
    d = (y_g[:, :N] - y_1h[:, :N]).abs().max().item()
    print(
        f"  M=1  gemv {t_g:7.1f} us            m1h {t_1h:7.1f} us            "
        f"{t_g / t_1h:.2f}x   absdiff {d:.2e}"
    )
    for M in (2, 4, 8):
        x = _pad2d(torch.randn(M, K, generator=g).to(bk.device), M, Kp)
        res = bk._zeros2(M, Np)
        t_f32, y_f32 = timeit(_F32[M], (x, wq1, sc1, osc1, res, 32, bN, 16))
        t_f16, y_f16 = run(
            "linear_fp4_gemv_sm70_m", x.half(), wq1, sc1, osc1, res, bN, 16, 20, M, 4, True
        )
        d = (y_f32[:, :N] - y_f16[:, :N]).abs().max().item()
        s = y_f32[:, :N].abs().max().item()
        print(
            f"  M={M}  f32 {t_f32:7.1f} us ({t_f32 / M:6.1f}/row)   "
            f"xh {t_f16:7.1f} us ({t_f16 / M:6.1f}/row)   "
            f"{t_f32 / t_f16:.2f}x   relerr {d / max(s, 1e-9):.2e}"
        )
