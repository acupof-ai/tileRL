"""Is the rung-8 cliff hardware, or a missing flag?

backend.py hands X to the sm70 GEMV pre-packed as f16 (``xh=True``) only on the
M<=8 branch; the M>8 branch calls the same factory WITHOUT it. That is what makes
M>8 cost 127 us/row against rung 8's 22 — and the packing is the documented reason
the ladder is fast at all ("Packing collapses 127 us/row flat to 24-45 us/row").

The extern is ``tl_fp4_gemv_tiles_f16_m_xh<G, M>``, templated on M with no upper
bound, so nothing in the kernel restricts packed X to 8 rows. If that is true, the
cliff is a dispatch omission and not a property of the hardware, which changes
several conclusions at once: batched verify (rows = B*W, so B=4 depth 3 is already
M=16), prefill chunking, and every "we cannot go wider than 8" argument.

Both arms at the shapes and row counts that matter, bit-exactness checked — both
paths round X to nearest f16, so a difference means the packed variant is reading
the wrong bytes at that M, not losing precision.

  scripts/v100.sh run xh '/usr/bin/python3 -u scripts/ab_gemv_xh_m32.py'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

os.environ.setdefault("TILERL_TARGET", "cuda")
import benchkit as bk  # noqa: E402
from tilerl_kernels import kernels_linear, reference  # noqa: E402
from tilerl_kernels.backend import _pad2d, get_backend  # noqa: E402

#: The two shapes carrying 67% of the weight bytes, plus lm_head.
SHAPES = [(17408, 5120), (5120, 17408), (248320, 5120)]
#: M=16 is B=4 at depth 3 — already served today by the unpacked M=32 kernel.
MS = [16, 32]
BLK = 32


def main() -> None:
    be = get_backend()
    dev = be.device
    print("# sm70 fp4 GEMV above rung 8: X unpacked (shipped) vs pre-packed f16")
    print(f"{'shape':>20} {'M':>3} {'f32 us':>8} {'/row':>7} {'xh us':>8} {'/row':>7} "
          f"{'gain':>6} {'absdiff':>9}")
    for N, K in SHAPES:
        g = torch.Generator().manual_seed(N * 7 + K)
        w = (torch.randn(N, K, generator=g) * 0.1)
        wq, sc = reference.pack_fp4(w, block=BLK)
        wq = reference.twiddle_fp4_f16(wq).to(dev)
        _, _, Np, Kp, _, bN = be._plan("linear_fp4", 1, N, K)
        wq1 = _pad2d(wq, Np, Kp // 2)
        sc1 = _pad2d(sc.to(dev), Np, Kp // BLK)
        osc = be._ones(Np)
        for M in MS:
            x = _pad2d(torch.randn(M, K, generator=g).to(dev), M, Kp)
            res = be._zeros2(M, Np)
            k32 = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M)
            kxh = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, xh=True)
            f32 = lambda: k32(x, wq1, sc1, osc, res, 32, bN, BLK)  # noqa: E731
            fxh = lambda: kxh(x.half(), wq1, sc1, osc, res, 32, bN, BLK)  # noqa: E731
            y32, yxh = f32(), fxh()
            # Both round X to nearest f16, so this should be ~0. A real difference
            # means the packed extern reads the wrong bytes at this M.
            d = (y32[:, :N] - yxh[:, :N]).abs().max().item()
            u32, uxh = bk.timeit(f32), bk.timeit(fxh)
            print(f"{f'{N}x{K}':>20} {M:>3} {u32 * 1000:>8.1f} {u32 * 1000 / M:>7.1f} "
                  f"{uxh * 1000:>8.1f} {uxh * 1000 / M:>7.1f} {u32 / uxh:>5.2f}x {d:>9.2e}")
    print("\nIf xh wins at M=32 with absdiff ~0, the rung-8 cliff is a dispatch")
    print("omission (backend.py passes xh only on the M<=8 branch), not hardware.")


if __name__ == "__main__":
    main()
