"""Split the fp8 linear's 2.5% into its two possible sources.

The kernel and the reference agree on paper: bf16 input, per-token scale,
FP8_MAX=448. So the gap is either the QUANT (rounding) or the GEMM (where
the weight scale lands). Compare the quantized activation directly, then
feed the reference's own xq through the kernel's weight path.

    CUDA_VISIBLE_DEVICES=7 python3 scripts/probe_fp8_quant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_ops_parity import _linear_fp8_ref, _quantize_fp8  # noqa: E402
from tilerl_kernels import reference  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402


def rel(a, b) -> float:
    return float((a - b).norm() / b.norm())


def main() -> None:
    b = get_backend()
    torch.manual_seed(26)
    M, N, K = 8, 128, 256
    w8, wscale = _quantize_fp8(torch.randn(N, K) * 0.1)
    x = torch.randn(M, K) * 0.5

    # --- the kernel's own quantizer, read back
    x2 = b._c(x.to(b.device).to(torch.bfloat16))
    xq = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=b.device)
    ascale = torch.empty((M,), dtype=torch.float32, device=b.device)
    b._kernel("quant_fp8")(x2, xq, ascale, 256)
    kern_x = (xq.float() / ascale.unsqueeze(1)).cpu()

    # --- the reference's quantizer
    xbf = x.to(torch.bfloat16).float()
    row_max = xbf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    rs = 448.0 / row_max
    ref_x = (xbf * rs).to(torch.float8_e4m3fn).float() / rs

    print(f"target {b.target}")
    print(f"  quantized activation   kernel vs ref   norm-rel {rel(kern_x, ref_x):.3e}")
    print(f"  scale                  kernel vs ref   norm-rel "
          f"{rel(ascale.cpu(), rs.squeeze(1)):.3e}")
    print(f"  e4m3 quant error alone (ref vs f32)    norm-rel {rel(ref_x, xbf):.3e}")

    # --- same weights, both activations, in f32: isolates the GEMM
    w = reference.dequant_fp8(w8, wscale)
    print(f"  f32 matmul on the kernel's own xq      norm-rel "
          f"{rel(kern_x @ w.t(), ref_x @ w.t()):.3e}")
    got = b.linear_fp8(x, w8, wscale).cpu().float()
    print(f"  full kernel vs reference               norm-rel "
          f"{rel(got, _linear_fp8_ref(x, w8, wscale)):.3e}")
    print(f"  full kernel vs f32 on ITS OWN xq       norm-rel "
          f"{rel(got, kern_x @ w.t()):.3e}")
    # If the kernel tracks the UNQUANTIZED activation instead, this path is
    # w8a16 and the w8a8 reference is modelling the wrong kernel.
    print(f"  full kernel vs bf16 activation (w8a16)  norm-rel "
          f"{rel(got, xbf @ w.t()):.3e}")
    for M2 in (1, 4, 8, 16, 32):
        xm = torch.randn(M2, K) * 0.5
        xmb = xm.to(torch.bfloat16).float()
        rmax = xmb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        s2 = 448.0 / rmax
        xq2 = (xmb * s2).to(torch.float8_e4m3fn).float() / s2
        g = b.linear_fp8(xm, w8, wscale).cpu().float()
        print(f"  M={M2:<3} vs w8a8 {rel(g, xq2 @ w.t()):.3e}   "
              f"vs w8a16 {rel(g, xmb @ w.t()):.3e}")


if __name__ == "__main__":
    main()
