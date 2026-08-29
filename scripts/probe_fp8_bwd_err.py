"""Where does the fp8 dX kernel's error come from — the weight cast or the
gradient cast?

The kernel dequantizes W8*scale into bf16 inside the K-loop, and the caller
casts the gradient to bf16 for the WGMMA. Both round; only one is worth fixing.
Compares against the f32 reference and against two f32 references that emulate
one rounding each.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_fp8_bwd_err.py
"""

from __future__ import annotations

import torch

from tilerl_kernels import reference as ref
from tilerl_kernels.backend import get_backend


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Norm-relative error. NOT the elementwise max ratio: a dX output has
    cancellation, so elements land near zero and that ratio reports thousands
    for a result that is fine."""
    return float((a - b).norm() / b.norm())


def main() -> None:
    b = get_backend()
    torch.manual_seed(11)
    for m, n, k in ((8, 256, 128), (64, 5120, 5120), (256, 248320, 5120)):
        dev = b.device
        w8 = (torch.randn(n, k, device=dev) * 0.3).to(torch.float8_e4m3fn)
        scale = (torch.rand(n // 128, k // 128, device=dev) + 0.5).float()
        osc = (torch.rand(n, device=dev) + 0.5).float()
        g = torch.randn(m, n, device=dev)
        exact = ref.linear_frozen_bwd(g, w8, scale, oscale=osc, fp8=True).float()
        got = b.linear_frozen_bwd(g, w8, scale, oscale=osc, fp8=True).float()

        s = scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
        w = w8.float() * s
        ga = (g * osc.reshape(1, -1))
        w_bf = w.bfloat16().float()          # the kernel's weight rounding
        g_bf = ga.bfloat16().float()         # the caller's gradient cast
        print(f"  M={m} N={n} K={k}")
        print(f"    kernel vs f32 reference     {rel(got, exact):.4f}")
        print(f"    weight->bf16 only           {rel(ga @ w_bf, exact):.4f}")
        print(f"    gradient->bf16 only         {rel(g_bf @ w, exact):.4f}")
        print(f"    both                        {rel(g_bf @ w_bf, exact):.4f}")
        del w8, scale, g, exact, got, s, w, ga, w_bf, g_bf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
