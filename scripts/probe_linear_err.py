"""Is a failing linear parity test a wrong kernel or a wrong metric? Reports the norm-relative
error beside the gate's max|a-b|, plus the bf16-accumulation floor.

    CUDA_VISIBLE_DEVICES=7 python3 scripts/probe_linear_err.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_ops_parity import _linear_fp4_fp8_ref, _quantize_fp4  # noqa: E402
from tilerl_kernels import reference  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402


def rel(a, b) -> float:
    return float((a - b).norm() / b.norm())


def report(name, out, ref, extra="") -> None:
    out, ref = out.cpu().float(), ref.cpu().float()
    mx = (out - ref).abs().max().item()
    gate = 0.01 + 0.01 * ref.abs().max().item()
    print(f"  {name:26} norm-rel {rel(out, ref):>9.2e}   gate {mx:>9.2e} vs {gate:>8.2e}"
          f"  {'PASS' if mx <= gate else 'FAIL'} {extra}")


def main() -> None:
    b = get_backend()
    cuda = b.target.startswith("cuda")
    print(f"target {b.target}")

    # M=1 takes the scalar GEMV, 2..16 the mma8 path, >16 the prefill tile.
    torch.manual_seed(4)
    w = torch.randn(24, 32)
    wq, scale = _quantize_fp4(w)
    for M in (1, 2, 3, 6, 8, 16, 17, 32):
        x = torch.randn(M, 32)
        ref4 = _linear_fp4_fp8_ref(x, wq, scale) if cuda else reference.linear_fp4(x, wq, scale)
        # if the kernel tracks the f32 reference, the gate's fp8 re-quantization is the gap
        ref_f32 = reference.linear_fp4(x, wq, scale)
        got = b.linear_fp4(x, wq, scale)
        report(f"linear_fp4 M={M:<2} vs fp8ref", got, ref4)
        report(f"linear_fp4 M={M:<2} vs f32ref", got, ref_f32)

    # native fp8: M=1 GEMV, 2.._MX mma8, above that the plan's tiled kernel.
    from test_ops_parity import _linear_fp8_ref, _quantize_fp8

    torch.manual_seed(26)
    for M, N, K in ((1, 128, 256), (4, 128, 256), (8, 128, 256), (16, 128, 256),
                    (8, 256, 128), (4, 256, 128)):
        w8, wscale = _quantize_fp8(torch.randn(N, K) * 0.1)
        xf = torch.randn(M, K) * 0.5
        got = b.linear_fp8(xf, w8, wscale)
        report(f"linear_fp8 M={M:<2} N={N} K={K}", got, _linear_fp8_ref(xf, w8, wscale))

    # What the kernel is allowed to lose: the same product in bf16 vs f32.
    for M, N, K in ((6, 24, 32), (8, 128, 256), (64, 512, 1024)):
        g = torch.Generator().manual_seed(0)
        xf = torch.randn(M, K, generator=g)
        wf = torch.randn(N, K, generator=g) * 0.1
        lo = xf.bfloat16().float() @ wf.bfloat16().float().t()
        hi = xf @ wf.t()
        report(f"bf16-accum floor {M}x{N}x{K}", lo, hi)


if __name__ == "__main__":
    main()
