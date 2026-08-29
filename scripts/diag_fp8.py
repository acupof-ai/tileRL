"""Diagnose the fp8 linear path: per-32-block activation quant error at
different K, and a direct tilelang-vs-torch e4m3 cast check.
    CUDA_VISIBLE_DEVICES=7 TILERL_TARGET=cuda PYTHONPATH=src \
        python3 scripts/diag_fp8.py
"""

from __future__ import annotations

import torch

from tilerl_kernels.backend import get_backend
from tilerl_kernels.reference import dequant_fp4, linear_fp4, pack_fp4


def main() -> None:
    backend = get_backend()
    torch.manual_seed(21)

    # --- direct e4m3 cast check: tilelang kernel vs torch -------------------
    # Quantize a known row via the kernel, dequant, compare to torch's e4m3.
    x = torch.randn(4, 128, dtype=torch.bfloat16) * 0.5
    x_dev = x.to(backend.device)
    xq = torch.empty((4, 128), dtype=torch.float8_e4m3fn, device=backend.device)
    ascale = torch.empty((4, 128 // 32), dtype=torch.float32, device=backend.device)
    backend._kernel("quant_fp8")(x_dev, xq, ascale, 32)
    # torch reference: per-32-block quant
    xt = x.float()
    blk = xt.reshape(4, 4, 32)
    blk_max = blk.abs().amax(dim=-1, keepdim=True)  # [4,4,1]
    s = (448.0 / blk_max.clamp_min(1e-12)).to(torch.float32)
    xq_ref = (blk * s).to(torch.float8_e4m3fn).reshape(4, 128)
    cast_diff = (xq.float().cpu() - xq_ref.float()).abs().max().item()
    print(f"e4m3 cast tilelang-vs-torch max diff: {cast_diff:.4e} (should be ~0)")

    # --- per-32-block quant + exact w, f32 matmul, vs reference -------------
    for M, N, K in [(8, 64, 256), (8, 64, 1024), (8, 64, 4096)]:
        w_master = torch.randn(N, K) * 0.1
        wq, scale = pack_fp4(w_master)
        x = torch.randn(M, K) * 0.5
        ref = linear_fp4(x, wq, scale)
        out = backend.linear_fp4(x, wq, scale)
        diff = (out.cpu() - ref).abs().max().item()
        rel = diff / ref.abs().max().item()
        print(f"M={M} N={N} K={K}: max abs diff {diff:.4e}, rel {rel:.4%}")

        # manual fp8: per-32-block quant x, exact w, f32 matmul
        Kp = ((K + 31) // 32) * 32
        Mp = ((M + 63) // 64) * 64
        xp = torch.nn.functional.pad(x.to(backend.device, torch.bfloat16), (0, Kp - K, 0, Mp - M))
        xq = torch.empty((Mp, Kp), dtype=torch.float8_e4m3fn, device=backend.device)
        asc = torch.empty((Mp, Kp // 32), dtype=torch.float32, device=backend.device)
        backend._kernel("quant_fp8")(xp, xq, asc, 32)
        x_deq = (xq[:M, :K].float() / asc[:M, : K // 32].repeat_interleave(32, dim=1)).cpu()
        w_deq = dequant_fp4(wq, scale)
        manual = x_deq @ w_deq.t()
        mdiff = (manual - ref).abs().max().item() / ref.abs().max().item()
        print(f"  manual fp8 (per-32-block x, exact w) rel err: {mdiff:.4%}")


if __name__ == "__main__":
    main()
