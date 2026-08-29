"""Micro-benchmark the fp8 prefill path: quant kernel vs fp8 GEMM vs bf16 GEMM
at prefill shapes, isolated with CUDA events.
    CUDA_VISIBLE_DEVICES=7 TILERL_TARGET=cuda PYTHONPATH=src \
        python3 scripts/bench_fp8_prefill.py
"""

from __future__ import annotations

import torch

from tilerl_kernels.backend import get_backend
from tilerl_kernels.reference import pack_fp4


def _bench(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


def main() -> None:
    backend = get_backend()
    torch.manual_seed(0)
    # prefill shapes: (M, K, N) for the big linears
    shapes = [
        (512, 5120, 17408),  # gate/up
        (512, 17408, 5120),  # down
        (512, 5120, 10240),  # in_proj_qkv
        (512, 5120, 6144),  # in_proj_z
        (512, 6144, 5120),  # out_proj
    ]
    for M, K, N in shapes:
        w_master = torch.randn(N, K) * 0.1
        wq, scale = pack_fp4(w_master)
        x = torch.randn(M, K) * 0.5
        x = backend._dev(x, torch.bfloat16)
        wq = backend._dev(wq, wq.dtype)
        scale = backend._f32(scale)

        # bf16 path (pop the fp8 key so it falls through to bf16)
        from tilerl_kernels.registry import _REGISTRY, _resolve

        cell = _REGISTRY[("bf16", "sm90")]
        had_fp8 = "linear_fp4_fp8" in cell
        if had_fp8:
            cell.pop("linear_fp4_fp8")
        try:
            t_bf16 = _bench(lambda: backend.linear_fp4(x, wq, scale))
        finally:
            if had_fp8:
                from tilerl_kernels import kernels_mma

                cell["linear_fp4_fp8"] = kernels_mma.make_linear_fp4_fp8_mma

        # fp8 path
        t_fp8 = _bench(lambda: backend.linear_fp4(x, wq, scale))

        # quant kernel alone
        BK = 32
        Mp = ((M + 63) // 64) * 64
        Kp = ((K + 31) // 32) * 32
        xp = torch.nn.functional.pad(x, (0, Kp - K, 0, Mp - M))
        xq = torch.empty((Mp, Kp), dtype=torch.float8_e4m3fn, device=backend.device)
        ascale = torch.empty((Mp,), dtype=torch.float32, device=backend.device)
        t_quant = _bench(lambda: backend._kernel("quant_fp8")(xp, xq, ascale, BK))

        flops = 2 * M * N * K
        print(
            f"M={M} K={K} N={N:>5}: bf16 {t_bf16:7.3f}ms ({flops / t_bf16 / 1e9:6.1f} TFLOP/s)  "
            f"fp8 {t_fp8:7.3f}ms ({flops / t_fp8 / 1e9:6.1f} TFLOP/s)  "
            f"quant {t_quant:6.3f}ms  speedup {t_bf16 / t_fp8:.2f}x"
        )


if __name__ == "__main__":
    main()
