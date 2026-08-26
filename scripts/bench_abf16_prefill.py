"""A/B the A=bf16 prefill GEMM arm against the shipped A=e4m3 path (W=4 fixed).

Arm A (shipped): per-token e4m3 activation quant + fp4->e4m3 dequant + fp8
WGMMA, k_split=2 (the backend's M>1 sm90 path, body mirrored end-to-end:
pad + quant + output zeroing + split kernel — all inside the timed region).
Arm B (bf16): no activation quant; fp4->bf16 dequant + bf16 WGMMA
(`make_linear_fp4_mma`, the pre-fp8 prefill kernel — the natural flip
candidate, already registered as `linear_fp4`).
Ref: `reference.linear_fp4` (torch-eager f32 dequant GEMM). Arm B's relerr is
the pure W4 error; arm A's relerr minus that is the A-quant contribution.

Usage (pod):
    BENCH_GPUS=7 scripts/_pod_bench.sh \\
        'PYTHONPATH=src python3 scripts/bench_abf16_prefill.py'
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "src")

import torch

import benchkit
from tilerl.ops import reference
from tilerl.ops.kernels_linear import (
    make_linear_fp4_fp8_mma,
    make_linear_fp4_mma,
    make_quant_fp8_e4m3,
)

SHAPES = [
    (512, 5120, 17408),  # gate/up
    (512, 17408, 5120),  # down
    (512, 5120, 10240),  # in_proj_qkv
    (512, 5120, 6144),  # in_proj_z
    (512, 6144, 5120),  # out_proj
]


def _ru(x, m):
    return ((x + m - 1) // m) * m


def _snap(m, cap):
    # backend._snap_mma_tile: warp-partition-valid WGMMA M tiles
    return min(cap, next((s for s in (16, 32, 64, 128) if s >= m), 128))


def main():
    dev = "cuda"
    quant = make_quant_fp8_e4m3(dev)
    ker_a = make_linear_fp4_fp8_mma(dev, k_split=2)
    ker_b = make_linear_fp4_mma(dev)
    ratios = []
    for M, K, N in SHAPES:
        torch.manual_seed(0)
        wq, scale = reference.pack_fp4(torch.randn(N, K) * 0.1)
        wq, scale = wq.to(dev), scale.to(dev)
        x = (torch.randn(M, K) * 0.5).to(torch.bfloat16).to(dev)

        def arm_a():
            bM, bN = _snap(M, 128), _ru(min(128, N), 32)
            Mp, Np, Kp = _ru(M, bM), _ru(N, bN), _ru(K, 128)
            x2 = torch.nn.functional.pad(x, (0, Kp - K, 0, Mp - M))
            xq = torch.empty((Mp, Kp), dtype=torch.float8_e4m3fn, device=dev)
            ascale = torch.empty((Mp,), dtype=torch.float32, device=dev)
            quant(x2, xq, ascale, 256)
            y = torch.zeros((Mp, Np), dtype=torch.float32, device=dev)
            ker_a(
                xq,
                torch.nn.functional.pad(wq, (0, Kp // 2 - wq.shape[1], 0, Np - N)),
                torch.nn.functional.pad(scale, (0, Kp // 32 - scale.shape[1], 0, Np - N)),
                ascale,
                y,
                bM,
                bN,
                64,
            )
            return (y[:M, :N],)

        def arm_b():
            bM, bN = _snap(M, 64), _ru(min(64, N), 16)
            Mp, Np, Kp = _ru(M, bM), _ru(N, bN), _ru(K, 64)
            x2 = torch.nn.functional.pad(x, (0, Kp - K, 0, Mp - M))
            y = ker_b(
                x2,
                torch.nn.functional.pad(wq, (0, Kp // 2 - wq.shape[1], 0, Np - N)),
                torch.nn.functional.pad(scale, (0, Kp // 32 - scale.shape[1], 0, Np - N)),
                bM,
                bN,
                64,
            )
            return (y[:M, :N],)

        ref = (reference.linear_fp4(x, wq, scale),)
        rows = benchkit.ab(
            f"prefill W4 A-precision M={M} K={K} N={N}",
            [("A shipped e4m3+fp8 ksplit2", arm_a), ("B bf16 no-quant", arm_b)],
            ref,
        )
        ms_a, ms_b = rows[0][1], rows[1][1]
        ratios.append(ms_a / ms_b)
        print(f"B/A speedup: {ms_a / ms_b:.3f}x", flush=True)
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f"\ngeo-mean B/A: {geo:.3f}x")
    print("per-shape B/A: " + ", ".join(f"{r:.3f}" for r in ratios))


if __name__ == "__main__":
    main()
