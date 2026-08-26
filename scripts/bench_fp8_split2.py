"""A/B the 2-way K-split fp8 prefill GEMM (v_n64_split2) against the shipped
k_split=1 launch — the output-buffer zeroing the split needs is INSIDE arm B's
timed region (the sweep that found +8% geo-mean excluded it).

Arm A: make_linear_fp4_fp8_mma(target, k_split=1) — the shipped kernel.
Arm B: torch.zeros([M,N] f32) then k_split=2 — f32 atomic add into the zeroed
       buffer (the zeroing cost the sweep excluded, now included).
Ref: arm A's output (same fp8 math, split reduction order -> relerr ~1e-3).

Usage (pod):
    BENCH_GPUS=6 scripts/_pod_bench.sh \\
        'PYTHONPATH=src python3 scripts/bench_fp8_split2.py'
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "src")

import torch

import benchkit
from tilerl.ops.kernels_linear import make_linear_fp4_fp8_mma, make_quant_fp8_e4m3
from tilerl.ops.reference import pack_fp4

SHAPES = [
    (512, 5120, 17408),  # gate/up
    (512, 17408, 5120),  # down
    (512, 5120, 10240),  # in_proj_qkv
    (512, 5120, 6144),  # in_proj_z
    (512, 6144, 5120),  # out_proj
]


def _round_up(x, m):
    return ((x + m - 1) // m) * m


def main():
    dev = "cuda"
    quant = make_quant_fp8_e4m3(dev)
    ker_a = make_linear_fp4_fp8_mma(dev, k_split=1)
    ker_b = make_linear_fp4_fp8_mma(dev, k_split=2)
    ratios = []
    for M, K, N in SHAPES:
        torch.manual_seed(0)
        w_master = torch.randn(N, K) * 0.1
        wq, scale = pack_fp4(w_master)
        x = torch.randn(M, K) * 0.5
        # K padded to 128 = _FP4_BLOCK_K * k_split so each split sums an exact
        # tile count; N padded to 128 (a multiple of the kernel's 64 N-tile).
        Mp, Np, Kp = _round_up(M, 64), _round_up(N, 128), _round_up(K, 128)
        x2 = torch.zeros(Mp, Kp, dtype=torch.bfloat16, device=dev)
        x2[:M, :K] = x.to(dev)
        xq = torch.empty((Mp, Kp), dtype=torch.float8_e4m3fn, device=dev)
        ascale = torch.empty((Mp,), dtype=torch.float32, device=dev)
        quant(x2, xq, ascale, 256)
        wq_p = torch.zeros(Np, Kp // 2, dtype=torch.uint8, device=dev)
        wq_p[:N, : K // 2] = wq.to(dev)
        scale_p = torch.zeros(Np, Kp // 32, dtype=torch.float32, device=dev)
        scale_p[:N, : K // 32] = scale.to(dev)

        def arm_a():
            return (ker_a(xq, wq_p, scale_p, ascale, 128, 128, 128)[:M, :N],)

        def arm_b():
            y = torch.zeros((Mp, Np), dtype=torch.float32, device=dev)
            ker_b(xq, wq_p, scale_p, ascale, y, 128, 128, 128)
            return (y[:M, :N],)

        ref = arm_a()
        rows = benchkit.ab(
            f"fp8 prefill split2 M={M} K={K} N={N}",
            [("A shipped k_split=1", arm_a), ("B split2+zero", arm_b)],
            ref,
        )
        ms_a, ms_b = rows[0][1], rows[1][1]
        ratios.append(ms_a / ms_b)
        print(f"speedup B/A: {ms_a / ms_b:.3f}x", flush=True)
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f"\ngeo-mean speedup B/A (zeroing included): {geo:.3f}x")
    print("per-shape B/A: " + ", ".join(f"{r:.3f}" for r in ratios))


if __name__ == "__main__":
    main()
