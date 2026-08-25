"""Diag: old (per-element dequant) vs new (vectorized shared dequant) fp4
prefill kernel, same process, back-to-back — the ratio is the dequant-schedule
win, contention-independent. Builds the OLD kernel inline (git HEAD version)
and the NEW kernel from kernels_mma, benches both at MLP prefill shapes.

Usage:
    TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=3 \\
        PYTHONPATH=src python3 scripts/diag_dequant.py
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T

from tilerl.ops import kernels_mma
from tilerl.ops.backend import _pad2d, _round_up, _THREADS, get_backend
from tilerl.ops.reference import pack_fp4


def old_linear_fp4_fp8_mma(target: str):
    """git HEAD version: per-element T.Parallel dequant in the K-loop,
    scale from global per element, block_K=64, num_stages=3."""
    _BLOCK_K = 64

    @tilelang.jit(
        target=target,
        pass_configs={
            "tl.disable_data_race_check": True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def linear_fp4_fp8(XQ, WQ, WScale, AScale, block_M, block_N, threads):
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        WQ: T.Tensor((N, K // 2), "uint8")
        WScale: T.Tensor((N, K // 32), "float32")
        AScale: T.Tensor((M,), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, _BLOCK_K), "float8_e4m3fn")
            WQ_shared = T.alloc_shared((block_N, _BLOCK_K // 2), "uint8")
            W_shared = T.alloc_shared((block_N, _BLOCK_K), "float8_e4m3fn")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _BLOCK_K, num_stages=3):
                T.copy(XQ[by * block_M, k * _BLOCK_K], X_shared)
                T.copy(WQ[bx * block_N, k * _BLOCK_K // 2], WQ_shared)
                for i, j in T.Parallel(block_N, _BLOCK_K):
                    byte = WQ_shared[i, j // 2]
                    nib = (byte >> ((j % 2) * 4)) & 15
                    ni32 = T.cast(nib, "int32")
                    bits = (
                        ((ni32 & 8) << 28) | ((126 + ((ni32 >> 1) & 3)) << 23) | ((ni32 & 1) << 22)
                    )
                    w = T.reinterpret(bits, "float32")
                    W_shared[i, j] = T.cast(
                        w * WScale[bx * block_N + i, (k * _BLOCK_K + j) // 32],
                        "float8_e4m3fn",
                    )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] / AScale[by * block_M + i]
            T.copy(C_local, Y[by * block_M, bx * block_N])
        return Y

    return linear_fp4_fp8


def _bench(fn, warmup=5, rep=20):
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
    shapes = [
        (512, 5120, 17408),  # gate/up
        (512, 17408, 5120),  # down
        (512, 5120, 10240),  # in_proj_qkv
        (512, 6144, 5120),  # out_proj
    ]
    old_k = old_linear_fp4_fp8_mma("cuda")
    new_k = kernels_mma.make_linear_fp4_fp8_mma("cuda")
    for M, K, N in shapes:
        w_master = torch.randn(N, K) * 0.1
        wq, scale = pack_fp4(w_master)
        x = torch.randn(M, K, device=backend.device, dtype=torch.bfloat16) * 0.5
        wq = wq.to(backend.device)
        scale = scale.to(backend.device)
        bM, bN = _round_up(min(128, M), 16), _round_up(min(128, N), 32)
        Mp, Np, Kp = _round_up(M, bM), _round_up(N, bN), _round_up(K, 64)
        x2 = _pad2d(x, Mp, Kp)
        xq = torch.empty((Mp, Kp), dtype=torch.float8_e4m3fn, device=backend.device)
        ascale = torch.empty((Mp,), dtype=torch.float32, device=backend.device)
        backend._kernel("quant_fp8")(x2, xq, ascale, 256)
        wqp = _pad2d(wq, Np, Kp // 2)
        sp = _pad2d(scale, Np, Kp // 32)
        args = (xq, wqp, sp, ascale, bM, bN, _THREADS)
        y_old = old_k(*args)
        y_new = new_k(*args)
        diff = (y_old[:M, :N] - y_new[:M, :N]).abs().max().item()
        ref = y_old[:M, :N].abs().max().item()
        t_old = _bench(lambda: old_k(*args))
        t_new = _bench(lambda: new_k(*args))
        flops = 2 * M * N * K
        print(
            f"M={M} K={K} N={N:>5}: old {t_old:7.3f}ms ({flops / t_old / 1e9:6.1f} TF)  "
            f"new {t_new:7.3f}ms ({flops / t_new / 1e9:6.1f} TF)  speedup {t_old / t_new:.2f}x  "
            f"maxdiff {diff:.1e} ({ref:.1e})",
            flush=True,
        )


if __name__ == "__main__":
    main()
