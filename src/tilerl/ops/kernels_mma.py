"""MMA (tensor-core) TileLang kernels for sm90 — SOTA schedules ported from
the tilelang examples. Registered only in the sm90 cell of the dispatch
matrix (backend.py); kernels.py keeps the portable floor (CPU T.gemm + naive
FMA) for cpu/metal. The MMA schedules do not lower on CPU (T.gemm -> WGMMA
only on sm90), which is why they live here, not in kernels.py.

All kernels are f32-IO (the backend casts bf16 at the boundary; eager JIT
does not specialize on dtype) and lower to TF32 WGMMA on sm90.
# ponytail: f32 IO day-1, bf16 IO day-2 (2x WGMMA throughput)
"""

from __future__ import annotations

import tilelang
import tilelang.language as T

__all__ = [
    "make_gemm_nt_mma",
    "make_gemm_nn_mma",
    "make_gemm_tn_mma",
    "make_linear_fp4_mma",
]

#: Reduction-tile size (K for gemm_nt/nn, M for gemm_tn, K for linear_fp4).
#: WGMMA K on sm90 is 8 (TF32); 32 is 4 K-steps, divides every model K dim
#: (all are multiples of 32), and matches examples/gemm/example_gemm.py.
#: The backend pads the reduction dim to a multiple of this on CUDA.
_RED_TILE = 32


def _pass_configs() -> dict[str, object]:
    # The static race check false-positives on per-thread fragments (same as
    # the cpu/metal cells in kernels.py).
    return {"tl.disable_data_race_check": True}


# ---------------------------------------------------------------- gemm (MMA)


def make_gemm_nt_mma(target: str):
    """C = A @ B.T + Bias. A [M,K], B [N,K] -> C [M,N].

    # SOTA copy: examples/gemm/example_gemm.py @ tilelang main
    # Adapted: f32 IO (TF32 WGMMA) instead of fp16; Bias fused into the
    # epilogue; reduction tile fixed at _RED_TILE (backend pads K).
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gemm_nt(A, B, Bias, block_M, block_N, threads):
        # 128 threads (warp group) for WGMMA on large tiles; small tiles
        # (block_M < 32) cannot be evenly partitioned across 4 warps, so
        # keep the caller's 64 (mma.sync per-warp, still tensor cores).
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), "float32")
        B: T.Tensor((N, K), "float32")
        Bias: T.Tensor((N,), "float32")
        C = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, _RED_TILE), "float32")
            B_shared = T.alloc_shared((block_N, _RED_TILE), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _RED_TILE, num_stages=3):
                T.copy(A[by * block_M, k * _RED_TILE], A_shared)
                T.copy(B[bx * block_N, k * _RED_TILE], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += Bias[bx * block_N + j]
            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

    return gemm_nt


def make_gemm_nn_mma(target: str):
    """C = A @ B. A [M,K], B [K,N] -> C [M,N].

    # SOTA copy: examples/gemm/example_gemm.py @ tilelang main
    # Adapted: f32 IO (TF32 WGMMA); B is [K,N] (loaded as-is, no transpose);
    # reduction tile fixed at _RED_TILE.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gemm_nn(A, B, block_M, block_N, threads):
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), "float32")
        B: T.Tensor((K, N), "float32")
        C = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, _RED_TILE), "float32")
            B_shared = T.alloc_shared((_RED_TILE, block_N), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _RED_TILE, num_stages=3):
                T.copy(A[by * block_M, k * _RED_TILE], A_shared)
                T.copy(B[k * _RED_TILE, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

    return gemm_nn


def make_gemm_tn_mma(target: str):
    """C = A.T @ B. A [M,N], B [M,K] -> C [N,K] (C_ij = sum_m A_mi B_mj).

    # SOTA copy: examples/gemm/example_gemm.py @ tilelang main
    # Adapted: f32 IO; transpose_A=True with the reduction over M tiled at
    # _RED_TILE; output tiles are (block_N, block_K) per the naive signature.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gemm_tn(A, B, block_N, block_K, threads):
        threads = 128 if block_N >= 32 else threads
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, N), "float32")
        B: T.Tensor((M, K), "float32")
        C = T.empty((N, K), "float32")
        with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((_RED_TILE, block_N), "float32")
            B_shared = T.alloc_shared((_RED_TILE, block_K), "float32")
            C_local = T.alloc_fragment((block_N, block_K), "float32")
            T.clear(C_local)
            for m in T.Pipelined(M // _RED_TILE, num_stages=3):
                T.copy(A[m * _RED_TILE, by * block_N], A_shared)
                T.copy(B[m * _RED_TILE, bx * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_A=True)
            T.copy(C_local, C[by * block_N, bx * block_K])
        return C

    return gemm_tn


# ---------------------------------------------------------------- linear fp4 (MMA)


def make_linear_fp4_mma(target: str):
    """Fused e2m1fn dequant + matmul (sm90 MMA).

    X [M,K] f32, WQ uint8 [N,K//2] (low nibble first), Scale [N,K//16] f32.
    Y[m,n] = sum_k X[m,k] * e2m1fn(WQ[n,k//2] nibble k%2) * Scale[n,k//16].

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py
    #   @ tilelang main (simple_dequant path)
    # Adapted: f32 IO instead of bf16; tileRL's float block scale (block_max/6
    #   per 16 elems) applied as a multiply instead of the example's integer-
    #   exponent scale; e2m1fn grid (matches pack_fp4 — no zero, so the
    #   backend zero-pads Scale for K-tail tiles).
    """

    @tilelang.jit(
        target=target,
        pass_configs={
            "tl.disable_data_race_check": True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def linear_fp4(X, WQ, Scale, block_M, block_N, threads):
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        X: T.Tensor((M, K), "float32")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 16), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, _RED_TILE), "float32")
            WQ_shared = T.alloc_shared((block_N, _RED_TILE // 2), "uint8")
            W_shared = T.alloc_shared((block_N, _RED_TILE), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _RED_TILE, num_stages=2):
                T.copy(X[by * block_M, k * _RED_TILE], X_shared)
                T.copy(WQ[bx * block_N, k * _RED_TILE // 2], WQ_shared)
                # e2m1fn dequant: nibble -> f32, times the per-16 block scale.
                for i, j in T.Parallel(block_N, _RED_TILE):
                    byte = WQ_shared[i, j // 2]
                    nib = (byte >> ((j % 2) * 4)) & 15
                    sign = T.cast(1 - 2 * T.cast(nib >> 3, "int32"), "float32")
                    e = T.cast((nib >> 1) & 3, "float32")
                    m = T.cast(nib & 1, "float32")
                    w = sign * (0.5 * T.exp2(e)) * (1.0 + 0.5 * m)
                    W_shared[i, j] = w * Scale[bx * block_N + i, (k * _RED_TILE + j) // 16]
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            T.copy(C_local, Y[by * block_M, bx * block_N])
        return Y

    return linear_fp4
