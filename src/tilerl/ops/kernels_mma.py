"""MMA (tensor-core) TileLang kernels for sm90 — SOTA schedules ported from
the tilelang examples. Registered only in the sm90 cell of the dispatch
matrix (backend.py); kernels.py keeps the portable floor (CPU T.gemm + naive
FMA) for cpu/metal. The MMA schedules do not lower on CPU (T.gemm -> WGMMA
only on sm90), which is why they live here, not in kernels.py.

All kernels are bf16-IO on sm90 (the backend casts f32 at the boundary on
CPU/metal; eager JIT does not specialize on dtype) and lower to bf16 WGMMA
with f32 accumulation. The fp4 kernels decode the e2m1fn grid with the
lop3-style integer bit-pattern fast decode (no exp2 in the loop).
"""

from __future__ import annotations

import tilelang
import tilelang.language as T

__all__ = [
    "make_gemm_nt_mma",
    "make_gemm_nn_mma",
    "make_gemm_tn_mma",
    "make_linear_fp4_mma",
    "make_linear_fp4_gemv",
    "make_linear_bf16_gemv",
    "make_quant_fp8_e4m3",
    "make_linear_fp4_fp8_mma",
    "make_linear_fp8_mma",
    "make_linear_fp8_gemv",
    "make_write_tokens",
    "make_gdn_decode_fused",
    "make_gdn_chunk_fused",
    "make_paged_attention_mma",
]

#: Reduction-tile size (K for gemm_nt/nn, M for gemm_tn, K for linear_fp4).
#: WGMMA K on sm90 is 16 (bf16); 32 is 2 K-steps, divides every model K dim
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
    # Adapted: bf16 IO (bf16 WGMMA, f32 accumulate) instead of fp16; Bias
    # fused into the epilogue; reduction tile fixed at _RED_TILE (backend
    # pads K).
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gemm_nt(A, B, Bias, block_M, block_N, threads):
        # 128 threads (warp group) for WGMMA on large tiles; small tiles
        # (block_M < 32) cannot be evenly partitioned across 4 warps, so
        # keep the caller's 64 (mma.sync per-warp, still tensor cores).
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), "bfloat16")
        B: T.Tensor((N, K), "bfloat16")
        Bias: T.Tensor((N,), "float32")
        C = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, _RED_TILE), "bfloat16")
            B_shared = T.alloc_shared((block_N, _RED_TILE), "bfloat16")
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
    # Adapted: bf16 IO (bf16 WGMMA, f32 accumulate); B is [K,N] (loaded
    # as-is, no transpose); reduction tile fixed at _RED_TILE.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gemm_nn(A, B, block_M, block_N, threads):
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), "bfloat16")
        B: T.Tensor((K, N), "bfloat16")
        C = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, _RED_TILE), "bfloat16")
            B_shared = T.alloc_shared((_RED_TILE, block_N), "bfloat16")
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
    # Adapted: bf16 IO (bf16 WGMMA, f32 accumulate); transpose_A=True with
    # the reduction over M tiled at _RED_TILE; output tiles are
    # (block_N, block_K) per the naive signature.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gemm_tn(A, B, block_N, block_K, threads):
        threads = 128 if block_N >= 32 else threads
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, N), "bfloat16")
        B: T.Tensor((M, K), "bfloat16")
        C = T.empty((N, K), "float32")
        with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((_RED_TILE, block_N), "bfloat16")
            B_shared = T.alloc_shared((_RED_TILE, block_K), "bfloat16")
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

    X [M,K] bf16, WQ uint8 [N,K//2] (low nibble first), Scale [N,K//32] f32.
    Y[m,n] = sum_k X[m,k] * e2m1fn(WQ[n,k//2] nibble k%2) * Scale[n,k//32].

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py
    #   @ tilelang main (simple_dequant path)
    # Adapted: bf16 IO (bf16 WGMMA, f32 accumulate); tileRL's float block
    #   scale (block_max/6 per 32 elems) applied as a multiply instead of the
    #   example's integer-exponent scale; e2m1fn grid (matches pack_fp4 — no
    #   zero, so the backend zero-pads Scale for K-tail tiles).
    # Fast decode: the e2m1fn grid is a power-of-two grid, so each nibble's
    #   fp32 bit pattern is pure integer math — sign<<31 | (126+e)<<23 |
    #   m<<22 — reinterpreted as float (the lop3-style fast decode).
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
        X: T.Tensor((M, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 32), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, _RED_TILE), "bfloat16")
            WQ_shared = T.alloc_shared((block_N, _RED_TILE // 2), "uint8")
            W_shared = T.alloc_shared((block_N, _RED_TILE), "bfloat16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _RED_TILE, num_stages=2):
                T.copy(X[by * block_M, k * _RED_TILE], X_shared)
                T.copy(WQ[bx * block_N, k * _RED_TILE // 2], WQ_shared)
                # e2m1fn dequant: nibble -> fp32 bits (integer math) -> bf16,
                # times the per-32 block scale.
                for i, j in T.Parallel(block_N, _RED_TILE):
                    byte = WQ_shared[i, j // 2]
                    nib = (byte >> ((j % 2) * 4)) & 15
                    ni32 = T.cast(nib, "int32")
                    bits = (
                        ((ni32 & 8) << 28) | ((126 + ((ni32 >> 1) & 3)) << 23) | ((ni32 & 1) << 22)
                    )
                    w = T.reinterpret(bits, "float32")
                    W_shared[i, j] = T.cast(
                        w * Scale[bx * block_N + i, (k * _RED_TILE + j) // 32], "bfloat16"
                    )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            T.copy(C_local, Y[by * block_M, bx * block_N])
        return Y

    return linear_fp4


# ---------------------------------------------------------------- linear fp4 (GEMV)


def make_linear_fp4_gemv(target: str):
    """Fused e2m1fn dequant + GEMV (sm90), the decode (M=1) path of linear_fp4.

    X [1,K] bf16, WQ uint8 [N,K//2] (low nibble first), Scale [N,K//32] f32.
    Y[0,n] = sum_k X[0,k] * e2m1fn(WQ[n,k//2] nibble k%2) * Scale[n,k//32].

    Decode is memory-bound: one warp group per 4 output rows streams WQ+Scale
    once (0.75 bytes/elem), dequantizing on the fly. Each thread owns a
    K-slice of block_K = reduce_thread * micro_size_k elems (micro_size_k =
    128-bit transaction / 16-bit bf16 = 8); partials reduce across the warp.
    Roofline = (N*K*0.75 + 2K) bytes / HBM BW.

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py
    #   @ tilelang main (dequantize_gemv, split-K + tvm_thread_allreduce path)
    # Adapted: bf16 IO (micro_size_k = 128/16 = 8, f32 accumulate) instead of
    #   fp16; e2m1fn grid (matches pack_fp4 — no zero, so the backend
    #   zero-pads Scale for K-tail tiles) with tileRL's per-16 float block
    #   scale applied per micro-tile; uint8 storage; M fixed at 1 (decode) so
    #   the grid has no M dim.
    # Fast decode: the e2m1fn grid is a power-of-two grid, so each nibble's
    #   fp32 bit pattern is pure integer math — sign<<31 | (126+e)<<23 |
    #   m<<22 — reinterpreted as float (the lop3-style fast decode; the LUT/
    #   exp2 path is 2x slower, see docs/experience/wins/2026-08-24-*.md).
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp4_gemv(X, WQ, Scale, reduce_thread, n_partition):
        N, K = T.const("N, K")
        micro_size_k = 8  # 128-bit transaction / 16-bit bf16
        block_K = reduce_thread * micro_size_k
        X: T.Tensor((1, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 32), "float32")
        Y = T.empty((1, N), "bfloat16")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            X_local = T.alloc_local((micro_size_k,), "bfloat16")
            WQ_local = T.alloc_local((micro_size_k // 2,), "uint8")
            acc = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            acc[0] = 0.0
            for ko in T.serial(T.ceildiv(K, block_K)):
                base = ko * block_K + kr * micro_size_k
                for v in T.vectorized(micro_size_k):
                    X_local[v] = X[0, base + v]
                for v in T.vectorized(micro_size_k // 2):
                    WQ_local[v] = WQ[n, base // 2 + v]
                # one scale per micro-tile: 8 elems never cross a 32-block
                s = Scale[n, base // 32]
                for ki in T.serial(micro_size_k):
                    byte = WQ_local[ki // 2]
                    nib = (byte >> ((ki % 2) * 4)) & 15
                    # e2m1fn -> fp32 bits: sign<<31 | (126+e)<<23 | m<<22
                    ni32 = T.cast(nib, "int32")
                    bits = (
                        ((ni32 & 8) << 28) | ((126 + ((ni32 >> 1) & 3)) << 23) | ((ni32 & 1) << 22)
                    )
                    w = T.reinterpret(bits, "float32")
                    acc[0] += T.cast(X_local[ki], "float32") * w * s
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), acc[0], True, reduced[0], kr, dtype="handle"
                    )
                )
            if kr == 0:
                Y[0, n] = T.cast(reduced[0], "bfloat16")
        return Y

    return linear_fp4_gemv


# ---------------------------------------------------------------- linear bf16 (GEMV)


def make_linear_bf16_gemv(target: str):
    """GEMV (sm90), the decode (M=1) path of linear: X[1,K] bf16 @ W[N,K]
    bf16 -> Y[1,N] f32.

    Decode is memory-bound: one warp group per n_partition output rows streams
    W once (2 bytes/elem). Each thread owns a K-slice of block_K =
    reduce_thread * micro_size_k elems (micro_size_k = 128-bit transaction /
    16-bit bf16 = 8); partials reduce across the warp. f32 accumulate of the
    bf16 products is exactly what WGMMA does (bf16*bf16 is exact in f32).
    Roofline = (N*K*2 + 2K) bytes / HBM BW.

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py
    #   @ tilelang main (dequantize_gemv, split-K + tvm_thread_allreduce path)
    # Adapted: bf16 W streamed directly (the dequant stage disappears), f32
    #   output (matches the WGMMA path's dtype); M fixed at 1 (decode) so the
    #   grid has no M dim.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_bf16_gemv(X, W, reduce_thread, n_partition):
        N, K = T.const("N, K")
        micro_size_k = 8  # 128-bit transaction / 16-bit bf16
        block_K = reduce_thread * micro_size_k
        X: T.Tensor((1, K), "bfloat16")
        W: T.Tensor((N, K), "bfloat16")
        Y = T.empty((1, N), "float32")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            X_local = T.alloc_local((micro_size_k,), "bfloat16")
            W_local = T.alloc_local((micro_size_k,), "bfloat16")
            acc = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            acc[0] = 0.0
            for ko in T.serial(T.ceildiv(K, block_K)):
                base = ko * block_K + kr * micro_size_k
                for v in T.vectorized(micro_size_k):
                    X_local[v] = X[0, base + v]
                for v in T.vectorized(micro_size_k):
                    W_local[v] = W[n, base + v]
                for ki in T.serial(micro_size_k):
                    acc[0] += T.cast(X_local[ki], "float32") * T.cast(W_local[ki], "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), acc[0], True, reduced[0], kr, dtype="handle"
                    )
                )
            if kr == 0:
                Y[0, n] = reduced[0]
        return Y

    return linear_bf16_gemv


# ---------------------------------------------------------------- linear fp8 (GEMV)


def make_linear_fp8_gemv(target: str):
    """GEMV (sm90), the decode (M=1) path of linear_fp8: X[1,K] bf16 @ W8[N,K]
    e4m3 with per-128-block scale -> Y[1,N] f32.

    Same split-K + warp-reduce schedule as make_linear_bf16_gemv, but W is
    e4m3 (micro_size_k=16, 128-bit/8-bit) and each thread's 16-elem slice
    stays within one 128-block scale (block_K=512=4 scale blocks), so one
    WScale lookup per chunk. Roofline = (N*K*1.03 + 2K) bytes / HBM BW.

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py
    #   @ tilelang main (dequantize_gemv, split-K + tvm_thread_allreduce path)
    # Adapted: e4m3 W streamed directly (1 byte/elem vs the bf16 GEMV's 2),
    #   per-128-block f32 scale applied per chunk; M fixed at 1 (decode).
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp8_gemv(X, W8, WScale, reduce_thread, n_partition):
        N, K = T.const("N, K")
        micro_size_k = 16  # 128-bit transaction / 8-bit e4m3
        block_K = reduce_thread * micro_size_k  # 512 = 4 scale blocks of 128
        X: T.Tensor((1, K), "bfloat16")
        W8: T.Tensor((N, K), "float8_e4m3fn")
        WScale: T.Tensor((T.ceildiv(N, 128), T.ceildiv(K, 128)), "float32")
        Y = T.empty((1, N), "float32")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            X_local = T.alloc_local((micro_size_k,), "bfloat16")
            W_local = T.alloc_local((micro_size_k,), "float8_e4m3fn")
            acc = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            acc[0] = 0.0
            for ko in T.serial(T.ceildiv(K, block_K)):
                base = ko * block_K + kr * micro_size_k
                # the 16-elem slice [base, base+16) never crosses a 128-block
                s = WScale[n // 128, base // 128]
                for v in T.vectorized(micro_size_k):
                    X_local[v] = X[0, base + v]
                for v in T.vectorized(micro_size_k):
                    W_local[v] = W8[n, base + v]
                for ki in T.serial(micro_size_k):
                    acc[0] += T.cast(X_local[ki], "float32") * T.cast(W_local[ki], "float32") * s
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), acc[0], True, reduced[0], kr, dtype="handle"
                    )
                )
            if kr == 0:
                Y[0, n] = reduced[0]
        return Y

    return linear_fp8_gemv


# ---------------------------------------------------------------- quant fp8 (per-token)


def make_quant_fp8_e4m3(target: str):
    """Per-token dynamic quant: X [M,K] bf16 -> XQ [M,K] e4m3 + Scale [M] f32.

    ``Scale[m] = FP8_MAX / max(|X[m,:]|)``; ``XQ = (X * Scale).to(e4m3)``. One
    block per row; threads reduce the row absmax via shared memory, then each
    thread quantizes its strided K-slice. The activation dequant
    (``Y = XQ @ W / Scale``) lives in the linear_fp4_fp8 epilogue.

    Per-token (not per-32-block) scale: the e4m3 multiplicative quant error
    (~2% RMS) does not average down over K either way, and a per-token scale
    lets the gemm epilogue be a single per-row divide (no per-K-tile temp
    fragment, which breaks the WGMMA pipeline and costs ~2x).

    # Original: the per-token W8A8 quant from vLLM/sglang
    #   (examples/cast/example_per_token_cast_to_fp8.py is the triton form),
    #   written as a tilelang block-per-row kernel with a shared-memory max
    #   reduction (same idiom as the softmax/gdn kernels in this file). e4m3fn
    #   finite max is 448; the per-token scale maps each row's absmax there.
    """
    FP8_MAX = 448.0  # e4m3fn finite max

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def quant_fp8(X, XQ, Scale, threads):
        M, K = T.const("M, K")
        X: T.Tensor((M, K), "bfloat16")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        Scale: T.Tensor((M,), "float32")
        with T.Kernel(M, threads=threads) as bx:
            tid = T.get_thread_binding(0)
            red = T.alloc_shared((threads,), "float32")
            amax = T.alloc_fragment((1,), "float32")
            amax[0] = 0.0
            for k in T.serial(T.ceildiv(K, threads)):
                idx = tid + k * threads
                if idx < K:
                    v = T.cast(X[bx, idx], "float32")
                    amax[0] = T.max(amax[0], T.max(v, 0.0 - v))
            red[tid] = amax[0]
            T.tvm_storage_sync("shared")
            if tid == 0:
                m = T.alloc_fragment((1,), "float32")
                m[0] = 0.0
                for i in T.serial(threads):
                    m[0] = T.max(m[0], red[i])
                red[0] = FP8_MAX / T.max(m[0], 1e-12)
            T.tvm_storage_sync("shared")
            s = red[0]
            for k in T.serial(T.ceildiv(K, threads)):
                idx = tid + k * threads
                if idx < K:
                    XQ[bx, idx] = T.cast(T.cast(X[bx, idx], "float32") * s, "float8_e4m3fn")
            if tid == 0:
                Scale[bx] = s

    return quant_fp8


# ---------------------------------------------------------------- linear fp4 (fp8 MMA)


def make_linear_fp4_fp8_mma(target: str):
    """Fused e2m1fn dequant-to-e4m3 + fp8 WGMMA (sm90 prefill path).

    XQ [M,K] e4m3 (per-token quantized activation, from make_quant_fp8_e4m3),
    WQ uint8 [N,K//2] (low nibble first), WScale [N,K//32] f32 (tileRL
    pack_fp4 per-32 block scale), AScale [M] f32 (per-token activation scale).
    ``Y[m,n] = (sum_k XQ[m,k] * e2m1fn(WQ) * WScale[n,k//32]) / AScale[m]``.

    The e2m1fn grid ({±0.5,±0.75,±1,±1.5,±2,±3,±4,±6}) is an exact subset of
    e4m3, so the dequant is: nibble -> fp32 grid (integer fast decode) ->
    *WScale -> cast to e4m3. The cast is a requant (the dequant weight is not
    exactly on the e4m3 grid), carrying ~1.7% error on top of the activation
    quant's ~2% e4m3 floor — the standard W4A8 trade-off. The per-32-block
    weight scale is applied inside the K-loop (one per tile, since fp8 WGMMA
    K=32); the per-token activation scale is one divide in the epilogue — no
    temp fragment, no per-tile epilogue (the exact-grid alternative needs
    both and runs ~2x slower: the manual scale-accumulate breaks the WGMMA
    pipeline).

    # SOTA copy: examples/gemm_fp8/example_tilelang_gemm_fp8.py @ tilelang main
    # Adapted: e4m3 operands (fp8 WGMMA, f32 accumulate), block_K=32 (the
    #   fp8 WGMMA K, 1 step); the B operand is dequantized on the fly from
    #   tileRL's e2m1fn packed format — the same fast integer decode as
    #   make_linear_fp4_mma, requanted to e4m3; per-token activation dequant
    #   (1/AScale[m]) in the epilogue. e2m1fn has no zero, so padded WQ bytes
    #   (0x00 -> 0.5) are killed by the zero-padded WScale.
    """
    _BLOCK_K = 64  # fp8 WGMMA K=32, 2 steps; amortizes the e4m3 dequant cast

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
                # e2m1fn grid -> fp32 (integer fast decode) -> *WScale -> e4m3.
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


# ---------------------------------------------------------------- linear fp8 (native MMA)


def make_linear_fp8_mma(target: str):
    """Native fp8 WGMMA linear (sm90 prefill path): e4m3 weights + e4m3
    activations, per-128-block weight scale, per-token activation scale.

    XQ [M,K] e4m3 (per-token quantized, from make_quant_fp8_e4m3),
    W8 [N,K] e4m3 (checkpoint-native, no dequant),
    WScale [ceil(N/128), K//128] f32 (per-128-block weight scale),
    AScale [M] f32 (per-token activation scale).
    ``Y[m,n] = (sum_k XQ[m,k] * W8[n,k] * WScale[n//128, k//128]) / AScale[m]``.

    The per-128-block weight scale is applied to the accumulator per K-chunk
    (block_K=128 = the fp8 WGMMA K=32 x 4 steps): C_local holds the chunk's
    unscaled MMA result, C_accum += C_local * WScale — the deepgemm 2xAcc
    pattern. No K-loop dequant: the weight operand is native e4m3, so the
    loop body is copy+copy+gemm+scale with no requant cast (the fp4 prefill
    path's dequant-to-e4m3 is what held it at 16-22% of peak).

    # SOTA copy: examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py
    #   @ tilelang main (per-128-block scale, C_local_accum += C_local *
    #   (scales_a * scales_b) per chunk, T.clear per chunk)
    # Adapted: per-token activation scale (one divide in the epilogue) instead
    #   of per-block scales_a; f32 output; e4m3 IO (fp8 WGMMA, f32 accumulate);
    #   WScale in the checkpoint's native [N//128, K//128] layout (one scalar
    #   per N-tile, broadcast over the fragment).
    """
    _BLOCK_K = 128  # matches the checkpoint's 128-block scale; fp8 WGMMA K=32 x 4

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp8(XQ, W8, WScale, AScale, block_M, block_N, threads):
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        W8: T.Tensor((N, K), "float8_e4m3fn")
        WScale: T.Tensor((T.ceildiv(N, 128), K // 128), "float32")
        AScale: T.Tensor((M,), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, _BLOCK_K), "float8_e4m3fn")
            W_shared = T.alloc_shared((block_N, _BLOCK_K), "float8_e4m3fn")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            C_accum = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_accum)
            T.clear(C_local)
            for k in T.Pipelined(K // _BLOCK_K, num_stages=4):
                T.copy(XQ[by * block_M, k * _BLOCK_K], X_shared)
                T.copy(W8[bx * block_N, k * _BLOCK_K], W_shared)
                scale_b = WScale[bx * block_N // 128, k]
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    C_accum[i, j] += C_local[i, j] * scale_b
                T.clear(C_local)
            for i, j in T.Parallel(block_M, block_N):
                C_accum[i, j] = C_accum[i, j] / AScale[by * block_M + i]
            T.copy(C_accum, Y[by * block_M, bx * block_N])
        return Y

    return linear_fp8


# ---------------------------------------------------------------- write tokens (paged scatter)


def make_write_tokens(target: str):
    """Scatter K/V [B,T,Hkv,D] into the paged pool at [seq_len-T, seq_len).

    Replaces PagedKvPool.write_tokens' host loop: its per-token ``int()``
    syncs (block table / seq_len are device tensors) cost one GPU->CPU sync
    per token per layer and make the write uncapturable. With BlockTable /
    SeqLens read on device, the whole write is one launch — and a
    stream-capturable one, so the decode CUDA graph can own it.

    # SOTA copy: vLLM reshape_and_cache (paged KV write, same indexing:
    #   blk = block_table[b, pos // block_size], off = pos % block_size)
    # Adapted: bf16 IO throughout (the pool is bf16; the backend casts the
    #   f32 rope outputs at the boundary), one block per (b*t, head) with a
    #   parallel D loop — decode is B=T=1, prefill T up to 512.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def write_tokens(K, V, KPool, VPool, BlockTable, SeqLens, block_size, threads):
        B, S, H, D = T.const("B, S, H, D")
        NB = T.const("NB")
        Mb = T.const("Mb")
        K: T.Tensor((B, S, H, D), "bfloat16")
        V: T.Tensor((B, S, H, D), "bfloat16")
        KPool: T.Tensor((NB, H, block_size, D), "bfloat16")
        VPool: T.Tensor((NB, H, block_size, D), "bfloat16")
        BlockTable: T.Tensor((B, Mb), "int32")
        SeqLens: T.Tensor((B,), "int32")
        with T.Kernel(B * S, H, threads=threads) as (bt, h):
            b = bt // S
            t = bt % S
            pos = SeqLens[b] - S + t
            blk = BlockTable[b, pos // block_size]
            off = pos % block_size
            for d in T.Parallel(D):
                KPool[blk, h, off, d] = K[b, t, h, d]
                VPool[blk, h, off, d] = V[b, t, h, d]

    return write_tokens


# ---------------------------------------------------------------- gated-delta decode (fused)


def make_gdn_decode_fused(target: str):
    """Fused gated-delta decode core (sm90): conv1d + SiLU + q/k L2-norm +
    decay-first delta recurrence + gated RMSNorm + z-gate, one launch for
    T=1.

    Replaces reference.gdn_forward's Python head loop (~384 tiny kernel
    launches per layer per decode tick on the 27B slice: 48 value heads x
    ~8 einsums each). One block per (value head, batch); thread tv owns
    state column S[:, tv] (state in HBM, two serial passes over K).

    # SOTA copy: examples/gdn/qwen36_gdr_decode_fused.py @
    #   tilelang branch feat/qwen36-gdn-megakernel (commit 0fb99503, unmerged)
    # Adapted: f32 IO (tileRL convention; the branch is bf16-IO and rounds
    #   preact to bf16 before SiLU for arle parity — skipped, f32 matches
    #   reference.gdn_forward); separate NewState/NewWindow outputs (the
    #   branch mutates state and a conv_state ring in place); time-major
    #   conv window [B, K-1, qkv] (the branch is channel-major); q/k/v
    #   passed as separate [B, QD]/[B, VD] tensors (the branch takes a
    #   catted qkv — separate tensors make QD a direct constexpr).
    # Recurrence: tileRL's decay-first form (S *= g, then p = k @ S) — the
    #   branch matches, verified equation-by-equation against
    #   reference.gdn_forward.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_decode_fused(
        Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Window, State, threads
    ):
        # QD (flat q/k dim) is the constexpr, not NKH: tilelang requires each
        # constexpr used directly in a buffer shape, and NKH appears only
        # indirectly (NKH*K). Q/Key are [B, QD] (QD direct); the GQA key head
        # is kh = vh*(QD//K)//NVH. Params are Key/Val (not K/V) to avoid
        # shadowing the K/V constexpr head dims.
        B, QD, NVH, K, V, KER = T.const("B, QD, NVH, K, V, KER")
        VD = NVH * V
        QKVD = 2 * QD + VD
        scale = T.rsqrt(T.cast(K, "float32"))
        Q: T.Tensor((B, QD), "float32")
        Key: T.Tensor((B, QD), "float32")
        Val: T.Tensor((B, VD), "float32")
        Z: T.Tensor((B, VD), "float32")
        GIn: T.Tensor((B, NVH), "float32")
        BIn: T.Tensor((B, NVH), "float32")
        DtBias: T.Tensor((NVH,), "float32")
        ALog: T.Tensor((NVH,), "float32")
        NormW: T.Tensor((V,), "float32")
        ConvW: T.Tensor((QKVD, KER), "float32")
        Window: T.Tensor((B, KER - 1, QKVD), "float32")
        State: T.Tensor((B, NVH, K, V), "float32")
        Out = T.empty((B, VD), "float32")
        NewState = T.empty((B, NVH, K, V), "float32")
        NewWindow = T.empty((B, KER - 1, QKVD), "float32")
        with T.Kernel(NVH, B, threads=threads) as (vh, bb):
            tv = T.get_thread_binding(0)
            kh = vh * (QD // K) // NVH
            is_rep = (vh % (NVH // (QD // K))) == 0
            qc = kh * K + tv  # Q tensor column == Window/ConvW q column
            kc = QD + kh * K + tv  # Window/ConvW k column (K tensor column == qc)
            vc = 2 * QD + vh * V + tv  # Window/ConvW v column (V tensor column = vh*V+tv)

            q_s = T.alloc_shared((K,), "float32")
            k_s = T.alloc_shared((K,), "float32")
            v_s = T.alloc_shared((V,), "float32")
            qn = T.alloc_shared((1,), "float32")
            kn = T.alloc_shared((1,), "float32")
            exp_g_s = T.alloc_shared((1,), "float32")
            beta_s = T.alloc_shared((1,), "float32")
            out_s = T.alloc_shared((V,), "float32")
            rms_s = T.alloc_shared((1,), "float32")

            # conv1d (K taps) + SiLU on this head's q/k/v channels
            cq = T.alloc_fragment((1,), "float32")
            ck = T.alloc_fragment((1,), "float32")
            cv = T.alloc_fragment((1,), "float32")
            cq[0] = Q[bb, qc] * ConvW[qc, KER - 1]
            ck[0] = Key[bb, qc] * ConvW[kc, KER - 1]
            cv[0] = Val[bb, vh * V + tv] * ConvW[vc, KER - 1]
            for tap in T.serial(KER - 1):
                cq[0] += Window[bb, tap, qc] * ConvW[qc, tap]
                ck[0] += Window[bb, tap, kc] * ConvW[kc, tap]
                cv[0] += Window[bb, tap, vc] * ConvW[vc, tap]
            q_s[tv] = cq[0] * T.sigmoid(cq[0])
            k_s[tv] = ck[0] * T.sigmoid(ck[0])
            v_s[tv] = cv[0] * T.sigmoid(cv[0])
            T.tvm_storage_sync("shared")

            # L2-norm + g/beta (thread 0 reduces, broadcasts via shared)
            if tv == 0:
                acc_q = T.alloc_fragment((1,), "float32")
                acc_k = T.alloc_fragment((1,), "float32")
                T.clear(acc_q)
                T.clear(acc_k)
                for j in T.serial(K):
                    acc_q[0] += q_s[j] * q_s[j]
                    acc_k[0] += k_s[j] * k_s[j]
                qn[0] = T.rsqrt(acc_q[0] + 1e-12)
                kn[0] = T.rsqrt(acc_k[0] + 1e-12)
                x = GIn[bb, vh] + DtBias[vh]
                sp = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                exp_g_s[0] = T.exp(-T.exp(ALog[vh]) * sp)
                beta_s[0] = T.sigmoid(BIn[bb, vh])
            T.tvm_storage_sync("shared")

            q_s[tv] = q_s[tv] * qn[0] * scale
            k_s[tv] = k_s[tv] * kn[0]
            T.tvm_storage_sync("shared")

            # recurrence: decay + kv_mem, then rank-1 update + out
            kv_mem = T.alloc_fragment((1,), "float32")
            T.clear(kv_mem)
            for j in T.serial(K):
                sj = State[bb, vh, j, tv] * exp_g_s[0]
                NewState[bb, vh, j, tv] = sj
                kv_mem[0] += sj * k_s[j]
            delta = (v_s[tv] - kv_mem[0]) * beta_s[0]
            acc_o = T.alloc_fragment((1,), "float32")
            T.clear(acc_o)
            for j in T.serial(K):
                sj = NewState[bb, vh, j, tv] + delta * k_s[j]
                NewState[bb, vh, j, tv] = sj
                acc_o[0] += sj * q_s[j]
            out_s[tv] = acc_o[0]
            T.tvm_storage_sync("shared")

            # gated RMSNorm + z-gate
            if tv == 0:
                acc_sq = T.alloc_fragment((1,), "float32")
                T.clear(acc_sq)
                for j in T.serial(V):
                    acc_sq[0] += out_s[j] * out_s[j]
                rms_s[0] = T.rsqrt(acc_sq[0] / T.cast(V, "float32") + 1e-6)
            T.tvm_storage_sync("shared")
            gate = Z[bb, vh * V + tv]
            Out[bb, vh * V + tv] = out_s[tv] * rms_s[0] * NormW[tv] * (gate * T.sigmoid(gate))

            # new conv window: shift left, append current qkv. q/k channels
            # are shared across the GQA group — only the representative writes.
            for tap in T.serial(KER - 2):
                NewWindow[bb, tap, vc] = Window[bb, tap + 1, vc]
            NewWindow[bb, KER - 2, vc] = Val[bb, vh * V + tv]
            if is_rep:
                for tap in T.serial(KER - 2):
                    NewWindow[bb, tap, qc] = Window[bb, tap + 1, qc]
                    NewWindow[bb, tap, kc] = Window[bb, tap + 1, kc]
                NewWindow[bb, KER - 2, qc] = Q[bb, qc]
                NewWindow[bb, KER - 2, kc] = Key[bb, qc]

        return Out, NewState, NewWindow

    return gdn_decode_fused


# ---------------------------------------------------------------- gated-delta chunk prefill (fused)


def make_gdn_chunk_fused(target: str):
    """Fused gated-delta chunk prefill core (sm90): the T>1 generalization of
    make_gdn_decode_fused. One block per (value head, batch); thread tv owns
    state column S[:, tv]; a serial scan over T tokens carries the state in
    HBM (decay-first recurrence, matching reference.gdn_forward).

    Replaces reference.gdn_forward's Python head loop on prefill (~150k tiny
    kernel launches per 512-token prefill on the 27B slice: 48 value heads x
    ~8 einsums x T). Same fused ops as decode: conv1d + SiLU + q/k L2-norm +
    decay-first delta recurrence + gated RMSNorm + z-gate. The conv1d history
    (carried Window ++ qkv) is read per tap from HBM like the decode kernel —
    a per-thread sliding window cannot live in shared memory (each thread
    owns a different channel; shared would race) and fragments forbid the
    rq[i]=rq[i+1] shift (uniform-index constraint).

    # Original: T-loop generalization of make_gdn_decode_fused (itself a SOTA
    # copy of examples/gdn/qwen36_gdr_decode_fused.py @ tilelang branch
    # feat/qwen36-gdn-megakernel). The branch's prefill path is chunkwise-WY
    # (qwen36_prefill_wy.py + qwen36_prefill_scan_o.py); tileRL's decay-first
    # recurrence is serial-within-block instead — within a chunk scan
    # serially over T steps, across chunks carry the state (input State /
    # output NewState are the carry). Not fla's chunk delta rule (that
    # freezes chunk-start state — incompatible with decay-first).
    # ponytail: state columns stream from HBM/L2 (2 passes per token, like
    # decode); a shared-memory state tile (K*V*4 = 64KB) is the upgrade when
    # the state traffic shows up on the profile.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_chunk_fused(
        Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Window, State, threads
    ):
        # TT (sequence length) is the const, not T: T is the tilelang.language
        # module alias and rebinding it would break T.serial/T.Kernel below.
        B, TT, QD, NVH, K, V, KER = T.const("B, TT, QD, NVH, K, V, KER")
        VD = NVH * V
        QKVD = 2 * QD + VD
        scale = T.rsqrt(T.cast(K, "float32"))
        Q: T.Tensor((B, TT, QD), "float32")
        Key: T.Tensor((B, TT, QD), "float32")
        Val: T.Tensor((B, TT, VD), "float32")
        Z: T.Tensor((B, TT, VD), "float32")
        GIn: T.Tensor((B, TT, NVH), "float32")
        BIn: T.Tensor((B, TT, NVH), "float32")
        DtBias: T.Tensor((NVH,), "float32")
        ALog: T.Tensor((NVH,), "float32")
        NormW: T.Tensor((V,), "float32")
        ConvW: T.Tensor((QKVD, KER), "float32")
        Window: T.Tensor((B, KER - 1, QKVD), "float32")
        State: T.Tensor((B, NVH, K, V), "float32")
        Out = T.empty((B, TT, VD), "float32")
        NewState = T.empty((B, NVH, K, V), "float32")
        NewWindow = T.empty((B, KER - 1, QKVD), "float32")
        with T.Kernel(NVH, B, threads=threads) as (vh, bb):
            tv = T.get_thread_binding(0)
            kh = vh * (QD // K) // NVH
            is_rep = (vh % (NVH // (QD // K))) == 0
            qc = kh * K + tv  # Q tensor column == Window/ConvW q column
            kc = QD + kh * K + tv  # Window/ConvW k column (K tensor column == qc)
            vc = 2 * QD + vh * V + tv  # Window/ConvW v column (V tensor column = vh*V+tv)

            q_s = T.alloc_shared((K,), "float32")
            k_s = T.alloc_shared((K,), "float32")
            v_s = T.alloc_shared((V,), "float32")
            qn = T.alloc_shared((1,), "float32")
            kn = T.alloc_shared((1,), "float32")
            exp_g_s = T.alloc_shared((1,), "float32")
            beta_s = T.alloc_shared((1,), "float32")
            out_s = T.alloc_shared((V,), "float32")
            rms_s = T.alloc_shared((1,), "float32")

            # per-token fragments, hoisted out of the serial scan
            cq = T.alloc_fragment((1,), "float32")
            ck = T.alloc_fragment((1,), "float32")
            cv = T.alloc_fragment((1,), "float32")
            kv_mem = T.alloc_fragment((1,), "float32")
            delta = T.alloc_fragment((1,), "float32")
            acc_o = T.alloc_fragment((1,), "float32")
            acc_q = T.alloc_fragment((1,), "float32")
            acc_k = T.alloc_fragment((1,), "float32")
            acc_sq = T.alloc_fragment((1,), "float32")

            # seed the running state; token 0's decay pass reads it
            for j in T.serial(K):
                NewState[bb, vh, j, tv] = State[bb, vh, j, tv]

            for t in T.serial(TT):
                # conv1d (KER taps over Window ++ qkv) + SiLU on this head's
                # q/k/v channels — same per-tap global reads as the decode
                # kernel, generalized with the t offset.
                cq[0] = 0.0
                ck[0] = 0.0
                cv[0] = 0.0
                for tap in T.serial(KER):
                    if t + tap < KER - 1:
                        cq[0] += Window[bb, t + tap, qc] * ConvW[qc, tap]
                        ck[0] += Window[bb, t + tap, kc] * ConvW[kc, tap]
                        cv[0] += Window[bb, t + tap, vc] * ConvW[vc, tap]
                    else:
                        cq[0] += Q[bb, t + tap - (KER - 1), qc] * ConvW[qc, tap]
                        ck[0] += Key[bb, t + tap - (KER - 1), qc] * ConvW[kc, tap]
                        cv[0] += Val[bb, t + tap - (KER - 1), vh * V + tv] * ConvW[vc, tap]
                q_s[tv] = cq[0] * T.sigmoid(cq[0])
                k_s[tv] = ck[0] * T.sigmoid(ck[0])
                v_s[tv] = cv[0] * T.sigmoid(cv[0])
                T.tvm_storage_sync("shared")

                # L2-norm + g/beta (thread 0 reduces, broadcasts via shared)
                if tv == 0:
                    T.clear(acc_q)
                    T.clear(acc_k)
                    for j in T.serial(K):
                        acc_q[0] += q_s[j] * q_s[j]
                        acc_k[0] += k_s[j] * k_s[j]
                    qn[0] = T.rsqrt(acc_q[0] + 1e-12)
                    kn[0] = T.rsqrt(acc_k[0] + 1e-12)
                    x = GIn[bb, t, vh] + DtBias[vh]
                    sp = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                    exp_g_s[0] = T.exp(-T.exp(ALog[vh]) * sp)
                    beta_s[0] = T.sigmoid(BIn[bb, t, vh])
                T.tvm_storage_sync("shared")

                q_s[tv] = q_s[tv] * qn[0] * scale
                k_s[tv] = k_s[tv] * kn[0]
                T.tvm_storage_sync("shared")

                # recurrence: decay + kv_mem, then rank-1 update + out
                T.clear(kv_mem)
                for j in T.serial(K):
                    sj = NewState[bb, vh, j, tv] * exp_g_s[0]
                    NewState[bb, vh, j, tv] = sj
                    kv_mem[0] += sj * k_s[j]
                delta[0] = (v_s[tv] - kv_mem[0]) * beta_s[0]
                T.clear(acc_o)
                for j in T.serial(K):
                    sj = NewState[bb, vh, j, tv] + delta[0] * k_s[j]
                    NewState[bb, vh, j, tv] = sj
                    acc_o[0] += sj * q_s[j]
                out_s[tv] = acc_o[0]
                T.tvm_storage_sync("shared")

                # gated RMSNorm + z-gate
                if tv == 0:
                    T.clear(acc_sq)
                    for j in T.serial(V):
                        acc_sq[0] += out_s[j] * out_s[j]
                    rms_s[0] = T.rsqrt(acc_sq[0] / T.cast(V, "float32") + 1e-6)
                T.tvm_storage_sync("shared")
                gate = Z[bb, t, vh * V + tv]
                Out[bb, t, vh * V + tv] = (
                    out_s[tv] * rms_s[0] * NormW[tv] * (gate * T.sigmoid(gate))
                )

            # new conv window: last KER-1 raw qkv tokens of (Window ++ qkv).
            # q/k channels are shared across the GQA group — only the
            # representative writes them.
            for tap in T.serial(KER - 1):
                if TT + tap < KER - 1:
                    NewWindow[bb, tap, vc] = Window[bb, TT + tap, vc]
                else:
                    NewWindow[bb, tap, vc] = Val[bb, TT + tap - (KER - 1), vh * V + tv]
            if is_rep:
                for tap in T.serial(KER - 1):
                    if TT + tap < KER - 1:
                        NewWindow[bb, tap, qc] = Window[bb, TT + tap, qc]
                        NewWindow[bb, tap, kc] = Window[bb, TT + tap, kc]
                    else:
                        NewWindow[bb, tap, qc] = Q[bb, TT + tap - (KER - 1), qc]
                        NewWindow[bb, tap, kc] = Key[bb, TT + tap - (KER - 1), qc]

        return Out, NewState, NewWindow

    return gdn_chunk_fused


# ---------------------------------------------------------------- paged attention (MMA)


def make_paged_attention_mma(target: str):
    """Paged causal attention, FlashAttention online-softmax schedule (sm90).

    # SOTA copy: examples/flash_attention/example_mha_fwd_bshd.py @ tilelang main
    # Adapted: paged KV pool (block-table gather replaces the dense K/V
    #   T.copy), GQA (kv head = h * Hkv // H), bf16 IO with f32 accumulate,
    #   the causal mask driven by the per-batch history (SeqLens - seq_q)
    #   instead of a dense tril, and block_M as a schedule arg: 16 for decode
    #   (M=1, padded at the boundary) — a 64-row tile would make decode
    #   compute-bound on 63 garbage rows — 64 for prefill. The 16-row tile's
    #   replicate-4 score fragment casts to bf16 through a shared-memory
    #   round-trip (the direct fragment copy conflicts on layout).
    # The backend pads Q's S dim to a multiple of block_M and passes the true
    # query length (seq_q) so the decode padding rows do not shift the
    # history; their gather positions clamp to the last block and are masked
    # out of the score. D must be a multiple of 16 (WGMMA K).
    # ponytail: decode (M=1) is ~30x off the memory roofline — tilelang
    # 0.1.13 lowers the paged gather to synchronous loads (no cp_async for
    # elementwise copies), so the kernel is latency-bound. Split-KV
    # flash-decoding with pipelined per-block T.copy gathers is the upgrade
    # when decode shows up on the profile.
    """
    block_N = 64
    accum_dtype = T.float32

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def paged_attention(
        Q,
        KCache,
        VCache,
        BlockTable,
        SeqLens,
        scale: T.float32,
        block_size,
        block_M,
        seq_q,
        threads,
    ):
        B, S, H, D = T.const("B, S, H, D")
        Hkv = T.const("Hkv")
        NB = T.const("NB")
        Mb = T.const("Mb")
        Q: T.Tensor((B, S, H, D), "bfloat16")
        KCache: T.Tensor((NB, Hkv, block_size, D), "bfloat16")
        VCache: T.Tensor((NB, Hkv, block_size, D), "bfloat16")
        BlockTable: T.Tensor((B, Mb), "int32")
        SeqLens: T.Tensor((B,), "int32")
        Out = T.empty((B, S, H, D), "float32")
        log2e = 1.4426950408889634
        policy = T.GemmWarpPolicy.FullRow if block_M >= 32 else T.GemmWarpPolicy.Square
        # 16-row tiles (decode) with 4 warps cannot partition the PV gemm when
        # D is small (each warp needs a multiple of 16 rows and 8 columns);
        # 2 warps is the partition that always works (gemm_nt precedent).
        threads = 128 if block_M >= 32 else 64
        with T.Kernel(T.ceildiv(S, block_M), H, B, threads=threads) as (bx, hh, bb):
            hkv = hh * Hkv // H
            hist = SeqLens[bb] - seq_q
            Q_shared = T.alloc_shared((block_M, D), "bfloat16")
            K_shared = T.alloc_shared((block_N, D), "bfloat16")
            V_shared = T.alloc_shared((block_N, D), "bfloat16")
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_s_cast = T.alloc_fragment((block_M, block_N), "bfloat16")
            acc_o = T.alloc_fragment((block_M, D), accum_dtype)
            scores_max = T.alloc_fragment((block_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)
            logsum = T.alloc_fragment((block_M,), accum_dtype)
            # disable_tma: the decode Q tile is S-padded at the boundary, and
            # TMA barriers misbehave on padded dims (flash_decoding example).
            T.copy(
                Q[bb, bx * block_M : (bx + 1) * block_M, hh, :],
                Q_shared,
                disable_tma=True,
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            loop_range = T.ceildiv(hist + (bx + 1) * block_M, block_N)
            for k in T.Pipelined(loop_range, num_stages=1):
                for i, d in T.Parallel(block_N, D):
                    p = k * block_N + i
                    bidx = T.min(p // block_size, Mb - 1)
                    K_shared[i, d] = KCache[BlockTable[bb, bidx], hkv, p % block_size, d]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        k * block_N + j < hist + bx * block_M + i + 1,
                        0,
                        -T.infinity(accum_dtype),
                    )
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=policy)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * scale * log2e)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale * log2e - scores_max[i] * scale * log2e
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                if block_M >= 32:
                    T.copy(acc_s, acc_s_cast)
                else:
                    # 16-row tiles with 128 threads: acc_s is replicate-4,
                    # acc_s_cast replicate-1 — the direct copy conflicts.
                    # Round-trip through shared (one writer per element).
                    acc_s_sh = T.alloc_shared((block_M, block_N), "float32")
                    T.copy(acc_s, acc_s_sh)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s_cast[i, j] = acc_s_sh[i, j]
                for i, j in T.Parallel(block_M, D):
                    acc_o[i, j] *= scores_scale[i]
                for i, d in T.Parallel(block_N, D):
                    p = k * block_N + i
                    bidx = T.min(p // block_size, Mb - 1)
                    V_shared[i, d] = VCache[BlockTable[bb, bidx], hkv, p % block_size, d]
                T.gemm(acc_s_cast, V_shared, acc_o, policy=policy)
            for i, j in T.Parallel(block_M, D):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Out[bb, bx * block_M : (bx + 1) * block_M, hh, :])
        return Out

    return paged_attention
