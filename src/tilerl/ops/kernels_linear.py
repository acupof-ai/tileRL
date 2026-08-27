"""Linear + quant MMA (tensor-core) TileLang kernels for sm90 — SOTA
schedules ported from the tilelang examples. Registered in the sm90 cell
of the dispatch matrix (registry.py); kernels.py keeps the portable floor
(CPU T.gemm + naive FMA) for cpu/metal. The MMA schedules do not lower on
CPU (T.gemm -> WGMMA only on sm90).

All kernels are bf16-IO on sm90 (the backend casts f32 at the boundary on
CPU/metal; eager JIT does not specialize on dtype) and lower to bf16 WGMMA
with f32 accumulation. The fp4 kernels decode the OCP e2m1 grid with the
lop3-style integer bit-pattern fast decode (no exp2 in the loop).
"""

from __future__ import annotations

import tilelang
import tilelang.language as T

from .kernels_mma import _pass_configs

__all__ = [
    "make_gemm_nt_mma",
    "make_gemm_nn_mma",
    "make_gemm_tn_mma",
    "make_linear_fp4_mma",
    "make_linear_fp4_gemv",
    "make_linear_bf16_gemv",
    "make_linear_fp8_gemv",
    "make_quant_fp8_e4m3",
    "make_linear_fp4_fp8_mma",
    "make_linear_fp8_mma",
]

#: Reduction-tile size (K for gemm_nt/nn, M for gemm_tn, K for linear_fp4).
#: WGMMA K on sm90 is 16 (bf16); 32 is 2 K-steps, divides every model K dim
#: (all are multiples of 32), and matches examples/gemm/example_gemm.py.
#: The backend pads the reduction dim to a multiple of this on CUDA.
_RED_TILE = 32

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


def _e2m1_fp32(nib):
    """OCP e2m1 nibble -> fp32 by IEEE bit-pattern synthesis: the grid is
    powers of two, so ``-min(e,1)`` drops the subnormals' mantissa ({0, 0.5})
    and ``-min(e|m,1)`` zeroes nibble 0. No exp2, no LUT load — the lop3-style
    fast decode, 2x the LUT/exp2 path (see
    docs/experience/wins/2026-08-24-fp4-gemv-bitcast-bf16.md)."""
    ni32 = T.cast(nib, "int32")
    e, m = (ni32 >> 1) & 3, ni32 & 1
    bits = (((ni32 & 8) << 28) | ((126 + e) << 23) | ((m << 22) & -T.min(e, 1))) & -T.min(e | m, 1)
    return T.reinterpret(bits, "float32")


def _dequant_fp4_macro(out_dtype, local_size, block):
    """Vectorized e2m1 dequant: a packed WQ tile -> a dequantized W tile in
    shared memory, 128-bit transactions (local_size elems out / local_size//2
    packed bytes in per chunk), one per-``block`` scale per chunk (``block``
    must be a multiple of local_size, so a chunk never crosses a scale block).

    The chunk loop is T.Parallel (not T.serial): a serial chunk loop obstructs
    the K-loop pipeliner on long K-loops (the dequant can't hide behind the
    WGMMA), costing ~20% on K=17408. T.Parallel lets the software pipeliner
    overlap dequant(k+1) with WGMMA(k).

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py
    #   @ tilelang main (fast_dequant path's per-thread vectorized macro)
    # Adapted: the e2m1 integer bitcast decode replaces the SOTA's twiddling
    #   extern (it only covers affine int4 grids, not e2m1's float grid);
    #   tileRL's float block scale is staged to shared (Scale_shared)
    #   and applied once per chunk (the chunk is aligned and never crosses a
    #   scale block); the chunk loop is T.Parallel (the SOTA's T.serial obstructs
    #   the K-loop pipeliner on long K). block_K must be a Python int (the
    #   vectorizer needs the literal divisor, like the SOTA's Block_QK).
    """
    local_compress = local_size // 2  # 2 nibbles per byte
    assert block % local_size == 0, f"scale block {block} must be a multiple of {local_size}"

    @T.macro
    def dequant(WQ_shared, Scale_shared, W_shared, block_N, block_K):
        for i in T.Parallel(block_N * block_K // local_size):
            WQ_local = T.alloc_local((local_compress,), "uint8")
            W_local = T.alloc_local((local_size,), out_dtype)
            cbase = i * local_compress
            nbase = i * local_size
            for v in T.vectorized(local_compress):
                WQ_local[v] = WQ_shared[(cbase + v) // (block_K // 2), (cbase + v) % (block_K // 2)]
            s = Scale_shared[nbase // block_K, (nbase % block_K) // block]
            for v in T.serial(local_size):
                byte = WQ_local[v // 2]
                w = _e2m1_fp32((byte >> ((v % 2) * 4)) & 15)
                W_local[v] = T.cast(w * s, out_dtype)
            for v in T.vectorized(local_size):
                W_shared[(nbase + v) // block_K, (nbase + v) % block_K] = W_local[v]

    return dequant


#: K-tile for the fp4 MMA paths: bf16 WGMMA K=16 (4 steps), fp8 WGMMA K=32
#: (2 steps). 64 amortizes the dequant over multiple WGMMA steps; the backend
#: pads K to a multiple of this on CUDA. Not the scale block: the scale block
#: (16 or 32) must divide this.
_FP4_BLOCK_K = 64

#: N-tile for the fp4 fp8 prefill path. 64 (not the caller's 128): the 128
#: tile left the small-N grids (down/out, N=5120 -> 40 N-tiles) under 1 wave,
#: so the dequant and WGMMA phases aligned across resident blocks and the
#: tensor cores idled. 64 doubles the N-tile count (2+ waves on every prefill
#: shape) for +33% geo-mean TFLOP/s (sweep 2026-08-25,
#: scripts/_sweep_fp8_prefill.py). The caller still pads N to a multiple of
#: 128, which is a multiple of 64.
_FP4_BLOCK_N = 64


def make_linear_fp4_mma(target: str):
    """Fused e2m1 dequant + matmul (sm90 MMA), bf16-IO.

    X [M,K] bf16, WQ uint8 [N,K//2] (low nibble first), Scale [N,K//block] f32.
    Y[m,n] = sum_k X[m,k] * e2m1(WQ[n,k//2] nibble k%2) * Scale[n,k//block].

    The dequant runs ahead of the WGMMA in the pipelined K-loop: each stage
    copies the X/WQ/Scale tiles to shared, vectorizes the dequant into W_shared
    (128-bit transactions, one scale per 8-elem chunk), then WGMMA reads
    W_shared. With num_stages=3 the dequant of stage k+1 issues while the
    WGMMA of stage k is in flight (the async WGMMA + double-buffered shared),
    so the dequant hides behind the MMA.

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py
    #   @ tilelang main (fast_dequant path: vectorized shared dequant before
    #   the WGMMA, pipelined)
    # Adapted: bf16 IO (bf16 WGMMA, f32 accumulate); tileRL's float per-block
    #   scale (block_max/6) staged to shared and applied per chunk
    #   instead of the example's integer-exponent scale; OCP e2m1 grid
    #   (matches pack_fp4; padded WQ bytes decode to 0.0, see _e2m1_fp32).
    """

    @tilelang.jit(
        target=target,
        pass_configs={
            "tl.disable_data_race_check": True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def linear_fp4(X, WQ, Scale, block_M, block_N, block, threads):
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        X: T.Tensor((M, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // block), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, _FP4_BLOCK_K), "bfloat16")
            WQ_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // 2), "uint8")
            W_shared = T.alloc_shared((block_N, _FP4_BLOCK_K), "bfloat16")
            Scale_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // block), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _FP4_BLOCK_K, num_stages=3):
                T.copy(X[by * block_M, k * _FP4_BLOCK_K], X_shared)
                T.copy(WQ[bx * block_N, k * _FP4_BLOCK_K // 2], WQ_shared)
                T.copy(Scale[bx * block_N, k * _FP4_BLOCK_K // block], Scale_shared)
                _dequant_fp4_macro("bfloat16", 8, block)(
                    WQ_shared, Scale_shared, W_shared, block_N, _FP4_BLOCK_K
                )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            T.copy(C_local, Y[by * block_M, bx * block_N])
        return Y

    return linear_fp4


# ---------------------------------------------------------------- linear fp4 (GEMV)


def make_linear_fp4_gemv(target: str, micro_size_k: int = 8, GROUP: int = 4):
    """Fused e2m1 dequant + GEMV (sm90), the decode (M=1) path of linear_fp4.

    X [1,K] bf16, WQ uint8 [N,K//2] (low nibble first), Scale [N,K//block] f32.
    Y[0,n] = sum_k X[0,k] * e2m1(WQ[n,k//2] nibble k%2) * Scale[n,k//block].

    Decode is memory-bound: one warp group per 4 output rows streams WQ+Scale
    once (0.75 bytes/elem), dequantizing on the fly. Each thread owns a
    K-slice of block_K = reduce_thread * micro_size_k elems; partials reduce
    across the warp. Roofline = (N*K*0.75 + 2K) bytes / HBM BW.

    The dequant is grouped 4 tiles at a time (GROUP=4): load 4 micro-tiles,
    decode all 4 (32 shuffles), then FMA all 4. Issuing every shuffle before
    its FMA hides the shuffle latency behind the FMA dependency chain — the
    flat per-tile decode (shuffle->FMA back-to-back) stalled each FMA on its
    shuffle. The grouped buffers are T.unroll(GROUP)-indexed (compile-time
    constant -> registers; a runtime %2 ping-pong spills to local memory).
    The block scale is applied once per micro-tile to the partial sum
    (``acc += s * sum(X*w)``, 1 FP op/elem), and the e2m1 grid is a
    16-entry warp-shuffle LUT (1 shuffle/elem, built once per thread via the
    integer bitcast). Sweeps (scripts/_sweep_gemv*.py): group4 = 45% roof vs
    flat 42%; 2 accumulators, shared-X/LUT, byte-LUT, 6-op bitcast, and
    register double-buffer all tested worse.

    ``micro_size_k`` x ``GROUP`` are sweep knobs — codegen/index gate in
    scripts/_sweep_gemv_micro.py, pod timings in scripts/bench_gemv_micro.py;
    the defaults are the shipped pair. micro_size_k sets the weight-stream load
    width alone (micro/2 bytes/thread: 8 -> LDG.32, 16 -> .64, 32 -> .128 —
    the bf16/fp8 siblings all issue .128), micro*GROUP sets the register
    footprint. micro > 8 is CUDA/ROCm-only — TileLang's Metal backend rejects a
    uint8 vector wider than 4 bytes, and this kernel is sm90-only anyway.
    The old "micro=16/32 tested worse" note is stale: it was measured
    on the flat pre-GROUP kernel, so it priced micro=32 only at GROUP=4, i.e. 4x
    the register arrays. When micro_size_k > block the partial sum is segmented
    one scale per block — a 32-elem micro-tile spans two block-16 scales, and a
    single scale on it is silently wrong.

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py
    #   @ tilelang main (dequantize_gemv, split-K + tvm_thread_allreduce path)
    # Adapted: bf16 IO (f32 accumulate) instead of fp16; OCP e2m1 grid
    #   (matches pack_fp4) with tileRL's float block scale on the
    #   partial sum; uint8 storage; M fixed at 1 (decode), no M grid dim.
    # Fast decode: _e2m1_fp32 builds the 16-entry warp LUT once per thread.
    #   The SOTA's lop3 intrin only covers affine int4 grids, not e2m1.
    # Constraint: reduce_thread must be <= 32 (the LUT is per-warp via
    #   tvm_warp_shuffle); the backend hardcodes 32.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp4_gemv(X, WQ, Scale, reduce_thread, n_partition, block):
        N, K = T.const("N, K")
        # scale segments per micro-tile: one when the tile fits a scale block
        assert micro_size_k % block == 0 or block % micro_size_k == 0
        nseg = max(1, micro_size_k // block)
        seg = micro_size_k // nseg
        xv = min(micro_size_k, 8)  # bf16 elems per 128-bit transaction
        block_K = reduce_thread * micro_size_k
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // GROUP
        X: T.Tensor((1, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // block), "float32")
        Y = T.empty((1, N), "bfloat16")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            Xs = T.alloc_local((GROUP, micro_size_k), "bfloat16")
            Ws = T.alloc_local((GROUP, micro_size_k // 2), "uint8")
            ws = T.alloc_local((GROUP, micro_size_k), "float32")
            acc = T.alloc_local((1,), "float32")
            partial = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            # 16-entry warp LUT: lane kr holds LUT[kr&15], built once via the
            # integer bitcast (no exp2). Each nibble is 1 shuffle.
            lut = _e2m1_fp32(kr & 15)
            acc[0] = 0.0
            for kg in T.serial(num_g):
                for g in T.unroll(GROUP):
                    base = (kg * GROUP + g) * block_K + kr * micro_size_k
                    # T.unroll, not the implicit serial split T.vectorized(>8)
                    # emits: a runtime-indexed Xs falls to local memory.
                    for c in T.unroll(micro_size_k // xv):
                        for v in T.vectorized(xv):
                            Xs[g, c * xv + v] = X[0, base + c * xv + v]
                    for v in T.vectorized(micro_size_k // 2):
                        Ws[g, v] = WQ[n, base // 2 + v]
                # decode all GROUP tiles before any FMA: the 32 shuffles
                # issue back-to-back, hiding their latency behind the FMA
                # dependency chain below.
                for g in T.unroll(GROUP):
                    for ki in T.unroll(micro_size_k):
                        byte = Ws[g, ki // 2]
                        nib = (byte >> ((ki % 2) * 4)) & 15
                        ws[g, ki] = T.tvm_warp_shuffle(
                            0xFFFFFFFF, lut, T.cast(nib, "int32"), 32, 32
                        )
                for g in T.unroll(GROUP):
                    base = (kg * GROUP + g) * block_K + kr * micro_size_k
                    for s in T.unroll(nseg):
                        partial[0] = 0.0
                        for ki in T.unroll(seg):
                            partial[0] += (
                                T.cast(Xs[g, s * seg + ki], "float32") * ws[g, s * seg + ki]
                            )
                        acc[0] += Scale[n, (base + s * seg) // block] * partial[0]
            # K-tail (num_ko % GROUP tiles), flat, reusing buffer slot 0
            for kt in T.serial(num_ko - num_g * GROUP):
                base = (num_g * GROUP + kt) * block_K + kr * micro_size_k
                for c in T.unroll(micro_size_k // xv):
                    for v in T.vectorized(xv):
                        Xs[0, c * xv + v] = X[0, base + c * xv + v]
                for v in T.vectorized(micro_size_k // 2):
                    Ws[0, v] = WQ[n, base // 2 + v]
                for s in T.unroll(nseg):
                    partial[0] = 0.0
                    for ki in T.unroll(seg):
                        byte = Ws[0, (s * seg + ki) // 2]
                        nib = (byte >> (((s * seg + ki) % 2) * 4)) & 15
                        w = T.tvm_warp_shuffle(0xFFFFFFFF, lut, T.cast(nib, "int32"), 32, 32)
                        partial[0] += T.cast(Xs[0, s * seg + ki], "float32") * w
                    acc[0] += Scale[n, (base + s * seg) // block] * partial[0]
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


def make_linear_fp8_gemv(target: str, GROUP: int = 4):
    """GEMV (sm90), the decode (M=1) path of linear_fp8: X[1,K] bf16 @ W8[N,K]
    e4m3 with per-128-block scale -> Y[1,N] f32.

    Same split-K + warp-reduce schedule as make_linear_bf16_gemv, but W is
    e4m3 (micro_size_k=16, 128-bit/8-bit) and each thread's 16-elem slice
    stays within one 128-block scale (block_K=512=4 scale blocks), so one
    WScale lookup per chunk. Roofline = (N*K*1.03 + 2K) bytes / HBM BW.

    GROUP chunks are loaded before any FMA (same as the fp4 GEMV's grouped
    dequant): the flat loop kept ONE 128-bit load in flight per thread, so the
    kernel was latency-bound at 13-26% roofline (graph profile 2026-08-27:
    68 us/call, 40% of the B=1 tick). The scale multiplies the chunk partial
    once, not every element.

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
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // GROUP
        X: T.Tensor((1, K), "bfloat16")
        W8: T.Tensor((N, K), "float8_e4m3fn")
        WScale: T.Tensor((T.ceildiv(N, 128), T.ceildiv(K, 128)), "float32")
        Y = T.empty((1, N), "float32")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            Xs = T.alloc_local((GROUP, micro_size_k), "bfloat16")
            Ws = T.alloc_local((GROUP, micro_size_k), "float8_e4m3fn")
            acc = T.alloc_local((1,), "float32")
            partial = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            acc[0] = 0.0
            for kg in T.serial(num_g):
                for g in T.unroll(GROUP):
                    base = (kg * GROUP + g) * block_K + kr * micro_size_k
                    for c in T.unroll(2):
                        for v in T.vectorized(8):
                            Xs[g, c * 8 + v] = X[0, base + c * 8 + v]
                    for v in T.vectorized(micro_size_k):
                        Ws[g, v] = W8[n, base + v]
                for g in T.unroll(GROUP):
                    base = (kg * GROUP + g) * block_K + kr * micro_size_k
                    partial[0] = 0.0
                    for ki in T.unroll(micro_size_k):
                        partial[0] += T.cast(Xs[g, ki], "float32") * T.cast(Ws[g, ki], "float32")
                    # the 16-elem slice [base, base+16) never crosses a 128-block
                    acc[0] += WScale[n // 128, base // 128] * partial[0]
            for kt in T.serial(num_ko - num_g * GROUP):  # K-tail, flat, slot 0
                base = (num_g * GROUP + kt) * block_K + kr * micro_size_k
                for c in T.unroll(2):
                    for v in T.vectorized(8):
                        Xs[0, c * 8 + v] = X[0, base + c * 8 + v]
                for v in T.vectorized(micro_size_k):
                    Ws[0, v] = W8[n, base + v]
                partial[0] = 0.0
                for ki in T.unroll(micro_size_k):
                    partial[0] += T.cast(Xs[0, ki], "float32") * T.cast(Ws[0, ki], "float32")
                acc[0] += WScale[n // 128, base // 128] * partial[0]
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


def make_linear_fp4_fp8_mma(target: str, k_split: int = 1):
    """Fused e2m1 dequant-to-e4m3 + fp8 WGMMA (sm90 prefill path).

    XQ [M,K] e4m3 (per-token quantized activation, from make_quant_fp8_e4m3),
    WQ uint8 [N,K//2] (low nibble first), WScale [N,K//block] f32 (the
    checkpoint's scale block, 16 or 32), AScale [M] f32 (per-token scale).
    ``Y[m,n] = (sum_k XQ[m,k] * e2m1(WQ) * WScale[n,k//block]) / AScale[m]``.

    Same dequant schedule as make_linear_fp4_mma (the vectorized shared-memory
    macro, SOTA: examples/dequantize_gemm/..._bf16_fp4_hopper.py), with the
    dequant target dtype e4m3 (16 elems per 128-bit transaction vs 8 for
    bf16): the K-loop copies XQ/WQ/Scale tiles to shared, vectorizes the
    dequant into W_shared (one scale per 16-elem chunk), then fp8 WGMMA reads
    W_shared — with num_stages=3 the dequant of stage k+1 issues while the
    WGMMA of stage k is in flight. The e2m1 grid ({0,±.5,±1,±1.5,±2,±3,±4,±6})
    is an exact subset of e4m3, so the dequant is: nibble -> fp32 grid
    (integer fast decode) -> *WScale -> cast to e4m3. The cast is a requant, so
    WScale must arrive renormalized (`reference.renorm_fp4_scale`) with
    ``6 * WScale`` inside e4m3's normal range: 2.3% weight error there, 50%
    when checkpoint-native magnitudes saturate. The per-token activation scale
    is one divide in the epilogue.

    k_split > 1 adds a K-split grid dim: each (bx, by, bk) block sums
    K/k_split K-tiles and f32 atomic-adds its partial into a caller-zeroed Y
    (the AScale divide distributes over the split sum). The caller must zero
    Y before launch and pad K to a multiple of _FP4_BLOCK_K * k_split.
    k_split=1 is the shipped kernel below, unchanged.

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py
    #   @ tilelang main (fast_dequant path: vectorized shared dequant before
    #   the WGMMA, pipelined)
    # Adapted: e4m3 operands (fp8 WGMMA, f32 accumulate), the dequant target
    #   dtype is e4m3 (requant) instead of bf16; per-token activation dequant
    #   (1/AScale[m]) in the epilogue. Padded WQ bytes (0x00) decode to 0.0.
    # SOTA copy (k_split > 1): examples/gemm_streamk/example_tilelang_gemm_streamk.py
    #   @ tilelang main (split-K grid + atomic-add reduction family)
    # Adapted: fixed 2-way equal K-split (not stream-K scheduling) on the
    #   dequant+WGMMA body above; f32 atomic add into a zeroed output.
    """

    @tilelang.jit(
        target=target,
        pass_configs={
            "tl.disable_data_race_check": True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def linear_fp4_fp8(XQ, WQ, WScale, AScale, block_M, block_N, block, threads):
        threads = 128 if block_M >= 32 else threads
        block_N = _FP4_BLOCK_N  # 64-tile: doubles the N-grid vs the caller's
        # 128, putting every prefill shape at 2+ waves (the dequant/WGMMA
        # phases no longer align across resident blocks). See _FP4_BLOCK_N.
        M, N, K = T.const("M, N, K")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        WQ: T.Tensor((N, K // 2), "uint8")
        WScale: T.Tensor((N, K // block), "float32")
        AScale: T.Tensor((M,), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, _FP4_BLOCK_K), "float8_e4m3fn")
            WQ_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // 2), "uint8")
            W_shared = T.alloc_shared((block_N, _FP4_BLOCK_K), "float8_e4m3fn")
            Scale_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // block), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _FP4_BLOCK_K, num_stages=3):
                T.copy(XQ[by * block_M, k * _FP4_BLOCK_K], X_shared)
                T.copy(WQ[bx * block_N, k * _FP4_BLOCK_K // 2], WQ_shared)
                T.copy(WScale[bx * block_N, k * _FP4_BLOCK_K // block], Scale_shared)
                _dequant_fp4_macro("float8_e4m3fn", 16, block)(
                    WQ_shared, Scale_shared, W_shared, block_N, _FP4_BLOCK_K
                )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] / AScale[by * block_M + i]
            T.copy(C_local, Y[by * block_M, bx * block_N])
        return Y

    if k_split == 1:
        return linear_fp4_fp8

    @tilelang.jit(
        target=target,
        pass_configs={
            "tl.disable_data_race_check": True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def linear_fp4_fp8_split(XQ, WQ, WScale, AScale, Y, block_M, block_N, block, threads):
        threads = 128 if block_M >= 32 else threads
        block_N = _FP4_BLOCK_N
        M, N, K = T.const("M, N, K")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        WQ: T.Tensor((N, K // 2), "uint8")
        WScale: T.Tensor((N, K // block), "float32")
        AScale: T.Tensor((M,), "float32")
        Y: T.Tensor((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), k_split, threads=threads) as (
            bx,
            by,
            bk,
        ):
            X_shared = T.alloc_shared((block_M, _FP4_BLOCK_K), "float8_e4m3fn")
            WQ_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // 2), "uint8")
            W_shared = T.alloc_shared((block_N, _FP4_BLOCK_K), "float8_e4m3fn")
            Scale_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // block), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            k0 = bk * (K // k_split // _FP4_BLOCK_K)
            k1 = (bk + 1) * (K // k_split // _FP4_BLOCK_K)
            for k in T.Pipelined(k1 - k0, num_stages=3):
                kk = k0 + k
                T.copy(XQ[by * block_M, kk * _FP4_BLOCK_K], X_shared)
                T.copy(WQ[bx * block_N, kk * _FP4_BLOCK_K // 2], WQ_shared)
                T.copy(WScale[bx * block_N, kk * _FP4_BLOCK_K // block], Scale_shared)
                _dequant_fp4_macro("float8_e4m3fn", 16, block)(
                    WQ_shared, Scale_shared, W_shared, block_N, _FP4_BLOCK_K
                )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] / AScale[by * block_M + i]
            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(Y[by * block_M + i, bx * block_N + j], C_local[i, j])

    return linear_fp4_fp8_split


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
