"""TileLang JIT kernels for tilerl, target-neutral (block-parallel, no warp
specifics) so every kernel compiles on CPU (``target="c"``) and GPU.

Verified CPU facts (tilelang 0.1.13, macOS arm64) baked in here: at most one
``T.alloc_shared`` per kernel (prefer fragments/global); ``T.clear`` does not
lower on shared memory for CPU (zero shared buffers with a parallel loop);
``T.gemm`` has a CPU scalar fallback (global operands + a shared accumulator
tile work; fragment accumulators fail layout inference); serial reductions
use a ``T.alloc_fragment((1,))`` accumulator (a bare Python float leaks the
loop var at ``MakePackedAPI``); eager ``@tilelang.jit`` does NOT specialize on
dtype — kernels are f32, the backend wrapper casts bf16 inputs to f32.
``# ponytail: f32 compute day-1, bf16 IO/mixed precision day-2``

Metal facts (tilelang 0.1.13, Apple Silicon, 2026-08-24): fragment-scalar
accumulations must use ``T.serial`` reduction loops — Metal does not
cross-thread-reduce a fragment scalar the way CPU's serial lowering does, so
``T.Parallel`` reductions silently compute per-thread partials. Metal's
``T.gemm`` rejects global-scope operands, so the metal dispatch cell swaps in
the naive FMA gemm schedules at the bottom of this file (same signatures,
same block semantics).
CUDA facts (tilelang 0.1.13, H20/sm90, 2026-08-24): the MMA lowering has the
same global-operand rejection ("Unsupported gemm combination, A: global,
B: global") and requires tile M/N divisible by 16, so the naive FMA schedules
below are the fallback for arches without an MMA cell (metal today); the
sm90 cell uses the WGMMA schedules in kernels_mma.py instead. The per-thread
fragment accumulator also false-positives the static data-race check (same
as Metal/CPU), so the CUDA cell disables it too. A serial ``j`` loop nested inside a parallel ``i``
loop miscompiles on Metal (output columns past the first few come back
wrong); the portable shape is a 2D ``for i, j in T.Parallel(...)`` nest with
the reduction serial inside — that is why ``linear_fp4`` is shaped like the
metal gemms.

The gated-delta chunk kernel is a target-neutral sequential-scan port of the
math in ``tilelang/examples/gdn/example_chunk_delta_h.py`` (the CUDA-scheduled
WY/chunkwise form there is the day-2 GPU upgrade path, not CPU-day-1 code).
"""

from __future__ import annotations

import tilelang
import tilelang.language as T

__all__ = [
    "make_rmsnorm_partial",
    "make_rmsnorm_apply",
    "make_rmsnorm_rstd",
    "make_rmsnorm_bwd_x",
    "make_gemm_nt",
    "make_gemm_nn",
    "make_gemm_tn",
    "make_gemm_nt_naive",
    "make_gemm_nn_naive",
    "make_gemm_tn_naive",
    "make_silu_mul",
    "make_softmax",
    "make_rope",
    "make_embedding",
    "make_linear_fp4",
    "make_paged_attention",
    "make_linear_attn_chunk",
]


def _pass_configs(target: str) -> dict[str, object]:
    if target in ("c", "llvm"):
        return {
            "tirx.disable_vectorize": True,
            "tl.disable_data_race_check": True,
        }
    if target == "metal" or target.startswith("cuda"):
        # The static race check false-positives on per-thread fragments
        # allocated inside a parallel loop (each thread owns its instance);
        # the CPU cell disables the check for the same pattern.
        return {"tl.disable_data_race_check": True}
    return {}


# ---------------------------------------------------------------- rmsnorm
#
# Split-K: the reduction dim is chunked across blocks (grid over chunks x
# rows), so decode (M=1) is not one serial block. Phase 1 writes per-chunk
# sums of squares; phase 2 reduces the few chunk sums and normalizes. The
# tilelang example idiom (examples/norm/rms_norm.py, T.reduce_sum over a
# whole-row fragment) is not portable: Metal does not cross-thread-reduce
# fragments, so the per-chunk accumulator stays a serial fragment scalar.


def make_rmsnorm_partial(target: str):
    """Phase 1: P[row, chunk] = sum_k x[row, k]^2 over the chunk's block_N."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rmsnorm_partial(X, block_N, num_chunks, threads):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), "float32")
        P = T.empty((M, num_chunks), "float32")
        with T.Kernel(T.ceildiv(N, block_N), M, threads=threads) as (bn, row):
            var = T.alloc_fragment((1,), "float32")
            var[0] = 0.0
            for k in T.serial(block_N):
                kk = bn * block_N + k
                if kk < N:
                    var[0] += X[row, kk] * X[row, kk]
            P[row, bn] = var[0]
        return P

    return rmsnorm_partial


def make_rmsnorm_apply(target: str):
    """Phase 2: rstd from the chunk sums, then y = x * rstd * w. Each block
    redundantly reduces the few chunk sums (cheap) and writes its own chunk,
    so the normalize pass is parallel even at M=1."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rmsnorm_apply(X, W, P, eps: T.float32, block_N, num_chunks, threads):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), "float32")
        W: T.Tensor((N,), "float32")
        P: T.Tensor((M, num_chunks), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), M, threads=threads) as (bn, row):
            var = T.alloc_fragment((1,), "float32")
            var[0] = 0.0
            for c in T.serial(num_chunks):
                var[0] += P[row, c]
            rstd = T.rsqrt(var[0] / N + eps)
            for k in T.Parallel(block_N):
                kk = bn * block_N + k
                if kk < N:
                    Y[row, kk] = X[row, kk] * rstd * W[kk]
        return Y

    return rmsnorm_apply


def make_rmsnorm_rstd(target: str):
    """Per-row rsqrt(mean(x^2) + eps) -> Rstd [M]. Shared by the bwd kernels."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rmsnorm_rstd(X, eps: T.float32, threads):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), "float32")
        Rstd = T.empty((M,), "float32")
        with T.Kernel(M, threads=threads) as row:
            var = T.alloc_fragment((1,), "float32")
            var[0] = 0.0
            for k in T.serial(N):
                var[0] += X[row, k] * X[row, k]
            var[0] = var[0] / N
            Rstd[row] = T.rsqrt(var[0] + eps)
        return Rstd

    return rmsnorm_rstd


def make_rmsnorm_bwd_x(target: str):
    """gx[row, k] = rstd * (grad*w - rstd^2 * x * mean(grad*w*x))."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rmsnorm_bwd_x(Grad, X, W, Rstd, threads):
        M, N = T.const("M, N")
        Grad: T.Tensor((M, N), "float32")
        X: T.Tensor((M, N), "float32")
        W: T.Tensor((N,), "float32")
        Rstd: T.Tensor((M,), "float32")
        Gx = T.empty((M, N), "float32")
        with T.Kernel(M, threads=threads) as row:
            rstd = Rstd[row]
            c = T.alloc_fragment((1,), "float32")
            c[0] = 0.0
            for k in T.serial(N):
                c[0] += Grad[row, k] * X[row, k] * W[k]
            c[0] = c[0] / N
            for k in T.Parallel(N):
                Gx[row, k] = rstd * (Grad[row, k] * W[k] - rstd * rstd * X[row, k] * c[0])
        return Gx

    return rmsnorm_bwd_x


# ---------------------------------------------------------------- gemm


def make_gemm_nt(target: str):
    """C = A @ B.T.  A [M, K], B [N, K] -> C [M, N]."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def gemm_nt(A, B, Bias, block_M, block_N, threads):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), "float32")
        B: T.Tensor((N, K), "float32")
        Bias: T.Tensor((N,), "float32")
        C = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            Cc = T.alloc_shared((block_M, block_N), "float32")
            for i in T.Parallel(block_M):
                for j in range(block_N):
                    Cc[i, j] = 0.0
            T.gemm(
                A[by * block_M : by * block_M + block_M, :],
                B[bx * block_N : bx * block_N + block_N, :],
                Cc,
                transpose_B=True,
            )
            for i in T.Parallel(block_M):
                for j in range(block_N):
                    Cc[i, j] += Bias[bx * block_N + j]
            T.copy(
                Cc, C[by * block_M : by * block_M + block_M, bx * block_N : bx * block_N + block_N]
            )
        return C

    return gemm_nt


def make_gemm_nn(target: str):
    """C = A @ B.  A [M, K], B [K, N] -> C [M, N]."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def gemm_nn(A, B, block_M, block_N, threads):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), "float32")
        B: T.Tensor((K, N), "float32")
        C = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            Cc = T.alloc_shared((block_M, block_N), "float32")
            for i in T.Parallel(block_M):
                for j in range(block_N):
                    Cc[i, j] = 0.0
            T.gemm(
                A[by * block_M : by * block_M + block_M, :],
                B[:, bx * block_N : bx * block_N + block_N],
                Cc,
            )
            T.copy(
                Cc, C[by * block_M : by * block_M + block_M, bx * block_N : bx * block_N + block_N]
            )
        return C

    return gemm_nn


def make_gemm_tn(target: str):
    """C = A.T @ B.  A [M, N], B [M, K] -> C [N, K] (C_ij = sum_m A_mi B_mj)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def gemm_tn(A, B, block_N, block_K, threads):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, N), "float32")
        B: T.Tensor((M, K), "float32")
        C = T.empty((N, K), "float32")
        with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            Cc = T.alloc_shared((block_N, block_K), "float32")
            for i in T.Parallel(block_N):
                for j in range(block_K):
                    Cc[i, j] = 0.0
            T.gemm(
                A[:, by * block_N : by * block_N + block_N],
                B[:, bx * block_K : bx * block_K + block_K],
                Cc,
                transpose_A=True,
            )
            T.copy(
                Cc, C[by * block_N : by * block_N + block_N, bx * block_K : bx * block_K + block_K]
            )
        return C

    return gemm_tn


# ---------------------------------------------------------------- gemm (naive FMA schedule)
#
# Metal's T.gemm lowering rejects global-scope operands ("Unsupported gemm
# combination, A: global, B: global"), and CUDA's MMA lowering rejects them
# too (plus the m16n8k16 MMA requires tile M/N divisible by 16). The CPU
# kernels' scalar-fallback T.gemm therefore has no Metal or CUDA path; both
# cells use these naive explicit-FMA schedules instead — same signatures and
# block semantics as the CPU kernels, no T.gemm, no shared-memory tiling.
# ponytail: naive FMA day-1, shared-memory tiled T.gemm day-2


def make_gemm_nt_naive(target: str):
    """C = A @ B.T + Bias.  A [M, K], B [N, K] -> C [M, N]."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def gemm_nt(A, B, Bias, block_M, block_N, threads):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), "float32")
        B: T.Tensor((N, K), "float32")
        Bias: T.Tensor((N,), "float32")
        C = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                acc = T.alloc_fragment((1,), "float32")
                acc[0] = 0.0
                for k in T.serial(K):
                    acc[0] += A[by * block_M + i, k] * B[bx * block_N + j, k]
                C[by * block_M + i, bx * block_N + j] = acc[0] + Bias[bx * block_N + j]
        return C

    return gemm_nt


def make_gemm_nn_naive(target: str):
    """C = A @ B.  A [M, K], B [K, N] -> C [M, N]."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def gemm_nn(A, B, block_M, block_N, threads):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), "float32")
        B: T.Tensor((K, N), "float32")
        C = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                acc = T.alloc_fragment((1,), "float32")
                acc[0] = 0.0
                for k in T.serial(K):
                    acc[0] += A[by * block_M + i, k] * B[k, bx * block_N + j]
                C[by * block_M + i, bx * block_N + j] = acc[0]
        return C

    return gemm_nn


def make_gemm_tn_naive(target: str):
    """C = A.T @ B.  A [M, N], B [M, K] -> C [N, K] (C_ij = sum_m A_mi B_mj)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def gemm_tn(A, B, block_N, block_K, threads):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, N), "float32")
        B: T.Tensor((M, K), "float32")
        C = T.empty((N, K), "float32")
        with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            for i, j in T.Parallel(block_N, block_K):
                acc = T.alloc_fragment((1,), "float32")
                acc[0] = 0.0
                for m in T.serial(M):
                    acc[0] += A[m, by * block_N + i] * B[m, bx * block_K + j]
                C[by * block_N + i, bx * block_K + j] = acc[0]
        return C

    return gemm_tn


# ---------------------------------------------------------------- silu mul


def make_silu_mul(target: str):
    """y = gate * sigmoid(gate) * up, elementwise.

    Grid over M in block_M chunks: the single-block schedule (T.Kernel(1),
    64 threads) left 8.9M elements on one block — 40% of the prefill tick on
    sm90. The tail block guards its OOB lanes.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def silu_mul(Gate, Up, block_M, threads):
        M = T.const("M")
        Gate: T.Tensor((M,), "float32")
        Up: T.Tensor((M,), "float32")
        Y = T.empty((M,), "float32")
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            for i in T.Parallel(block_M):
                idx = bx * block_M + i
                if idx < M:
                    s = T.sigmoid(Gate[idx])
                    Y[idx] = Gate[idx] * s * Up[idx]
        return Y

    return silu_mul


# ---------------------------------------------------------------- softmax


def make_softmax(target: str):
    """Row-wise softmax (axis = last dim)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def softmax(X, threads):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(M, threads=threads) as row:
            mx = T.alloc_fragment((1,), "float32")
            mx[0] = -1.0e30
            for j in T.serial(N):
                mx[0] = T.max(mx[0], X[row, j])
            sm = T.alloc_fragment((1,), "float32")
            sm[0] = 0.0
            for j in T.serial(N):
                sm[0] += T.exp(X[row, j] - mx[0])
            for j in T.Parallel(N):
                Y[row, j] = T.exp(X[row, j] - mx[0]) / sm[0]
        return Y

    return softmax


# ---------------------------------------------------------------- rope


def make_rope(target: str):
    """Rotary embedding. X [B, T, H, D], Positions [B, T] int, InvFreq [D/2].

    out[..., 2d]   = x0*cos - x1*sin
    out[..., 2d+1] = x0*sin + x1*cos
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rope(X, Positions, InvFreq, threads):
        B, S, H, D = T.const("B, S, H, D")
        X: T.Tensor((B, S, H, D), "float32")
        Positions: T.Tensor((B, S), "int32")
        InvFreq: T.Tensor((D // 2,), "float32")
        Y = T.empty((B, S, H, D), "float32")
        with T.Kernel(B * S, H, threads=threads) as (bt, h):
            b = bt // S
            t = bt % S
            pos = Positions[b, t]
            for d in T.Parallel(D // 2):
                ang = T.cast(pos, "float32") * InvFreq[d]
                c = T.cos(ang)
                s = T.sin(ang)
                x0 = X[b, t, h, 2 * d]
                x1 = X[b, t, h, 2 * d + 1]
                Y[b, t, h, 2 * d] = x0 * c - x1 * s
                Y[b, t, h, 2 * d + 1] = x0 * s + x1 * c
        return Y

    return rope


# ---------------------------------------------------------------- embedding


def make_embedding(target: str):
    """Gather: Out[i, d] = Table[Idx[i], d]. Idx int32 [M]."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def embedding(Idx, Table, threads):
        M, D = T.const("M, D")
        V = T.const("V")
        Idx: T.Tensor((M,), "int32")
        Table: T.Tensor((V, D), "float32")
        Y = T.empty((M, D), "float32")
        with T.Kernel(M, threads=threads) as i:
            for d in T.Parallel(D):
                Y[i, d] = Table[Idx[i], d]
        return Y

    return embedding


# ---------------------------------------------------------------- linear fp4


def make_linear_fp4(target: str):
    """Fused e2m1 dequant + matmul.

    X [M, K] f32, WQ uint8 [N, K//2] (low nibble first), Scale [N, K//32] f32.
    Y[m, n] = sum_k X[m, k] * e2m1(WQ[n, k//2] nibble k%2) * Scale[n, k//32].

    # ponytail: dequant-in-kernel scalar decode, native fp4 tensor cores day-2
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def linear_fp4(X, WQ, Scale, block_M, block_N, threads):
        M, N, K = T.const("M, N, K")
        X: T.Tensor((M, K), "float32")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 32), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            Cc = T.alloc_shared((block_M, block_N), "float32")
            # 2D parallel nest + serial k reduction: the same schedule shape
            # as the metal gemm kernels. (A serial j loop inside a parallel i
            # loop miscompiles on Metal — columns past the first few come back
            # wrong; on CPU T.Parallel lowers to serial either way.)
            for i, j in T.Parallel(block_M, block_N):
                acc = T.alloc_fragment((1,), "float32")
                acc[0] = 0.0
                for k0 in T.serial(K // 32):
                    for kk in range(32):
                        k = k0 * 32 + kk
                        byte = WQ[bx * block_N + j, k // 2]
                        nib = (byte >> ((k % 2) * 4)) & 15
                        sign = nib >> 3
                        e = (nib >> 1) & 3
                        m = nib & 1
                        mag = (
                            0.5 * T.exp2(T.cast(e, "float32")) * (1.0 + T.cast(m, "float32") * 0.5)
                        )
                        w = (
                            T.cast(1 - 2 * T.cast(sign, "int32"), "float32")
                            * mag
                            * Scale[bx * block_N + j, k0]
                        )
                        acc[0] += X[by * block_M + i, k] * w
                Cc[i, j] = acc[0]
            T.copy(
                Cc, Y[by * block_M : by * block_M + block_M, bx * block_N : bx * block_N + block_N]
            )
        return Y

    return linear_fp4


# ---------------------------------------------------------------- paged attention


def make_paged_attention(target: str):
    """Paged causal attention, online softmax.

    Q [B, T, H, D], K/V cache [num_blocks, Hkv, BLOCK, D], BlockTable [B, Mb]
    int32, SeqLens [B] int32 (total length after this forward; query t sees
    keys [0, seq_lens - T + t)), SeqQLens [B] int32 (valid query tokens per
    row — mixed batches pad rows to a shared T; padding positions are
    unmasked garbage the caller never reads). GQA: kv head = h * Hkv // H.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def paged_attention(
        Q, KCache, VCache, BlockTable, SeqLens, SeqQLens, scale: T.float32, block_size, threads
    ):
        B, S, H, D = T.const("B, S, H, D")
        Hkv = T.const("Hkv")
        NB = T.const("NB")
        Mb = T.const("Mb")
        Q: T.Tensor((B, S, H, D), "float32")
        KCache: T.Tensor((NB, Hkv, block_size, D), "float32")
        VCache: T.Tensor((NB, Hkv, block_size, D), "float32")
        BlockTable: T.Tensor((B, Mb), "int32")
        SeqLens: T.Tensor((B,), "int32")
        SeqQLens: T.Tensor((B,), "int32")
        Out = T.empty((B, S, H, D), "float32")
        with T.Kernel(B, H, threads=threads) as (bb, hh):
            hkv = hh * Hkv // H
            hist = SeqLens[bb] - SeqQLens[bb]
            for t in T.serial(S):
                if t < SeqQLens[bb]:
                    m = T.alloc_fragment((1,), "float32")
                    m[0] = -1.0e30
                    l = T.alloc_fragment((1,), "float32")
                    l[0] = 0.0
                    acc = T.alloc_fragment((D,), "float32")
                    for d in T.Parallel(D):
                        acc[d] = 0.0
                    upper = hist + t + 1
                    for pos in T.serial(upper):
                        blk = BlockTable[bb, pos // block_size]
                        off = pos % block_size
                        s = T.alloc_fragment((1,), "float32")
                        s[0] = 0.0
                        for d in T.serial(D):
                            s[0] += Q[bb, t, hh, d] * KCache[blk, hkv, off, d]
                        s[0] = s[0] * scale
                        m_new = T.max(m[0], s[0])
                        corr = T.exp(m[0] - m_new)
                        p = T.exp(s[0] - m_new)
                        l[0] = l[0] * corr + p
                        for d in T.Parallel(D):
                            acc[d] = acc[d] * corr + p * VCache[blk, hkv, off, d]
                        m[0] = m_new
                    for d in T.Parallel(D):
                        Out[bb, t, hh, d] = acc[d] / l[0]
        return Out

    return paged_attention


# ---------------------------------------------------------------- gated delta (linear attention)


def make_linear_attn_chunk(target: str):
    """Gated-delta chunk recurrence, target-neutral per-column serial scan.

    Q, K, V [B, C, H, D] f32; G, Beta [B, C, H] f32; State [B, H, D, D] f32
    (layout S[key, value], matching agent-infer linear_attention.rs).

    One block per (b, h, value-column j), serial over t, the column living in
    per-block fragments (no shared memory):
        S_j *= g_t
        p     = sum_i S_ij k_i
        d_j   = beta_t * (v_j - p)
        S_ij += k_i * d_j
        out_j = sum_i S_ij q_i

    The previous schedule (one block per (b,h), T.Parallel over columns with a
    shared (D,D) state) is nondeterministic on Metal: tilelang 0.1.13's Metal
    codegen races cross-loop shared-memory visibility (observed drift 2.4e-2
    across identical runs). On CPU T.Parallel lowers to serial anyway, so the
    per-column scan is the portable day-1 form; threads=1 because the scan is
    fully serial per block. The CUDA-scheduled WY/chunkwise form
    (tilelang/examples/gdn/) is the day-2 GPU upgrade path.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def linear_attn_chunk(Q, K, V, G, Beta, State, threads):
        B, C, H, D = T.const("B, C, H, D")
        Q: T.Tensor((B, C, H, D), "float32")
        K: T.Tensor((B, C, H, D), "float32")
        V: T.Tensor((B, C, H, D), "float32")
        G: T.Tensor((B, C, H), "float32")
        Beta: T.Tensor((B, C, H), "float32")
        State: T.Tensor((B, H, D, D), "float32")
        Out = T.empty((B, C, H, D), "float32")
        NewState = T.empty((B, H, D, D), "float32")
        with T.Kernel(B * H * D, threads=1) as bhd:
            b = bhd // (H * D)
            hd = bhd % (H * D)
            h = hd // D
            j = hd % D
            S = T.alloc_fragment((D,), "float32")
            for i in range(D):
                S[i] = State[b, h, i, j]
            for t in T.serial(C):
                gt = G[b, t, h]
                for i in range(D):
                    S[i] *= gt
                p = T.alloc_fragment((1,), "float32")
                p[0] = 0.0
                for i in range(D):
                    p[0] += S[i] * K[b, t, h, i]
                d = (V[b, t, h, j] - p[0]) * Beta[b, t, h]
                for i in range(D):
                    S[i] += K[b, t, h, i] * d
                o = T.alloc_fragment((1,), "float32")
                o[0] = 0.0
                for i in range(D):
                    o[0] += S[i] * Q[b, t, h, i]
                Out[b, t, h, j] = o[0]
            for i in range(D):
                NewState[b, h, i, j] = S[i]
        return Out, NewState

    return linear_attn_chunk
