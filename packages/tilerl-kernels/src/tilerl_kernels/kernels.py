"""Target-neutral TileLang kernels: the CPU/metal floor, f32 compute.

Portability rules baked in (tilelang 0.1.13): one T.alloc_shared per kernel;
serial reductions into a T.alloc_fragment((1,)) scalar (Metal does not
cross-thread-reduce a fragment, and a serial j loop inside a parallel i loop
miscompiles there, so reductions sit inside a 2D T.Parallel nest); T.gemm
rejects global operands on Metal/CUDA, so those cells use the naive FMA gemms.
# ponytail: f32 compute day-1, bf16 IO/mixed precision day-2
"""

from __future__ import annotations

import tilelang
import tilelang.language as T


def _pass_configs(target: str) -> dict[str, object]:
    # the static race check false-positives on per-thread fragments
    if target in ("c", "llvm"):
        return {"tirx.disable_vectorize": True, "tl.disable_data_race_check": True}
    if target == "metal" or target.startswith("cuda"):
        return {"tl.disable_data_race_check": True}
    return {}


# ---------------------------------------------------------------- rmsnorm
# split-K over the reduction dim so decode (M=1) is not one serial block


def make_rmsnorm_partial(target: str):
    """Phase 1: P[row, chunk] = sum_k x[row, k]^2 over the chunk's block_N."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rmsnorm_partial(X, block_N, num_chunks, threads):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), "float32")
        P = T.empty((M, num_chunks), "float32")
        with T.Kernel(M, T.ceildiv(N, block_N), threads=threads) as (row, bn):
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
    """Phase 2: rstd from the chunk sums, then y = x * rstd * w per chunk."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rmsnorm_apply(X, W, P, eps: T.float32, block_N, num_chunks, threads):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), "float32")
        W: T.Tensor((N,), "float32")
        P: T.Tensor((M, num_chunks), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(M, T.ceildiv(N, block_N), threads=threads) as (row, bn):
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


def make_rmsnorm_apply_bf16(target: str):
    """make_rmsnorm_apply writing bf16 (sm90: the consumer GEMVs are bf16-IO)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rmsnorm_apply(X, W, P, eps: T.float32, block_N, num_chunks, threads):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), "float32")
        W: T.Tensor((N,), "float32")
        P: T.Tensor((M, num_chunks), "float32")
        Y = T.empty((M, N), "bfloat16")
        with T.Kernel(M, T.ceildiv(N, block_N), threads=threads) as (row, bn):
            var = T.alloc_fragment((1,), "float32")
            var[0] = 0.0
            for c in T.serial(num_chunks):
                var[0] += P[row, c]
            rstd = T.rsqrt(var[0] / N + eps)
            for k in T.Parallel(block_N):
                kk = bn * block_N + k
                if kk < N:
                    Y[row, kk] = T.cast(X[row, kk] * rstd * W[kk], "bfloat16")
        return Y

    return rmsnorm_apply


def make_rmsnorm_fused_bf16(target: str):
    """One-launch rmsnorm (sm90): a block per row, block-wide allreduce of the
    squared sum, bf16 out. A serial single-thread reduce regressed 20%
    (errors/2026-08-27-fused-rmsnorm-regression.md)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rmsnorm_fused(X, W, eps: T.float32, threads):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), "float32")
        W: T.Tensor((N,), "float32")
        Y = T.empty((M, N), "bfloat16")
        with T.Kernel(M, threads=threads) as row:
            tx = T.get_thread_binding(0)
            part = T.alloc_local((1,), "float32")
            tot = T.alloc_local((1,), "float32")
            part[0] = 0.0
            for i in T.serial(T.ceildiv(N, threads)):
                kk = i * threads + tx
                if kk < N:
                    part[0] += X[row, kk] * X[row, kk]
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(T.uint32(1), part[0], True, tot[0], tx, dtype="handle")
                )
            rstd = T.rsqrt(tot[0] / N + eps)
            for i in T.serial(T.ceildiv(N, threads)):
                kk = i * threads + tx
                if kk < N:
                    Y[row, kk] = T.cast(X[row, kk] * rstd * W[kk], "bfloat16")
        return Y

    return rmsnorm_fused


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
# Metal's T.gemm rejects global operands; same signatures as the CPU gemms.
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
    """y = gate * sigmoid(gate) * up, gridded over M (one block was 40% of a prefill tick)."""

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


def make_silu_mul_bf16(target: str):
    """make_silu_mul writing bf16 (sm90: f32 in from the GEMV, bf16 out for the down GEMV)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def silu_mul(Gate, Up, block_M, threads):
        M = T.const("M")
        Gate: T.Tensor((M,), "float32")
        Up: T.Tensor((M,), "float32")
        Y = T.empty((M,), "bfloat16")
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            for i in T.Parallel(block_M):
                idx = bx * block_M + i
                if idx < M:
                    s = T.sigmoid(Gate[idx])
                    Y[idx] = T.cast(Gate[idx] * s * Up[idx], "bfloat16")
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
    """Rotary embedding. X [B, T, H, D], Positions [B, T] int, InvFreq [D/2]."""

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
            # rotate_half (Qwen/Llama): d pairs with d + D/2, not GPT-J's (2d, 2d+1)
            half = D // 2
            for d in T.Parallel(half):
                ang = T.cast(pos, "float32") * InvFreq[d]
                c = T.cos(ang)
                s = T.sin(ang)
                x0 = X[b, t, h, d]
                x1 = X[b, t, h, d + half]
                Y[b, t, h, d] = x0 * c - x1 * s
                Y[b, t, h, d + half] = x1 * c + x0 * s
        return Y

    return rope


# ---------------------------------------------------------------- embedding


def make_embedding(target: str, dtype: str = "float32"):
    """Gather: Out[i, d] = Table[Idx[i], d], f32 out; the table is read in its
    own dtype. Two bodies: a T.Tensor annotation is a string evaluated against
    module globals, so it cannot close over ``dtype``."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def embedding_f32(Idx, Table, threads):
        M, D = T.const("M, D")
        V = T.const("V")
        Idx: T.Tensor((M,), "int32")
        Table: T.Tensor((V, D), "float32")
        Y = T.empty((M, D), "float32")
        with T.Kernel(M, threads=threads) as i:
            for d in T.Parallel(D):
                Y[i, d] = Table[Idx[i], d]
        return Y

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def embedding_bf16(Idx, Table, threads):
        M, D = T.const("M, D")
        V = T.const("V")
        Idx: T.Tensor((M,), "int32")
        Table: T.Tensor((V, D), "bfloat16")
        Y = T.empty((M, D), "float32")
        with T.Kernel(M, threads=threads) as i:
            for d in T.Parallel(D):
                Y[i, d] = T.cast(Table[Idx[i], d], "float32")
        return Y

    return {"float32": embedding_f32, "bfloat16": embedding_bf16}[dtype]


# ---------------------------------------------------------------- linear fp4


def make_linear_fp4(target: str):
    """Fused e2m1 dequant + matmul.

    X [M, K] f32, WQ uint8 [N, K//2] (low nibble first), Scale [N, K//block]
    f32 (block = the checkpoint's scale block, 16 or 32).
    Y[m, n] = sum_k X[m, k] * e2m1(WQ[n, k//2] nibble k%2) * Scale[n, k//block].

    # ponytail: dequant-in-kernel scalar decode, native fp4 tensor cores day-2
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def linear_fp4(X, WQ, Scale, block_M, block_N, block, threads):
        M, N, K = T.const("M, N, K")
        X: T.Tensor((M, K), "float32")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // block), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            Cc = T.alloc_shared((block_M, block_N), "float32")
            for i, j in T.Parallel(block_M, block_N):
                acc = T.alloc_fragment((1,), "float32")
                acc[0] = 0.0
                for k0 in T.serial(K // block):
                    for kk in range(block):
                        k = k0 * block + kk
                        byte = WQ[bx * block_N + j, k // 2]
                        ni32 = T.cast((byte >> ((k % 2) * 4)) & 15, "int32")
                        e = (ni32 >> 1) & 3
                        m = ni32 & 1
                        # branch-free e2m1: -min(e,1) drops the subnormal mantissa, -min(e|m,1) zeros nibble 0
                        bits = (
                            ((ni32 & 8) << 28) | ((126 + e) << 23) | ((m << 22) & -T.min(e, 1))
                        ) & -T.min(e | m, 1)
                        w = T.reinterpret(bits, "float32") * Scale[bx * block_N + j, k0]
                        acc[0] += X[by * block_M + i, k] * w
                Cc[i, j] = acc[0]
            T.copy(
                Cc, Y[by * block_M : by * block_M + block_M, bx * block_N : bx * block_N + block_N]
            )
        return Y

    return linear_fp4


# ---------------------------------------------------------------- paged attention


def make_paged_attention(target: str):
    """Paged causal GQA attention, online softmax. SeqLens is the total length
    after this forward (query t sees keys [0, seq_len - seq_q + t]); SeqQLens
    bounds each row, padding rows are garbage the caller never reads."""

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
