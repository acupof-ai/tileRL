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


def make_rmsnorm_apply_bf16(target: str, out_dtype: str = "bfloat16"):
    """make_rmsnorm_apply narrowing its output to the consumer GEMV's IO dtype:
    bfloat16 on sm90, float16 on sm70. Producing it here removes a separate cast
    of the same bytes at the GEMV's dispatch (193 launches/token on the 27B)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rmsnorm_apply(X, W, P, eps: T.float32, block_N, num_chunks, threads):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), "float32")
        W: T.Tensor((N,), "float32")
        P: T.Tensor((M, num_chunks), "float32")
        Y = T.empty((M, N), out_dtype)
        with T.Kernel(M, T.ceildiv(N, block_N), threads=threads) as (row, bn):
            var = T.alloc_fragment((1,), "float32")
            var[0] = 0.0
            for c in T.serial(num_chunks):
                var[0] += P[row, c]
            rstd = T.rsqrt(var[0] / N + eps)
            for k in T.Parallel(block_N):
                kk = bn * block_N + k
                if kk < N:
                    Y[row, kk] = T.cast(X[row, kk] * rstd * W[kk], out_dtype)
        return Y

    return rmsnorm_apply


def make_rmsnorm_fused(target: str, out_dtype: str = "bfloat16"):
    """One-launch rmsnorm (sm90): a block per row, block-wide allreduce of the
    squared sum. A serial single-thread reduce regressed 20%
    (errors/2026-08-27-fused-rmsnorm-regression.md).

    ``out_dtype`` is bf16 where the consumer is a bf16-IO GEMM and f32 where the
    output survives to a stored value: q/k norm feeds rope and then the bf16 KV
    pool, so a bf16 output there rounds twice and costs one extra ulp
    (errors/2026-09-03-unfused-prelude-double-rounds.md)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def rmsnorm_fused(X, W, eps: T.float32, threads):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), "float32")
        W: T.Tensor((N,), "float32")
        Y = T.empty((M, N), out_dtype)
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
                    Y[row, kk] = T.cast(X[row, kk] * rstd * W[kk], out_dtype)
        return Y

    return rmsnorm_fused


def make_rmsnorm_fused_bf16(target: str):
    return make_rmsnorm_fused(target, "bfloat16")


def make_rmsnorm_fused_f32(target: str):
    return make_rmsnorm_fused(target, "float32")


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


def make_silu_mul_bf16(target: str, out_dtype: str = "bfloat16"):
    """make_silu_mul narrowing its output to the consumer GEMV's IO dtype: f32 in
    from the up/gate GEMV, bfloat16 (sm90) or float16 (sm70) out for down."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def silu_mul(Gate, Up, block_M, threads):
        M = T.const("M")
        Gate: T.Tensor((M,), "float32")
        Up: T.Tensor((M,), "float32")
        Y = T.empty((M,), out_dtype)
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            for i in T.Parallel(block_M):
                idx = bx * block_M + i
                if idx < M:
                    s = T.sigmoid(Gate[idx])
                    Y[idx] = T.cast(Gate[idx] * s * Up[idx], out_dtype)
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

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def embedding_f16(Idx, Table, threads):
        M, D = T.const("M, D")
        V = T.const("V")
        Idx: T.Tensor((M,), "int32")
        Table: T.Tensor((V, D), "float16")
        Y = T.empty((M, D), "float32")
        with T.Kernel(M, threads=threads) as i:
            for d in T.Parallel(D):
                Y[i, d] = T.cast(Table[Idx[i], d], "float32")
        return Y

    return {"float32": embedding_f32, "bfloat16": embedding_bf16, "float16": embedding_f16}[dtype]


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


# ---------------------------------------------------------------- gated delta


def make_gdn_prep(target: str):
    """Front half of a GDN layer in one launch: conv1d + SiLU over q/k/v, the
    q/k L2-norm, the log gate and beta, and the next conv window -- exactly the
    operands the chunkwise-WY kernels consume. Conv channels are laid out
    q ++ k ++ v; one block per (value head, token, row), and every block in a
    GQA group recomputes that group's q/k, which is why sm90 has its own cell."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def gdn_prep(Q, Key, Val, GIn, BIn, DtBias, ALog, ConvW, Window, threads):
        B, TT, HK, DK, NVH, DV, KER, QKVD = T.const("B, TT, HK, DK, NVH, DV, KER, QKVD")
        Q: T.Tensor((B, TT, HK, DK), "float32")
        Key: T.Tensor((B, TT, HK, DK), "float32")
        Val: T.Tensor((B, TT, NVH, DV), "float32")
        GIn: T.Tensor((B, TT, NVH), "float32")
        BIn: T.Tensor((B, TT, NVH), "float32")
        DtBias: T.Tensor((NVH,), "float32")
        ALog: T.Tensor((NVH,), "float32")
        ConvW: T.Tensor((QKVD, KER), "float32")
        Window: T.Tensor((B, KER - 1, QKVD), "float32")
        Qo = T.empty((B, TT, HK, DK), "float32")
        Ko = T.empty((B, TT, HK, DK), "float32")
        Vo = T.empty((B, TT, NVH, DV), "float32")
        Go = T.empty((B, TT, NVH), "float32")
        Bo = T.empty((B, TT, NVH), "float32")
        NewWindow = T.empty((B, KER - 1, QKVD), "float32")
        scale = T.rsqrt(T.cast(DK, "float32"))
        with T.Kernel(NVH, TT, B, threads=threads) as (vh, t, bb):
            kh = vh * HK // NVH
            qc, kc, vc = kh * DK, HK * DK + kh * DK, 2 * HK * DK + vh * DV
            qa = T.alloc_fragment((DK,), "float32")
            ka = T.alloc_fragment((DK,), "float32")
            va = T.alloc_fragment((DV,), "float32")
            nq = T.alloc_fragment((1,), "float32")
            nk = T.alloc_fragment((1,), "float32")

            for j in T.serial(DK):
                qa[j] = 0.0
                ka[j] = 0.0
            for j in T.serial(DV):
                va[j] = 0.0
            for tap in T.serial(KER):
                if t + tap < KER - 1:  # the carried window, else this segment's raw qkv
                    for j in T.serial(DK):
                        qa[j] += Window[bb, t + tap, qc + j] * ConvW[qc + j, tap]
                        ka[j] += Window[bb, t + tap, kc + j] * ConvW[kc + j, tap]
                    for j in T.serial(DV):
                        va[j] += Window[bb, t + tap, vc + j] * ConvW[vc + j, tap]
                else:
                    for j in T.serial(DK):
                        qa[j] += Q[bb, t + tap - (KER - 1), kh, j] * ConvW[qc + j, tap]
                        ka[j] += Key[bb, t + tap - (KER - 1), kh, j] * ConvW[kc + j, tap]
                    for j in T.serial(DV):
                        va[j] += Val[bb, t + tap - (KER - 1), vh, j] * ConvW[vc + j, tap]

            nq[0] = 0.0
            nk[0] = 0.0
            for j in T.serial(DK):
                qa[j] = qa[j] * T.sigmoid(qa[j])
                ka[j] = ka[j] * T.sigmoid(ka[j])
                nq[0] += qa[j] * qa[j]
                nk[0] += ka[j] * ka[j]
            nq[0] = T.rsqrt(nq[0] + 1e-12) * scale
            nk[0] = T.rsqrt(nk[0] + 1e-12)
            if vh % (NVH // HK) == 0:  # one value head per GQA group writes q/k
                for j in T.serial(DK):
                    Qo[bb, t, kh, j] = qa[j] * nq[0]
                    Ko[bb, t, kh, j] = ka[j] * nk[0]
            for j in T.serial(DV):
                Vo[bb, t, vh, j] = va[j] * T.sigmoid(va[j])
            x = GIn[bb, t, vh] + DtBias[vh]
            Go[bb, t, vh] = -T.exp(ALog[vh]) * T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
            Bo[bb, t, vh] = T.sigmoid(BIn[bb, t, vh])
            if t == 0:  # next window: the last KER-1 raw tokens of Window ++ qkv
                for tap in T.serial(KER - 1):
                    if TT + tap < KER - 1:
                        for j in T.serial(DK):
                            NewWindow[bb, tap, qc + j] = Window[bb, TT + tap, qc + j]
                            NewWindow[bb, tap, kc + j] = Window[bb, TT + tap, kc + j]
                        for j in T.serial(DV):
                            NewWindow[bb, tap, vc + j] = Window[bb, TT + tap, vc + j]
                    else:
                        for j in T.serial(DK):
                            NewWindow[bb, tap, qc + j] = Q[bb, TT + tap - (KER - 1), kh, j]
                            NewWindow[bb, tap, kc + j] = Key[bb, TT + tap - (KER - 1), kh, j]
                        for j in T.serial(DV):
                            NewWindow[bb, tap, vc + j] = Val[bb, TT + tap - (KER - 1), vh, j]
        return Qo, Ko, Vo, Go, Bo, NewWindow

    return gdn_prep


def make_gdn_post(target: str, io: str = "float32"):
    """GDN epilogue in one launch: RMSNorm over the value dim, the norm weight,
    then the SiLU(z) gate. Rows are [B*T*value heads, V] off the chunk core."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def gdn_post(Core, Z, NormW, eps: T.float32, threads):
        M, N = T.const("M, N")
        Core: T.Tensor((M, N), "float32")
        Z: T.Tensor((M, N), io)
        NormW: T.Tensor((N,), "float32")
        Y = T.empty((M, N), io)
        with T.Kernel(M, threads=threads) as row:
            var = T.alloc_fragment((1,), "float32")
            var[0] = 0.0
            for j in T.serial(N):
                var[0] += Core[row, j] * Core[row, j]
            rstd = T.rsqrt(var[0] / N + eps)
            for j in T.Parallel(N):
                g = T.cast(Z[row, j], "float32")
                Y[row, j] = T.cast(Core[row, j] * rstd * NormW[j] * g * T.sigmoid(g), io)
        return Y

    return gdn_post


# ------------------------------------------------- split-KV decode attention
#
# The kernel above grids over (B, H): at B=1 that is H blocks (24 on the 27B)
# on an 80-SM card, and its dot product is a T.serial(D) fragment reduction, so
# each block runs ONE active thread. Measured 0.76 ns per scalar FMA on a V100
# = 1.16 clocks @1.53 GHz, the single-thread serial rate — 2K context cost
# 155 of 190 ms per token while the same KV is 0.15 ms at bandwidth.
#
# So split the position loop across the grid instead: (KVSPLIT, H, B) blocks,
# each owning a contiguous slice of the history, then combine the partials in
# the log domain.
#
# The grid alone is not enough. A first cut kept the per-position dot as
# T.serial(D) and measured 948 us/call at 4K context — every thread in the
# block ran the SAME D-step chain, each step waiting on its own global load.
# The signature was cost RISING with thread count (4K ctx, split only: 32t
# 780us, 64t 950, 128t 2165, 256t 4066), which is redundancy, not work.
#
# So the block stages block_N positions into fragments and reduces with
# T.reduce_sum over a (block_N, D) product — thread-parallel, operands already
# in registers. Sharing that K/V tile across the GQA group would cut cache
# traffic 6x more, but a (gq, D) fragment fails LayoutInference
# ("CanProveEqual(abs(source->scale), 1)") even padded to a power of two, so
# the block stays per-QUERY-head.
# ponytail: per-query-head reads the same K/V gq times, revisit if a
# (gq, D) fragment layout lands upstream.


def make_paged_attention_split(target: str, KVSPLIT: int = 32, block_N: int = 16):
    """Phase 1 of split-KV attention: per-slice online-softmax partials.

    Q [B, S, H, D], K/V cache [num_blocks, Hkv, BLOCK, D], BlockTable [B, Mb],
    SeqLens [B] (total length after this forward), SeqQLens [B] (valid query
    positions per row). Query s sees keys [0, seq_lens - S + s) — the same
    causal rule as the dense kernel, so a speculative verify at S>1 is served
    here too, not just S=1 decode. Writes PO [B, S, H, KVSPLIT, D] **f16**
    (unnormalized accumulator), PM/PL [B, S, H, KVSPLIT] f32 (running max and
    sum). An empty slice emits m=-inf, l=0 so the combine weights it to zero.

    PO is f16 because it is the only allocation here that scales with S, and S is
    the whole tick's padded width (engine.py:729): at B=4 S=512 H=24 D=256
    KVSPLIT=32 it is 1.500 GiB in f32, which OOMed a 32 GB card 123 MiB short.
    f16 halves it. Safe because PO holds exp(s - m) * V with the max already
    subtracted, so values are O(1) -- measured max|PO| 9.3 against f16's 65504 --
    and PM/PL stay f32, which is where the range actually lives.

    S enters the GRID, so W verify positions run concurrently instead of the
    dense kernel's serial ``for t in T.serial(S)``: at S=4 that kernel measured
    1018 ms against 39 ms at S=1, because its cost is S*history in one thread.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def paged_attention_split(Q, KCache, VCache, BlockTable, SeqLens, SeqQLens,
                              scale: T.float32, block_size, threads):
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
        PO = T.empty((B, S, H, KVSPLIT, D), "float16")
        PM = T.empty((B, S, H, KVSPLIT), "float32")
        PL = T.empty((B, S, H, KVSPLIT), "float32")
        with T.Kernel(KVSPLIT, S * H, B, threads=threads) as (sp, th, bb):
            tt = th // H
            hh = th % H
            hkv = hh * Hkv // H
            # Causal bound for THIS query: the dense kernel's hist + t + 1.
            n = SeqLens[bb] - SeqQLens[bb] + tt + 1
            per = T.ceildiv(n, KVSPLIT)
            p0 = sp * per
            p1 = T.min(n, p0 + per)
            Qf = T.alloc_fragment((D,), "float32")
            Kf = T.alloc_fragment((block_N, D), "float32")
            Vf = T.alloc_fragment((block_N, D), "float32")
            pr = T.alloc_fragment((block_N, D), "float32")
            acc = T.alloc_fragment((D,), "float32")
            s = T.alloc_fragment((block_N,), "float32")
            m = T.alloc_fragment((1,), "float32")
            mn = T.alloc_fragment((1,), "float32")
            ssum = T.alloc_fragment((1,), "float32")
            l = T.alloc_fragment((1,), "float32")
            for d in T.Parallel(D):
                Qf[d] = Q[bb, tt, hh, d]
                acc[d] = 0.0
            m[0] = -1.0e30
            l[0] = 0.0
            # A padded query row (tt >= SeqQLens) still runs: its window is
            # bounded by n above, and the caller never reads its output.
            for k in T.serial(T.ceildiv(p1 - p0, block_N)):
                for j, d in T.Parallel(block_N, D):
                    # Clamped so an out-of-range lane loads a live address; its
                    # score is masked to -inf below, so the value never counts.
                    p = T.min(p0 + k * block_N + j, p1 - 1)
                    blk = BlockTable[bb, T.min(p // block_size, Mb - 1)]
                    Kf[j, d] = KCache[blk, hkv, p % block_size, d]
                    Vf[j, d] = VCache[blk, hkv, p % block_size, d]
                    pr[j, d] = Qf[d] * Kf[j, d]
                T.reduce_sum(pr, s, dim=1)
                for j in T.Parallel(block_N):
                    s[j] = T.if_then_else(
                        p0 + k * block_N + j < p1, s[j] * scale, -1.0e30
                    )
                mn[0] = m[0]
                T.reduce_max(s, mn, dim=0, clear=False)  # running max over tiles
                corr = T.exp(m[0] - mn[0])
                for j in T.Parallel(block_N):
                    s[j] = T.exp(s[j] - mn[0])
                T.reduce_sum(s, ssum, dim=0)
                l[0] = l[0] * corr + ssum[0]
                m[0] = mn[0]
                for j, d in T.Parallel(block_N, D):
                    pr[j, d] = s[j] * Vf[j, d]
                for d in T.Parallel(D):
                    acc[d] = acc[d] * corr
                for j in T.serial(block_N):
                    for d in T.Parallel(D):
                        acc[d] += pr[j, d]
            for d in T.Parallel(D):
                PO[bb, tt, hh, sp, d] = T.cast(acc[d], "float16")
            PM[bb, tt, hh, sp] = m[0]
            PL[bb, tt, hh, sp] = l[0]
        return PO, PM, PL

    return paged_attention_split


def make_paged_attention_split_combine(target: str, KVSPLIT: int = 32):
    """Phase 2: merge the slice partials.

    Out[b,s,h,d] = sum_j w_j PO[j,d] / sum_j w_j PL[j], w_j = exp(PM_j - max PM).
    PO arrives f16 (see the split kernel) and is widened on read; the merge itself
    stays f32, as do PM/PL.

    D is the parallel axis and KVSPLIT the serial one, with the accumulator
    hoisted out of the loop: allocating a fragment INSIDE a T.Parallel(D) body
    is the shape kernels_attn.py:267 measured at 40-66 us/call, and it cost
    60-77 us here — flat in context, so it was pure overhead on every layer of
    every token.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs(target))
    def paged_attention_split_combine(PO, PM, PL, threads):
        B, S, H, D = T.const("B, S, H, D")
        PO: T.Tensor((B, S, H, KVSPLIT, D), "float16")
        PM: T.Tensor((B, S, H, KVSPLIT), "float32")
        PL: T.Tensor((B, S, H, KVSPLIT), "float32")
        Out = T.empty((B, S, H, D), "float32")
        with T.Kernel(S * H, B, threads=threads) as (th, bb):
            tt = th // H
            hh = th % H
            m = T.alloc_fragment((1,), "float32")
            l = T.alloc_fragment((1,), "float32")
            o = T.alloc_fragment((D,), "float32")
            m[0] = -1.0e30
            for sp in T.serial(KVSPLIT):
                m[0] = T.max(m[0], PM[bb, tt, hh, sp])
            l[0] = 0.0
            for d in T.Parallel(D):
                o[d] = 0.0
            for sp in T.serial(KVSPLIT):
                w = T.exp(PM[bb, tt, hh, sp] - m[0])
                l[0] += w * PL[bb, tt, hh, sp]
                for d in T.Parallel(D):
                    o[d] += w * T.cast(PO[bb, tt, hh, sp, d], "float32")
            for d in T.Parallel(D):
                # l > 0 always: n >= 1 for every row the dispatch can produce, so
                # per = ceildiv(n, KVSPLIT) >= 1 and split 0 gets p1 = min(n, per) >= 1 --
                # it runs a tile holding key 0, which every query may attend. An all-empty
                # row would divide by exactly 0 here (m init is finite, so w = exp(0) = 1),
                # and tests/test_split_combine_denominator.py pins the arithmetic because
                # the split kernel has no CPU twin to check it end to end.
                Out[bb, tt, hh, d] = o[d] / l[0]
        return Out

    return paged_attention_split_combine
