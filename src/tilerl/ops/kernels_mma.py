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
    "make_write_tokens",
    "make_gdn_decode_fused",
    "make_gdn_chunk_fused",
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

    X [M,K] bf16, WQ uint8 [N,K//2] (low nibble first), Scale [N,K//16] f32.
    Y[m,n] = sum_k X[m,k] * e2m1fn(WQ[n,k//2] nibble k%2) * Scale[n,k//16].

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py
    #   @ tilelang main (simple_dequant path)
    # Adapted: bf16 IO (bf16 WGMMA, f32 accumulate); tileRL's float block
    #   scale (block_max/6 per 16 elems) applied as a multiply instead of the
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
        Scale: T.Tensor((N, K // 16), "float32")
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
                # times the per-16 block scale.
                for i, j in T.Parallel(block_N, _RED_TILE):
                    byte = WQ_shared[i, j // 2]
                    nib = (byte >> ((j % 2) * 4)) & 15
                    ni32 = T.cast(nib, "int32")
                    bits = (
                        ((ni32 & 8) << 28) | ((126 + ((ni32 >> 1) & 3)) << 23) | ((ni32 & 1) << 22)
                    )
                    w = T.reinterpret(bits, "float32")
                    W_shared[i, j] = T.cast(
                        w * Scale[bx * block_N + i, (k * _RED_TILE + j) // 16], "bfloat16"
                    )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            T.copy(C_local, Y[by * block_M, bx * block_N])
        return Y

    return linear_fp4


# ---------------------------------------------------------------- linear fp4 (GEMV)


def make_linear_fp4_gemv(target: str):
    """Fused e2m1fn dequant + GEMV (sm90), the decode (M=1) path of linear_fp4.

    X [1,K] bf16, WQ uint8 [N,K//2] (low nibble first), Scale [N,K//16] f32.
    Y[0,n] = sum_k X[0,k] * e2m1fn(WQ[n,k//2] nibble k%2) * Scale[n,k//16].

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
        Scale: T.Tensor((N, K // 16), "float32")
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
                # one scale per micro-tile: 8 elems never cross a 16-block
                s = Scale[n, base // 16]
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
