"""Paged causal attention for sm90: FlashAttention online softmax, ported from
the tilelang flash_attention / flash_decoding examples."""

from __future__ import annotations

import tilelang
import tilelang.language as T

from .kernels_mma import _pass_configs


def make_paged_attention_mma(target: str):
    """Paged causal GQA attention (example_mha_fwd_bshd.py + block-table gather).
    block_M 16 for decode, 64 for prefill; the backend pads S to block_M and
    SeqQLens carries the true per-row query length so padding rows are masked.
    D must be a multiple of 16 (WGMMA K).
    # ponytail: the paged gather lowers to synchronous loads (latency-bound at
    # M=1); pipelined per-block T.copy gathers when decode shows on the profile.
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
        SeqQLens,
        scale: T.float32,
        block_size,
        block_M,
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
        SeqQLens: T.Tensor((B,), "int32")
        Out = T.empty((B, S, H, D), "float32")
        log2e = 1.4426950408889634
        policy = T.GemmWarpPolicy.FullRow if block_M >= 32 else T.GemmWarpPolicy.Square
        # 16-row tiles cannot partition the PV gemm across 4 warps at small D
        threads = 128 if block_M >= 32 else 64
        with T.Kernel(T.ceildiv(S, block_M), H, B, threads=threads) as (bx, hh, bb):
            hkv = hh * Hkv // H
            hist = SeqLens[bb] - SeqQLens[bb]
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
            # TMA barriers misbehave on the S-padded decode Q tile
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
                    # replicate-4 -> replicate-1 fragment copy conflicts; go via shared
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


def make_paged_attention_decode(target: str, KVSPLIT: int = 16):
    """Decode (S=1) split-KV flash-decoding (example_gqa_decode.py + paged
    gather). Grid (KVSPLIT, Hkv, B): the GQA group is the 16-row M tile so a KV
    slice is read once per group; partials (PO, PM, PL) go to the combine
    kernel. Empty slices emit m=-inf, l=0."""
    block_N = 64
    block_M = 16
    accum_dtype = T.float32

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def paged_attention_decode(Q, KCache, VCache, BlockTable, SeqLens, PO, PM, PL, scale: T.float32, block_size):
        B, H, D = T.const("B, H, D")
        Hkv = T.const("Hkv")
        NB = T.const("NB")
        Mb = T.const("Mb")
        Q: T.Tensor((B, H, D), "bfloat16")
        KCache: T.Tensor((NB, Hkv, block_size, D), "bfloat16")
        VCache: T.Tensor((NB, Hkv, block_size, D), "bfloat16")
        BlockTable: T.Tensor((B, Mb), "int32")
        SeqLens: T.Tensor((B,), "int32")
        PO: T.Tensor((B, Hkv, KVSPLIT, block_M, D), "float32")
        PM: T.Tensor((B, Hkv, KVSPLIT, block_M), "float32")
        PL: T.Tensor((B, Hkv, KVSPLIT, block_M), "float32")
        G = H // Hkv
        log2e = 1.4426950408889634
        with T.Kernel(KVSPLIT, Hkv, B, threads=64) as (sp, hkv, bb):
            n = SeqLens[bb]
            tiles = T.ceildiv(n, block_N)
            per = T.ceildiv(tiles, KVSPLIT)
            t0 = sp * per
            t1 = T.min(tiles, t0 + per)
            Q_shared = T.alloc_shared((block_M, D), "bfloat16")
            K_shared = T.alloc_shared((block_N, D), "bfloat16")
            V_shared = T.alloc_shared((block_N, D), "bfloat16")
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_s_sh = T.alloc_shared((block_M, block_N), "float32")
            acc_s_cast = T.alloc_fragment((block_M, block_N), "bfloat16")
            acc_o = T.alloc_fragment((block_M, D), accum_dtype)
            scores_max = T.alloc_fragment((block_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)
            logsum = T.alloc_fragment((block_M,), accum_dtype)
            for i, d in T.Parallel(block_M, D):
                Q_shared[i, d] = T.if_then_else(i < G, Q[bb, hkv * G + T.min(i, G - 1), d], T.cast(0, "bfloat16"))
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            for k in T.Pipelined(t1 - t0, num_stages=1):
                for i, d in T.Parallel(block_N, D):
                    p = (t0 + k) * block_N + i
                    bidx = T.min(p // block_size, Mb - 1)
                    K_shared[i, d] = KCache[BlockTable[bb, bidx], hkv, p % block_size, d]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else((t0 + k) * block_N + j < n, 0, -T.infinity(accum_dtype))
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.Square)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * scale * log2e)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale * log2e - scores_max[i] * scale * log2e)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_sh)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s_cast[i, j] = acc_s_sh[i, j]
                for i, j in T.Parallel(block_M, D):
                    acc_o[i, j] *= scores_scale[i]
                for i, d in T.Parallel(block_N, D):
                    p = (t0 + k) * block_N + i
                    bidx = T.min(p // block_size, Mb - 1)
                    V_shared[i, d] = VCache[BlockTable[bb, bidx], hkv, p % block_size, d]
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.Square)
            # partials in the scaled-log2 domain: PM = max * scale*log2e, PL = sum
            T.copy(acc_o, PO[bb, hkv, sp, :, :])
            for i in T.Parallel(block_M):
                PM[bb, hkv, sp, i] = scores_max[i] * scale * log2e
                PL[bb, hkv, sp, i] = logsum[i]

    return paged_attention_decode


def make_paged_attention_combine(target: str, KVSPLIT: int = 16):
    """Merge split-KV partials: Out[b,h,d] = sum_s w_s PO_s / sum_s w_s PL_s,
    w_s = 2^(PM_s - max_s PM_s). Empty slices carry PM=-inf, PL=0."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def paged_attention_combine(PO, PM, PL, G):
        B, Hkv, D = T.const("B, Hkv, D")
        PO: T.Tensor((B, Hkv, KVSPLIT, 16, D), "float32")
        PM: T.Tensor((B, Hkv, KVSPLIT, 16), "float32")
        PL: T.Tensor((B, Hkv, KVSPLIT, 16), "float32")
        Out = T.empty((B, Hkv, G, D), "bfloat16")
        # one warp per row with scalar locals: a T.Parallel(D) body ran at 40-66 us/call
        with T.Kernel(B * Hkv * G, threads=32) as row:
            lane = T.get_thread_binding(0)
            bb = row // (Hkv * G)
            hkv = (row // G) % Hkv
            g = row % G
            w = T.alloc_local((KVSPLIT,), "float32")
            m = T.alloc_local((1,), "float32")
            l = T.alloc_local((1,), "float32")
            acc = T.alloc_local((1,), "float32")
            m[0] = -T.infinity("float32")
            for sp in T.unroll(KVSPLIT):
                m[0] = T.max(m[0], PM[bb, hkv, sp, g])
            l[0] = 0.0
            for sp in T.unroll(KVSPLIT):
                w[sp] = T.exp2(PM[bb, hkv, sp, g] - m[0])
                l[0] += w[sp] * PL[bb, hkv, sp, g]
            for i in T.unroll(D // 32):
                acc[0] = 0.0
                for sp in T.unroll(KVSPLIT):
                    acc[0] += w[sp] * PO[bb, hkv, sp, g, i * 32 + lane]
                Out[bb, hkv, g, i * 32 + lane] = T.cast(acc[0] / l[0], "bfloat16")
        return Out

    return paged_attention_combine
