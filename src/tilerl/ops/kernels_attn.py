"""Paged causal attention for sm90 — FlashAttention online-softmax schedule
ported from the tilelang examples. Registered in the sm90 cell of the
dispatch matrix (registry.py); kernels.py keeps the portable floor for
cpu/metal.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T

from .kernels_mma import _pass_configs

__all__ = [
    "make_paged_attention_mma",
]


# ---------------------------------------------------------------- paged attention (MMA)


def make_paged_attention_mma(target: str):
    """Paged causal attention, FlashAttention online-softmax schedule (sm90).

    # SOTA copy: examples/flash_attention/example_mha_fwd_bshd.py @ tilelang main
    # Adapted: paged KV pool (block-table gather replaces the dense K/V
    #   T.copy), GQA (kv head = h * Hkv // H), bf16 IO with f32 accumulate,
    #   the causal mask driven by the per-row history (SeqLens - SeqQLens)
    #   instead of a dense tril, and block_M as a schedule arg: 16 for decode
    #   (M=1, padded at the boundary) — a 64-row tile would make decode
    #   compute-bound on 63 garbage rows — 64 for prefill. The 16-row tile's
    #   replicate-4 score fragment casts to bf16 through a shared-memory
    #   round-trip (the direct fragment copy conflicts on layout).
    # The backend pads Q's S dim to a multiple of block_M and passes the true
    # per-row query lengths (SeqQLens) so decode padding rows do not shift the
    # history; their gather positions clamp to the last block and are masked
    # out of the score. Mixed batches (decode rows + a prefill chunk sharing
    # one forward) pad every row to the chunk's T — padding query positions
    # compute garbage the caller never reads, bounded by the same mask. D must
    # be a multiple of 16 (WGMMA K).
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
        # 16-row tiles (decode) with 4 warps cannot partition the PV gemm when
        # D is small (each warp needs a multiple of 16 rows and 8 columns);
        # 2 warps is the partition that always works (gemm_nt precedent).
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
