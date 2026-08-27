"""Shared pass config + write_tokens for the sm90 cell. The MMA kernel
families live in kernels_linear.py / kernels_gdn.py / kernels_attn.py;
kernels.py keeps the portable floor (CPU T.gemm + naive FMA) for cpu/metal.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T

__all__ = [
    "make_write_tokens",
]


def _pass_configs() -> dict[str, object]:
    # The static race check false-positives on per-thread fragments (same as
    # the cpu/metal cells in kernels.py).
    return {"tl.disable_data_race_check": True}


# ---------------------------------------------------------------- write tokens (paged scatter)


def make_write_tokens(target: str):
    """Scatter K/V [B,T,Hkv,D] into the paged pool at [seq_len-T, seq_len).

    Replaces PagedKvPool.write_tokens' host loop: its per-token ``int()``
    syncs (block table / seq_len are device tensors) cost one GPU->CPU sync
    per token per layer and make the write uncapturable. With BlockTable /
    SeqLens read on device, the whole write is one launch — and a
    stream-capturable one, so the decode CUDA graph can own it.

    Rows are left-aligned valid tokens: SeqQLens [B] is the per-row valid
    count (decode row: 1, prefill row: the chunk length); the write covers
    [seq_len-seq_q, seq_len) and padding positions are skipped. Mixed
    batches pad every row to the shared S this way.

    # SOTA copy: vLLM reshape_and_cache (paged KV write, same indexing:
    #   blk = block_table[b, pos // block_size], off = pos % block_size)
    # Adapted: bf16 IO throughout (the pool is bf16; the backend casts the
    #   f32 rope outputs at the boundary), one block per (b*t, head) with a
    #   parallel D loop — decode is B=T=1, prefill T up to 512.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def write_tokens(K, V, KPool, VPool, BlockTable, SeqLens, SeqQLens, block_size, threads):
        B, S, H, D = T.const("B, S, H, D")
        NB = T.const("NB")
        Mb = T.const("Mb")
        K: T.Tensor((B, S, H, D), "bfloat16")
        V: T.Tensor((B, S, H, D), "bfloat16")
        KPool: T.Tensor((NB, H, block_size, D), "bfloat16")
        VPool: T.Tensor((NB, H, block_size, D), "bfloat16")
        BlockTable: T.Tensor((B, Mb), "int32")
        SeqLens: T.Tensor((B,), "int32")
        SeqQLens: T.Tensor((B,), "int32")
        with T.Kernel(B * S, H, threads=threads) as (bt, h):
            b = bt // S
            t = bt % S
            if t < SeqQLens[b]:
                pos = SeqLens[b] - SeqQLens[b] + t
                blk = BlockTable[b, pos // block_size]
                off = pos % block_size
                for d in T.Parallel(D):
                    KPool[blk, h, off, d] = K[b, t, h, d]
                    VPool[blk, h, off, d] = V[b, t, h, d]

    return write_tokens


def make_attn_prep(target: str):
    """Decode/prefill attention prep in ONE launch: per-head q_norm/k_norm,
    partial RoPE, and the paged K/V write, straight off the fused-qkv GEMV
    output — replaces 2 rmsnorm (2 launches each), 2 rope (+slice/cat copies),
    and write_tokens: ~11 launches -> 1 per full-attn layer.

    QKV [B, S, NQKV] bf16 rows are [hq x (query D ; gate D)] ++ [hkv x D] ++
    [hkv x D] (gate stays in QKV for the caller). Block (b*s, h) normalizes
    and rotates q head h -> Qn; blocks h < hkv also do k head h and the K/V
    pool write. Reduction over D is serial per block (D=256: ~1us, far under
    the launch cost — the split-K rmsnorm exists for N=5120, not this).

    # SOTA copy: agent-infer decode_prep (norm+rope+kv-write fused).
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def attn_prep(
        QKV, Wq, Wk, Positions, InvFreq, KPool, VPool, BlockTable, SeqLens, SeqQLens,
        eps: T.float32, hq, hkv, block_size, threads,
    ):
        B, S, NQKV = T.const("B, S, NQKV")
        D = T.const("D")
        RD2 = T.const("RD2")
        NB = T.const("NB")
        Mb = T.const("Mb")
        QKV: T.Tensor((B, S, NQKV), "bfloat16")
        Wq: T.Tensor((D,), "float32")
        Wk: T.Tensor((D,), "float32")
        Positions: T.Tensor((B, S), "int32")
        InvFreq: T.Tensor((RD2,), "float32")
        KPool: T.Tensor((NB, hkv, block_size, D), "bfloat16")
        VPool: T.Tensor((NB, hkv, block_size, D), "bfloat16")
        BlockTable: T.Tensor((B, Mb), "int32")
        SeqLens: T.Tensor((B,), "int32")
        SeqQLens: T.Tensor((B,), "int32")
        Qn = T.empty((B, S, hq, D), "bfloat16")
        q_rows = hq * 2 * D
        with T.Kernel(B * S, hq, threads=threads) as (bt, h):
            b = bt // S
            t = bt % S
            pos = Positions[b, t]
            posf = T.cast(pos, "float32")
            q0 = h * 2 * D
            var = T.alloc_fragment((1,), "float32")
            var[0] = 0.0
            for k in T.serial(D):
                xv = T.cast(QKV[b, t, q0 + k], "float32")
                var[0] += xv * xv
            rstd = T.rsqrt(var[0] / D + eps)
            for d in T.Parallel(D):
                Qn[b, t, h, d] = T.cast(T.cast(QKV[b, t, q0 + d], "float32") * rstd * Wq[d], "bfloat16")
            # rotate_half over the leading RD = 2*RD2 dims; rest pass through.
            for d in T.Parallel(RD2):
                ang = posf * InvFreq[d]
                c = T.cos(ang)
                s = T.sin(ang)
                x0 = T.cast(QKV[b, t, q0 + d], "float32") * rstd * Wq[d]
                x1 = T.cast(QKV[b, t, q0 + d + RD2], "float32") * rstd * Wq[d + RD2]
                Qn[b, t, h, d] = T.cast(x0 * c - x1 * s, "bfloat16")
                Qn[b, t, h, d + RD2] = T.cast(x1 * c + x0 * s, "bfloat16")
            if h < hkv and t < SeqQLens[b]:
                k0 = q_rows + h * D
                v0 = q_rows + hkv * D + h * D
                wpos = SeqLens[b] - SeqQLens[b] + t
                blk = BlockTable[b, wpos // block_size]
                off = wpos % block_size
                var[0] = 0.0
                for k in T.serial(D):
                    xv = T.cast(QKV[b, t, k0 + k], "float32")
                    var[0] += xv * xv
                rstdk = T.rsqrt(var[0] / D + eps)
                for d in T.Parallel(D):
                    KPool[blk, h, off, d] = T.cast(T.cast(QKV[b, t, k0 + d], "float32") * rstdk * Wk[d], "bfloat16")
                    VPool[blk, h, off, d] = QKV[b, t, v0 + d]
                for d in T.Parallel(RD2):
                    ang = posf * InvFreq[d]
                    c = T.cos(ang)
                    s = T.sin(ang)
                    x0 = T.cast(QKV[b, t, k0 + d], "float32") * rstdk * Wk[d]
                    x1 = T.cast(QKV[b, t, k0 + d + RD2], "float32") * rstdk * Wk[d + RD2]
                    KPool[blk, h, off, d] = T.cast(x0 * c - x1 * s, "bfloat16")
                    KPool[blk, h, off, d + RD2] = T.cast(x1 * c + x0 * s, "bfloat16")
        return Qn

    return attn_prep
