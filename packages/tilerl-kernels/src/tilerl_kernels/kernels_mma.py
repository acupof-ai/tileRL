"""sm90 pass config, paged K/V write and the fused attention prep."""

from __future__ import annotations

import tilelang
import tilelang.language as T


def _pass_configs() -> dict[str, object]:
    # the static race check false-positives on per-thread fragments
    return {"tl.disable_data_race_check": True}


def make_write_tokens(target: str):
    """Scatter K/V [B,S,Hkv,D] into the paged pool at [seq_len-seq_q, seq_len)
    (vLLM reshape_and_cache indexing). One launch, graph-capturable; rows are
    left-aligned valid tokens, SeqQLens bounds each row."""

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
    """q/k norm + partial RoPE + paged K/V write in one launch off the fused-qkv
    output (agent-infer decode_prep). QKV rows: [hq x (query D ; gate D)] ++
    [hkv x D] ++ [hkv x D]; block (b*s, h) does q head h, and k/v head h when
    h < hkv."""

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
        QKV: T.Tensor((B, S, NQKV), "float32")  # the fp4 GEMV writes f32
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
                    VPool[blk, h, off, d] = T.cast(QKV[b, t, v0 + d], "bfloat16")
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
