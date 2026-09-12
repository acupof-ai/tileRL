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
    left-aligned valid tokens, SeqQLens bounds each row. bf16 IO follows the
    sm90 pool; sm70's f32 pool has its own twin below."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def write_tokens(K, V, KPool, VPool, BlockTable, SeqLens, SeqQLens, PageBase,
                     block_size, threads):
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
        # Logical page of each row's table column 0 (dense 0; sparse own-only
        # table starts at page_base). Index the table by the RELATIVE column: an
        # absolute index reads padding past the own table.
        PageBase: T.Tensor((B,), "int32")
        with T.Kernel(B * S, H, threads=threads) as (bt, h):
            b = bt // S
            t = bt % S
            if t < SeqQLens[b]:
                pos = SeqLens[b] - SeqQLens[b] + t
                blk = BlockTable[b, pos // block_size - PageBase[b]]
                off = pos % block_size
                for d in T.Parallel(D):
                    KPool[blk, h, off, d] = K[b, t, h, d]
                    VPool[blk, h, off, d] = V[b, t, h, d]

    return write_tokens


def make_write_tokens_f32(target: str):
    """f32-pool twin of :func:`make_write_tokens` for sm70.

    sm70's attention kernel is f32-IO, and a bf16 pool made every attention
    call cast the WHOLE plane — 4.71 ms/token, 14% of a 4096-ctx token. The
    body is duplicated rather than parameterized because the eager builder
    re-executes it with only its own kwargs bound, so a closure dtype is not in
    scope inside a T.Tensor annotation.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def write_tokens_f32(K, V, KPool, VPool, BlockTable, SeqLens, SeqQLens, PageBase,
                         block_size, threads):
        B, S, H, D = T.const("B, S, H, D")
        NB = T.const("NB")
        Mb = T.const("Mb")
        K: T.Tensor((B, S, H, D), "float32")
        V: T.Tensor((B, S, H, D), "float32")
        KPool: T.Tensor((NB, H, block_size, D), "float32")
        VPool: T.Tensor((NB, H, block_size, D), "float32")
        BlockTable: T.Tensor((B, Mb), "int32")
        SeqLens: T.Tensor((B,), "int32")
        SeqQLens: T.Tensor((B,), "int32")
        PageBase: T.Tensor((B,), "int32")  # see write_tokens
        with T.Kernel(B * S, H, threads=threads) as (bt, h):
            b = bt // S
            t = bt % S
            if t < SeqQLens[b]:
                pos = SeqLens[b] - SeqQLens[b] + t
                blk = BlockTable[b, pos // block_size - PageBase[b]]
                off = pos % block_size
                for d in T.Parallel(D):
                    KPool[blk, h, off, d] = K[b, t, h, d]
                    VPool[blk, h, off, d] = V[b, t, h, d]

    return write_tokens_f32


def make_write_tokens_fp8(target: str):
    """fp8-pool twin of :func:`make_write_tokens`: quantizes per token as it scatters.

    The scale is per (block, head, token) because that is the only grid this launch shape can
    produce: one thread block per (b, t, h), so an amax over head_dim is a reduction inside
    the block, while a per-block amax would span the 16 thread blocks sharing a pool block.
    Body duplicated rather than parameterized for the same reason as the f32 twin -- a closure
    dtype is not in scope inside a T.Tensor annotation.
    """
    FP8_MAX = 448.0  # e4m3fn finite max

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def write_tokens_fp8(K, V, KPool, VPool, KScale, VScale, BlockTable, SeqLens, SeqQLens,
                         PageBase, block_size, threads):
        B, S, H, D = T.const("B, S, H, D")
        NB = T.const("NB")
        Mb = T.const("Mb")
        K: T.Tensor((B, S, H, D), "bfloat16")
        V: T.Tensor((B, S, H, D), "bfloat16")
        KPool: T.Tensor((NB, H, block_size, D), "float8_e4m3fn")
        VPool: T.Tensor((NB, H, block_size, D), "float8_e4m3fn")
        KScale: T.Tensor((NB, H, block_size), "float32")
        VScale: T.Tensor((NB, H, block_size), "float32")
        BlockTable: T.Tensor((B, Mb), "int32")
        SeqLens: T.Tensor((B,), "int32")
        SeqQLens: T.Tensor((B,), "int32")
        PageBase: T.Tensor((B,), "int32")  # see write_tokens
        with T.Kernel(B * S, H, threads=threads) as (bt, h):
            b = bt // S
            t = bt % S
            if t < SeqQLens[b]:
                pos = SeqLens[b] - SeqQLens[b] + t
                blk = BlockTable[b, pos // block_size - PageBase[b]]
                off = pos % block_size
                tid = T.get_thread_binding(0)
                # strided partials then one shared reduce, not T.serial(D) per thread: the
                # redundant-loop form cost gdn_prep 4.89x (see registry sm70 notes).
                # Two 1-element fragments, not one of size 2: a fragment is per-thread and
                # tilelang allows only fragment[0].
                red = T.alloc_shared((threads, 2), "float32")
                pk = T.alloc_fragment((1,), "float32")
                pv = T.alloc_fragment((1,), "float32")
                pk[0] = 0.0
                pv[0] = 0.0
                for i in T.serial(T.ceildiv(D, threads)):
                    d = tid + i * threads
                    if d < D:
                        kv = T.cast(K[b, t, h, d], "float32")
                        vv = T.cast(V[b, t, h, d], "float32")
                        pk[0] = T.max(pk[0], T.max(kv, 0.0 - kv))
                        pv[0] = T.max(pv[0], T.max(vv, 0.0 - vv))
                red[tid, 0] = pk[0]
                red[tid, 1] = pv[0]
                T.tvm_storage_sync("shared")
                if tid == 0:
                    mk = T.alloc_fragment((1,), "float32")
                    mv = T.alloc_fragment((1,), "float32")
                    mk[0] = 0.0
                    mv[0] = 0.0
                    for i in T.serial(threads):
                        mk[0] = T.max(mk[0], red[i, 0])
                        mv[0] = T.max(mv[0], red[i, 1])
                    red[0, 0] = T.max(mk[0], 1e-12) / FP8_MAX
                    red[0, 1] = T.max(mv[0], 1e-12) / FP8_MAX
                T.tvm_storage_sync("shared")
                ks = red[0, 0]
                vs = red[0, 1]
                for d in T.Parallel(D):
                    KPool[blk, h, off, d] = T.cast(
                        T.cast(K[b, t, h, d], "float32") / ks, "float8_e4m3fn")
                    VPool[blk, h, off, d] = T.cast(
                        T.cast(V[b, t, h, d], "float32") / vs, "float8_e4m3fn")
                if tid == 0:
                    KScale[blk, h, off] = ks
                    VScale[blk, h, off] = vs

    return write_tokens_fp8


def make_attn_prep(target: str):
    """q/k norm + partial RoPE + paged K/V write in one launch off the fused-qkv
    output (agent-infer decode_prep). QKV rows: [hq x (query D ; gate D)] ++
    [hkv x D] ++ [hkv x D]; block (b*s, h) does q head h, and k/v head h when
    h < hkv."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def attn_prep(
        QKV, Wq, Wk, Positions, InvFreq, KPool, VPool, BlockTable, SeqLens, SeqQLens,
        PageBase, eps: T.float32, hq, hkv, block_size, threads,
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
        PageBase: T.Tensor((B,), "int32")  # see write_tokens
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
                blk = BlockTable[b, wpos // block_size - PageBase[b]]
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


def make_attn_prep_fp8(target: str):
    """fp8-pool twin of :func:`make_attn_prep`: same q/k norm + RoPE, quantized K/V write.

    The K amax is taken over the POST-RoPE values, not the normalized ones. RoPE is a
    rotation, so it preserves the pairwise norm but not the per-element absmax: a rotated
    element can reach sqrt(2)x its inputs, which against a pre-RoPE scale would quantize to
    633 and saturate e4m3's 448. K is staged in a fragment, reduced, then written.
    """
    FP8_MAX = 448.0

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def attn_prep_fp8(
        QKV, Wq, Wk, Positions, InvFreq, KPool, VPool, KScale, VScale, BlockTable, SeqLens,
        SeqQLens, PageBase, eps: T.float32, hq, hkv, block_size, threads,
    ):
        B, S, NQKV = T.const("B, S, NQKV")
        D = T.const("D")
        RD2 = T.const("RD2")
        NB = T.const("NB")
        Mb = T.const("Mb")
        QKV: T.Tensor((B, S, NQKV), "float32")
        Wq: T.Tensor((D,), "float32")
        Wk: T.Tensor((D,), "float32")
        Positions: T.Tensor((B, S), "int32")
        InvFreq: T.Tensor((RD2,), "float32")
        KPool: T.Tensor((NB, hkv, block_size, D), "float8_e4m3fn")
        VPool: T.Tensor((NB, hkv, block_size, D), "float8_e4m3fn")
        KScale: T.Tensor((NB, hkv, block_size), "float32")
        VScale: T.Tensor((NB, hkv, block_size), "float32")
        BlockTable: T.Tensor((B, Mb), "int32")
        SeqLens: T.Tensor((B,), "int32")
        SeqQLens: T.Tensor((B,), "int32")
        PageBase: T.Tensor((B,), "int32")  # see write_tokens
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
                blk = BlockTable[b, wpos // block_size - PageBase[b]]
                off = wpos % block_size
                var[0] = 0.0
                for k in T.serial(D):
                    xv = T.cast(QKV[b, t, k0 + k], "float32")
                    var[0] += xv * xv
                rstdk = T.rsqrt(var[0] / D + eps)
                # stage the FINAL K (normed, then roped over the leading 2*RD2 dims)
                Ks = T.alloc_shared((D,), "float32")
                for d in T.Parallel(D):
                    Ks[d] = T.cast(QKV[b, t, k0 + d], "float32") * rstdk * Wk[d]
                T.tvm_storage_sync("shared")
                for d in T.Parallel(RD2):
                    ang = posf * InvFreq[d]
                    c = T.cos(ang)
                    s = T.sin(ang)
                    x0 = T.cast(QKV[b, t, k0 + d], "float32") * rstdk * Wk[d]
                    x1 = T.cast(QKV[b, t, k0 + d + RD2], "float32") * rstdk * Wk[d + RD2]
                    Ks[d] = x0 * c - x1 * s
                    Ks[d + RD2] = x1 * c + x0 * s
                T.tvm_storage_sync("shared")
                tid = T.get_thread_binding(0)
                red = T.alloc_shared((threads, 2), "float32")
                pk = T.alloc_fragment((1,), "float32")
                pv = T.alloc_fragment((1,), "float32")
                pk[0] = 0.0
                pv[0] = 0.0
                for i in T.serial(T.ceildiv(D, threads)):
                    d = tid + i * threads
                    if d < D:
                        vv = T.cast(QKV[b, t, v0 + d], "float32")
                        pk[0] = T.max(pk[0], T.max(Ks[d], 0.0 - Ks[d]))
                        pv[0] = T.max(pv[0], T.max(vv, 0.0 - vv))
                red[tid, 0] = pk[0]
                red[tid, 1] = pv[0]
                T.tvm_storage_sync("shared")
                if tid == 0:
                    mk = T.alloc_fragment((1,), "float32")
                    mv = T.alloc_fragment((1,), "float32")
                    mk[0] = 0.0
                    mv[0] = 0.0
                    for i in T.serial(threads):
                        mk[0] = T.max(mk[0], red[i, 0])
                        mv[0] = T.max(mv[0], red[i, 1])
                    red[0, 0] = T.max(mk[0], 1e-12) / FP8_MAX
                    red[0, 1] = T.max(mv[0], 1e-12) / FP8_MAX
                T.tvm_storage_sync("shared")
                ks = red[0, 0]
                vs = red[0, 1]
                for d in T.Parallel(D):
                    KPool[blk, h, off, d] = T.cast(Ks[d] / ks, "float8_e4m3fn")
                    VPool[blk, h, off, d] = T.cast(
                        T.cast(QKV[b, t, v0 + d], "float32") / vs, "float8_e4m3fn")
                if tid == 0:
                    KScale[blk, h, off] = ks
                    VScale[blk, h, off] = vs
        return Qn

    return attn_prep_fp8
