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
