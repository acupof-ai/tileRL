"""Test/selfcheck utilities: the torch-eager reference backend.

NOT framework code — the deterministic CPU backend for tests and selfchecks
that want a backend without the TileLang JIT. Ops delegate to
:mod:`tilerl.ops.reference` (the parity oracle) via ``__getattr__``; the ops
the reference lacks live here: gated dense attention, fp4 linear (drops the
recording-only ``master`` kwarg), elementwise add, gated paged attention
(reference has no gated paged path), and the paged KV scatter (the pool's
torch loop — the sm90 kernel's reference semantics).
"""

from __future__ import annotations

import torch

__all__ = ["RefBackend"]

#: Ops delegated verbatim to tilerl.ops.reference.
_REF_OPS = frozenset(
    {
        "rmsnorm",
        "rmsnorm_bwd",
        "rope",
        "rope_bwd",
        "linear",
        "linear_bwd",
        "linear_fp4_bwd",
        "linear_attn_chunk",
        "linear_attn_step",
        "linear_attn_bwd",
        "silu_mul",
        "silu_mul_bwd",
        "softmax",
        "embedding",
        "embedding_bwd",
        "sample",
    }
)


class RefBackend:
    """Torch-eager reference backend implementing the model->backend contract."""

    name = "reference"
    target = "cpu"
    device = torch.device("cpu")

    def __getattr__(self, name):
        if name in _REF_OPS:
            from .ops import reference

            return getattr(reference, name)
        raise AttributeError(name)

    def linear_fp4(self, x, wq, scale, master=None):
        from .ops import reference

        return reference.linear_fp4(x, wq, scale)  # master is recording-only

    def attention(self, q, k, v, scale, gate=None):
        from .ops import reference

        out = reference.dense_attention(q, k, v, scale)
        if gate is not None:
            out = out * torch.sigmoid(gate.float())
        return out

    def add(self, a, b):
        return a + b

    def write_tokens(self, k, v, kv, layer_idx):
        # The pool's torch loop (the sm90 tilelang kernel has no reference
        # counterpart — the pool loop IS the reference semantics).
        kv.kv_pool.write_tokens(k, v, kv, layer_idx)

    def paged_attention(self, q, k_pool, v_pool, block_table, seq_lens, scale, gate=None):
        b, t, hq, d = q.shape
        hkv = k_pool.shape[1]
        rep = hq // hkv
        out = torch.zeros(b, t, hq, d, dtype=q.dtype)
        for bi in range(b):
            s = int(seq_lens[bi])
            nblk = (s + k_pool.shape[2] - 1) // k_pool.shape[2]
            blks = block_table[bi, :nblk].long()
            # [nblk,Hkv,BLOCK,D] -> [Hkv,s,D]
            k_seq = k_pool[blks].permute(1, 0, 2, 3).reshape(hkv, -1, d)[:, :s]
            v_seq = v_pool[blks].permute(1, 0, 2, 3).reshape(hkv, -1, d)[:, :s]
            k_seq = k_seq.unsqueeze(1).expand(hkv, rep, s, d).reshape(hq, s, d)
            v_seq = v_seq.unsqueeze(1).expand(hkv, rep, s, d).reshape(hq, s, d)
            scores = torch.einsum("thd,hsd->ths", q[bi].float(), k_seq.float()) * scale
            q_pos = torch.arange(s - t, s)
            causal = torch.arange(s).unsqueeze(0) <= q_pos.unsqueeze(1)  # [t,s]
            scores = scores.masked_fill(~causal.unsqueeze(1), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            ob = torch.einsum("ths,hsd->thd", attn, v_seq.float())
            if gate is not None:
                ob = ob * torch.sigmoid(gate[bi].float())
            out[bi] = ob.to(q.dtype)
        return out
