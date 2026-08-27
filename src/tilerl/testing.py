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
        "linear_attn_chunk",
        "linear_attn_bwd",
        "attention_gate_bwd",
        "silu_mul",
        "silu_mul_bwd",
        "softmax",
        "cross_entropy_loss_grad",
        "state_gather",
        "state_scatter",
        "embedding",
        "embedding_bwd",
        "sample",
        "sample_batch",
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

    def materialize(self, params):
        return params  # torch serves every format on CPU: nothing to convert or move

    def gdn_decode(self, *args, **kwargs):
        return None  # no fused in-place decode here: the model takes the gather/scatter path

    def attn_prep(self, *args, **kwargs):
        return None

    def linear_fp4(self, x, wq, scale, master=None, oscale=None):
        from .ops import reference

        return reference.linear_fp4(x, wq, scale, oscale)  # master is recording-only

    def linear_fp8(self, x, w8, wscale, master=None, oscale=None):
        from .ops import reference

        return reference.linear_fp8(x, w8, wscale, oscale)  # master is recording-only

    def attention(self, q, k, v, scale, gate=None):
        from .ops import reference

        out = reference.dense_attention(q, k, v, scale)
        if gate is not None:
            out = out * torch.sigmoid(gate.float())
        return out

    def attention_bwd(self, grad, q, k, v, scale):
        from .ops import reference

        return reference.dense_attention_bwd(grad, q, k, v, float(scale))

    def add(self, a, b):
        return a + b

    def write_tokens(self, k, v, kv, layer_idx):
        # The pool's torch loop (the sm90 tilelang kernel has no reference
        # counterpart — the pool loop IS the reference semantics).
        kv.kv_pool.write_tokens(k, v, kv, layer_idx)

    def paged_attention(
        self, q, k_pool, v_pool, block_table, seq_lens, scale, gate=None, seq_q_lens=None
    ):
        b, t, hq, d = q.shape
        hkv = k_pool.shape[1]
        rep = hq // hkv
        out = torch.zeros(b, t, hq, d, dtype=q.dtype, device=q.device)
        for bi in range(b):
            s = int(seq_lens[bi])
            sq = t if seq_q_lens is None else int(seq_q_lens[bi])
            nblk = (s + k_pool.shape[2] - 1) // k_pool.shape[2]
            blks = block_table[bi, :nblk].long()
            # [nblk,Hkv,BLOCK,D] -> [Hkv,s,D]
            k_seq = k_pool[blks].permute(1, 0, 2, 3).reshape(hkv, -1, d)[:, :s]
            v_seq = v_pool[blks].permute(1, 0, 2, 3).reshape(hkv, -1, d)[:, :s]
            k_seq = k_seq.unsqueeze(1).expand(hkv, rep, s, d).reshape(hq, s, d)
            v_seq = v_seq.unsqueeze(1).expand(hkv, rep, s, d).reshape(hq, s, d)
            scores = torch.einsum("thd,hsd->ths", q[bi, :sq].float(), k_seq.float()) * scale
            q_pos = torch.arange(s - sq, s, device=q.device)
            causal = torch.arange(s, device=q.device).unsqueeze(0) <= q_pos.unsqueeze(1)
            scores = scores.masked_fill(~causal.unsqueeze(1), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            ob = torch.einsum("ths,hsd->thd", attn, v_seq.float())
            if gate is not None:
                ob = ob * torch.sigmoid(gate[bi, :sq].float())
            out[bi, :sq] = ob.to(q.dtype)
        return out
