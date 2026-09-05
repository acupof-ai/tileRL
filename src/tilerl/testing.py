"""The torch-eager reference backend for tests and selfchecks (no TileLang JIT).
Ops delegate to :mod:`tilerl_kernels.reference` via ``__getattr__``; the ops the
reference lacks live here."""

from __future__ import annotations

import torch

_REF_OPS = frozenset(
    {
        "rmsnorm",
        "rmsnorm_bwd",
        "rope",
        "rope_bwd",
        "linear",
        "linear_bwd",
        "linear_frozen_bwd",
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
    name = "reference"
    target = "cpu"
    device = torch.device("cpu")
    tp_world = 1
    tp_rank = 0

    def init_tp(self, world: int, rank: int) -> None:
        """Same seam as ``Backend.init_tp``; gloo, so the TP gate runs CPU-only."""
        if world == 1:
            return
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group("gloo", world_size=world, rank=rank)
        self.tp_world, self.tp_rank = world, rank

    def all_reduce(self, x):
        if self.tp_world == 1:
            return x
        import torch.distributed as dist

        dist.all_reduce(x)
        return x.view_as(x)  # distinct object: the tape addresses by id()

    def all_gather(self, x, dim: int = -1):
        if self.tp_world == 1:
            return x
        import torch.distributed as dist

        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(self.tp_world)]
        dist.all_gather(parts, x)
        return torch.cat(parts, dim=dim)

    def tp_fork(self, x):
        return x if self.tp_world == 1 else x.view_as(x)

    def __getattr__(self, name):
        if name in _REF_OPS:
            from tilerl_kernels import reference

            return getattr(reference, name)
        raise AttributeError(name)

    def materialize(self, params):
        return params

    def gdn_decode(self, *args, **kwargs):
        return None  # no fused in-place decode: the model takes the gather/scatter path

    def attn_prep(self, *args, **kwargs):
        return None

    def rmsnorm_f32(self, x, w, eps):
        from tilerl_kernels import reference

        return reference.rmsnorm(x, w, eps)  # the reference is f32 already

    def linear_fp4(self, x, wq, scale, master=None, oscale=None):
        from tilerl_kernels import reference

        return reference.linear_fp4(x, wq, scale, oscale)  # master is recording-only

    def linear_fp8(self, x, w8, wscale, master=None, oscale=None):
        from tilerl_kernels import reference

        return reference.linear_fp8(x, w8, wscale, oscale)

    def attention(self, q, k, v, scale, gate=None):
        from tilerl_kernels import reference

        out = reference.dense_attention(q, k, v, scale)
        if gate is not None:
            out = out * torch.sigmoid(gate.float())
        return out

    def attention_bwd(self, grad, q, k, v, scale):
        from tilerl_kernels import reference

        return reference.dense_attention_bwd(grad, q, k, v, float(scale))

    def add(self, a, b):
        return a + b

    def write_tokens(self, k, v, kv, layer_idx):
        kv.kv_pool.write_tokens(k, v, kv, layer_idx)  # the pool loop is the reference semantics

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
