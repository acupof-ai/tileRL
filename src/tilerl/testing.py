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
        "gdn_span_ab",
        "gdn_span_ab_raw",
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

    _tp_pg = None
    dp_world = 1
    _dp_pg = None
    cp_world = 1
    cp_rank = 0
    _cp_pg = None

    def init_tp(self, world: int, rank: int, tp_groups: list[list[int]] | None = None,
                dp_groups: list[list[int]] | None = None,
                cp_groups: list[list[int]] | None = None) -> None:
        """Same seam as ``Backend.init_tp``; gloo, so the TP gate runs CPU-only."""
        if world == 1:
            return
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group("gloo", world_size=world, rank=rank)

        def join(groups, axis):
            mine, pg_mine = None, None
            for g in groups:  # every rank builds every group, in order
                pg = dist.new_group(list(g))
                if rank in g:
                    mine, pg_mine = g, pg
            if mine is None:
                raise ValueError(f"rank {rank} is in none of the {axis} groups {groups}")
            return mine, pg_mine

        if tp_groups:
            mine, self._tp_pg = join(tp_groups, "tp")
            self.tp_world, self.tp_rank = len(mine), mine.index(rank)
        else:
            self.tp_world, self.tp_rank = world, rank
        if dp_groups:
            mine, self._dp_pg = join(dp_groups, "dp")
            self.dp_world = len(mine)
        if cp_groups:
            mine, self._cp_pg = join(cp_groups, "cp")
            self.cp_world, self.cp_rank = len(mine), mine.index(rank)

    def dp_reduce(self, x):
        """Mean across the dp replicas; see ``Backend.dp_reduce``."""
        if self.dp_world == 1:
            return x
        import torch.distributed as dist

        dist.all_reduce(x, group=self._dp_pg)
        return x.div_(self.dp_world)

    def all_reduce(self, x):
        if self.tp_world == 1:
            return x
        import torch.distributed as dist

        dist.all_reduce(x, group=self._tp_pg)
        return x.view_as(x)  # distinct object: the tape addresses by id()

    def all_gather(self, x, dim: int = -1):
        if self.tp_world == 1:
            return x
        import torch.distributed as dist

        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(self.tp_world)]
        dist.all_gather(parts, x, group=self._tp_pg)
        return torch.cat(parts, dim=dim)

    def tp_fork(self, x):
        return x if self.tp_world == 1 else x.view_as(x)

    def cp_gather(self, x, dim: int = 1):
        """See ``Backend.cp_gather``: chunks come back in cp_rank order."""
        if self.cp_world == 1:
            return x
        import torch.distributed as dist

        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(self.cp_world)]
        dist.all_gather(parts, x, group=self._cp_pg)
        return torch.cat(parts, dim=dim)

    def cp_reduce_scatter(self, x, dim: int = 1):
        """See ``Backend.cp_reduce_scatter``: a sum, not a slice."""
        if self.cp_world == 1:
            return x
        import torch.distributed as dist

        parts = [p.contiguous() for p in x.chunk(self.cp_world, dim=dim)]
        out = torch.empty_like(parts[0])
        dist.reduce_scatter(out, parts, group=self._cp_pg)
        return out

    def cp_prefix_scan(self, a, b, chunk_ids=None):
        """See ``Backend.cp_prefix_scan``; the same shared implementation."""
        if self.cp_world == 1:
            return None, None
        from tilerl_kernels import reference

        return reference.affine_prefix_scan(a, b, self._cp_pg, self.cp_rank,
                                            self.cp_world, chunk_ids)

    def cp_halo(self, x, ids_by_rank, width: int):
        """See ``Backend.cp_halo``: left context for the depthwise conv."""
        if self.cp_world == 1:
            return [None] * len(ids_by_rank[self.cp_rank])
        import torch.distributed as dist

        tails = x[:, :, -width:].contiguous()
        parts = [torch.empty_like(tails) for _ in range(self.cp_world)]
        dist.all_gather(parts, tails, group=self._cp_pg)
        owner = {c: (r, i) for r, ids in enumerate(ids_by_rank) for i, c in enumerate(ids)}
        return [None if c == 0 else parts[owner[c - 1][0]][owner[c - 1][1]]
                for c in ids_by_rank[self.cp_rank]]

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

    def attention(self, q, k, v, scale, gate=None, q_pos=None, k_pos=None):
        from tilerl_kernels import reference

        out = reference.dense_attention(q, k, v, scale, q_pos, k_pos)
        if gate is not None:
            out = out * torch.sigmoid(gate.float())
        return out

    def attention_bwd(self, grad, q, k, v, scale, q_pos=None, k_pos=None):
        from tilerl_kernels import reference

        return reference.dense_attention_bwd(grad, q, k, v, float(scale), q_pos, k_pos)

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
