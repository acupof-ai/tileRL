"""Tensor parallelism: shard the config, then shard the weights to match.

Every shape the model reshapes with comes from ``cfg``, so dividing head counts
and ``intermediate_size`` by the world size runs ``_full_attn``/``_gdn``/``_mlp``
on the local shard with no edit, and the KV/GDN pools shard with it.
Column-parallel (split output rows) needs no communication; row-parallel
(split input columns) leaves a partial sum, all-reduced in ``Model._add_via``.

Rules from vLLM / SGLang / TensorRT-LLM: fewer KV heads than ranks replicates
whole KV heads at load time (``rank // kv_replicas``), never ``head_dim``; a
quantization block that does not divide the shard is a hard error; slice in the
logical layout before ``Backend._served_fp4`` twiddles the bytes. Sharding runs
BEFORE ``_fuse_projections``, so only the two weights the checkpoint stores
concatenated need segment bookkeeping.

Capacity, 27B on 8 H20s (4 KV heads, so TP=8 stores the cache twice), concurrent
requests at depth 8192: DP=8 826, TP=8 605, TP=4 x DP=2 1042. TP=4 x DP=2 is
this model's configuration.

# ponytail: torch.distributed collectives cannot be captured in a CUDA graph, so
# TP forfeits the decode graph (7.9x at B=1) until Backend.all_reduce grows a
# pynccl-style path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

#: keys split on dim 0 (output rows). Value is the segment list for a weight
#: the checkpoint stores concatenated; () means one contiguous output range.
_COLUMN: dict[str, tuple[str, ...]] = {
    "q_proj": (),
    "k_proj": (),
    "v_proj": (),
    "gate_proj": (),
    "up_proj": (),
    "in_proj_z": (),
    "in_proj_a": (),
    "in_proj_b": (),
    "dt_bias": (),
    "a_log": (),
    "lm_head": (),  # vocab-parallel: logits are all-gathered after the head
    "in_proj_qkv": ("lq", "lk", "lv"),
    "conv1d": ("lq", "lk", "lv"),
}

#: keys split on dim 1 (input columns): each rank produces a partial sum.
_ROW = frozenset({"o_proj", "down_proj", "out_proj"})

_QUANT_SUFFIX = ("wq", "scale", "oscale", "w8", "wscale", "lora_a", "lora_b")

#: fp4 packs two values per byte and carries a scale per 16 input columns.
_FP4_ROW_ALIGN = 32
_FP8_BLOCK = 128


@dataclass(frozen=True)
class Mesh:
    """A rank's position in the ``(dp, tp, cp)`` grid. Sizes multiply to ``world``.

    Layout is **cp fastest, then tp, then dp**::

        rank = ((dp_i * tp) + tp_i) * cp + cp_i

    so a tp group is contiguous and lands inside one node, which is where NVLink
    is. dp ranks talk once per step and go outermost.

    ``cp > 1`` is refused: the axis is in the arithmetic because leaving it out
    would mean renumbering every rank later, but no context-parallel op exists,
    and a mesh that accepts ``cp=2`` would silently run tp-only math on a
    sequence it claims to have split.
    """

    dp: int = 1
    tp: int = 1
    cp: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        for name, n in (("dp", self.dp), ("tp", self.tp), ("cp", self.cp)):
            if n < 1:
                raise ValueError(f"{name}={n} must be >= 1")
        if self.cp != 1:
            raise ValueError(
                "cp>1 is not implemented: no context-parallel op exists, so a cp mesh "
                "would run tp-only math on a sequence it claims to have split"
            )
        if not 0 <= self.rank < self.world:
            raise ValueError(f"rank={self.rank} outside world={self.world}")

    @property
    def world(self) -> int:
        return self.dp * self.tp * self.cp

    @property
    def cp_rank(self) -> int:
        return self.rank % self.cp

    @property
    def tp_rank(self) -> int:
        return (self.rank // self.cp) % self.tp

    @property
    def dp_rank(self) -> int:
        return self.rank // (self.cp * self.tp)

    def tp_group(self) -> list[int]:
        """The ranks sharing this rank's tp group, in tp_rank order."""
        base = self.dp_rank * self.tp * self.cp + self.cp_rank
        return [base + i * self.cp for i in range(self.tp)]

    def dp_group(self) -> list[int]:
        """The ranks sharing this rank's tp and cp position, one per dp replica."""
        off = self.tp_rank * self.cp + self.cp_rank
        return [d * self.tp * self.cp + off for d in range(self.dp)]


def pad_vocab(vocab: int, world: int, to: int = 64) -> int:
    """Vocab rounded up so the TP divide is exact; trim ``logits[:, :vocab]`` after the gather."""
    step = to * world
    return -(-vocab // step) * step


def kv_replicas(total_kv_heads: int, world: int) -> tuple[int, int]:
    """``(kv_heads_per_rank, replicas)``: fewer KV heads than ranks means each
    rank keeps one head and consecutive groups of ``replicas`` ranks share it."""
    if world >= total_kv_heads:
        if world % total_kv_heads:
            raise ValueError(
                f"tp={world} must be a multiple of num_kv_heads={total_kv_heads} to replicate"
            )
        return 1, world // total_kv_heads
    if total_kv_heads % world:
        raise ValueError(f"tp={world} must divide num_kv_heads={total_kv_heads}")
    return total_kv_heads // world, 1


def is_sharded(key: str) -> bool:
    """Does this rank hold only a SLICE of this tensor, or the whole of it?

    Names only, so it answers from the sharded config too -- ``model.cfg`` after
    ``tp_config`` no longer knows the original head counts, and reading
    ``num_kv_heads`` off it would call a sharded ``k_proj`` replicated (tiny at
    tp=2 shards kv 2 -> 1, and 1 reads as "one head, replicated everywhere").
    Callers that reduce across ranks need this: a replicated tensor is already
    identical on every rank, and summing it counts it ``world`` times.

    A LoRA pair splits on ONE side only, because it is attached after sharding
    and inherits the base weight's narrow dimension: on a column-parallel base
    (split output rows) B is [n/w, r] and A is the full [r, k]; on a row-parallel
    base (split input columns) A is [r, k/w] and B is the full [n, r]. Measured
    on tiny at tp=2: q_proj.lora_a (4,64) both ways while lora_b goes 128->64,
    and o_proj/down_proj the mirror image.
    # ponytail: a model with exactly ONE kv head replicates k/v to every rank, and
    # this still says sharded; take the full cfg here when such a config exists.
    """
    head, _, suf = key.rpartition(".")
    if suf in ("lora_a", "lora_b"):
        base = head.rsplit(".", 1)[-1]
        # A is narrow on a row-parallel base, B is narrow on a column-parallel one.
        return base in _ROW if suf == "lora_a" else base in _COLUMN
    base = _base_name(key)
    return base in _ROW or base in _COLUMN


def _base_name(key: str) -> str:
    head, _, suf = key.rpartition(".")
    return (head if suf in _QUANT_SUFFIX else key).rsplit(".", 1)[-1]


def _kind(key: str, cfg, world: int):
    """``(None|"row"|"col", dim, segs, replicas)`` -- how ``shard_params`` splits
    a key of the FULL model. ``replicas`` is 1 except for K/V, which replicate
    whole heads across the ranks sharing one."""
    base = _base_name(key)
    replicas = 1
    if base in ("k_proj", "v_proj"):
        _, replicas = kv_replicas(cfg.num_kv_heads, world)
        if world // replicas == 1:  # one KV head in the whole model: every rank keeps it
            return None, 0, (), 1
    if base in _ROW:
        return "row", 1, (), replicas
    if base in _COLUMN:
        return "col", 0, _segment_sizes(cfg, _COLUMN[base]), replicas
    return None, 0, (), 1  # norms, embed_tokens, biases: replicated


def tp_config(cfg, world: int):
    """``cfg`` with every sharded dimension divided by ``world``; ``hidden_size``
    stays whole (the replicated activation width)."""
    if world == 1:
        return cfg
    for name, val in (
        ("num_attention_heads", cfg.num_attention_heads),
        ("intermediate_size", cfg.intermediate_size),
        ("linear_num_value_heads", cfg.linear_num_value_heads),
        ("linear_num_key_heads", cfg.linear_num_key_heads),
    ):
        if val % world:
            raise ValueError(f"tp={world} does not divide {name}={val}")
    kv, _ = kv_replicas(cfg.num_kv_heads, world)
    return replace(
        cfg,
        num_attention_heads=cfg.num_attention_heads // world,
        num_kv_heads=kv,
        intermediate_size=cfg.intermediate_size // world,
        linear_num_value_heads=cfg.linear_num_value_heads // world,
        linear_num_key_heads=cfg.linear_num_key_heads // world,
    )


def _segment_sizes(cfg, names: tuple[str, ...]) -> tuple[int, ...]:
    sizes = {"lq": cfg.linear_q_dim, "lk": cfg.linear_k_dim, "lv": cfg.linear_v_dim}
    return tuple(sizes[n] for n in names)


def _check_align(stem: str, kind: str, n: int, world: int, fp8: bool) -> None:
    if kind == "row":
        need = _FP8_BLOCK if fp8 else _FP4_ROW_ALIGN
    else:
        need = _FP8_BLOCK if fp8 else 1
    if need > 1 and (n // world) % need:
        raise ValueError(
            f"{stem}: {kind} shard {n}/{world} = {n // world} is not a multiple of the "
            f"{'fp8 128-block' if fp8 else 'fp4 16-column scale group'} ({need})"
        )


def _take(t: torch.Tensor, segs: tuple[int, ...], rank: int, world: int, dim: int, div: int = 1):
    """Rank's share of each segment along ``dim``, re-concatenated. ``div`` is the
    dim's packing factor: 2 for fp4 columns, 16 for an fp4 scale, 128 for an fp8 scale."""
    segs = segs or (t.shape[dim] * div,)
    parts, off = [], 0
    for seg in segs:
        n = seg // div
        step = n // world
        parts.append(t.narrow(dim, off + rank * step, step))
        off += n
    return torch.cat(parts, dim=dim).contiguous() if len(parts) > 1 else parts[0].contiguous()


def shard_params(params: dict, cfg, rank: int, world: int) -> dict:
    """``params`` for the FULL model -> this rank's shard. ``cfg`` is the full
    config and ``params`` must be unfused (``load_hf(fuse_projections=False)``)."""
    if world == 1:
        return params
    out: dict[str, torch.Tensor] = {}
    for key, t in params.items():
        head, _, suf = key.rpartition(".")
        stem = head if suf in _QUANT_SUFFIX else key
        kind, dim, segs, replicas = _kind(key, cfg, world)
        if kind is None:  # norms, embed_tokens, biases: replicated
            out[key] = t
            continue
        # K and V replicate across the ranks that share a head; Q does not.
        r, w = rank // replicas, world // replicas
        if t.ndim == 1:  # oscale / dt_bias / a_log
            out[key] = t if kind == "row" else _take(t, segs, r, w, 0)
            continue
        if suf == "wq":
            _check_align(stem, kind, t.shape[1] * 2 if dim == 1 else t.shape[0], w, False)
            out[key] = _take(t, segs, r, w, dim, div=2 if dim == 1 else 1)
        elif suf == "scale":  # fp4 block scale: one per 16 input columns
            out[key] = _take(t, segs, r, w, dim, div=16 if dim == 1 else 1)
        elif suf == "wscale":  # fp8 block scale: 128 x 128
            _check_align(stem, kind, t.shape[dim] * _FP8_BLOCK, w, True)
            out[key] = _take(t, segs, r, w, dim, div=_FP8_BLOCK)
        elif suf == "w8":
            _check_align(stem, kind, t.shape[dim], w, True)
            out[key] = _take(t, segs, r, w, dim)
        else:
            out[key] = _take(t, segs, r, w, dim)
    return out


if __name__ == "__main__":  # runnable check: the rules, not the plumbing
    from .config import qwen36_27b

    cfg = qwen36_27b()

    assert kv_replicas(4, 4) == (1, 1)
    assert kv_replicas(4, 8) == (1, 2)
    assert kv_replicas(4, 2) == (2, 1)

    c8 = tp_config(cfg, 8)
    assert c8.num_kv_heads == 1 and c8.num_attention_heads == 3
    assert c8.hidden_size == cfg.hidden_size

    # k/v must land on the same slice for both ranks of a replica pair.
    kv = torch.arange(cfg.num_kv_heads * cfg.head_dim, dtype=torch.float32).unsqueeze(1)
    p = {"layers.0.k_proj": kv}
    a = shard_params(p, cfg, 0, 8)["layers.0.k_proj"]
    b = shard_params(p, cfg, 1, 8)["layers.0.k_proj"]
    assert torch.equal(a, b), "ranks sharing a KV head must hold the same rows"
    c = shard_params(p, cfg, 2, 8)["layers.0.k_proj"]
    assert not torch.equal(a, c), "the next replica group must hold a different head"

    bad = {"layers.0.down_proj.w8": torch.zeros(64, 8 * 100)}
    try:
        shard_params(bad, cfg, 0, 8)
    except ValueError as e:
        assert "128-block" in str(e), e
    else:
        raise AssertionError("misaligned fp8 row shard must raise")

    assert pad_vocab(248320, 4) == 248320
    assert pad_vocab(1000, 8) == 1024

    # is_sharded must agree with what shard_params actually did, for EVERY param
    # including the LoRA pairs -- which split on one side only, and whose side
    # flips between a row- and a column-parallel base. A wrong answer here is a
    # gradient counted twice in the clip norm, which is silent.
    from . import model as _model_mod
    from .config import tiny as _tiny

    _cfg = _tiny()
    _one = _model_mod.build_random(_cfg, seed=0, keep_master=True)
    _model_mod.add_lora(_one, rank=4)
    _loc = _model_mod.Model(tp_config(_cfg, 2),
                            shard_params(_model_mod.build_random(
                                _cfg, seed=0, keep_master=True).params, _cfg, 0, 2))
    _model_mod.add_lora(_loc, rank=4)
    _n = 0
    for _k, _full in _one.params.items():
        if _k not in _loc.params:
            continue
        _n += 1
        assert is_sharded(_k) == (_full.shape != _loc.params[_k].shape), (
            f"{_k}: is_sharded={is_sharded(_k)} but {tuple(_full.shape)} -> "
            f"{tuple(_loc.params[_k].shape)}")
    assert _n >= 59, f"only {_n} params compared"

    # Mesh: every rank in a (dp=2, tp=4) world lands in exactly one tp group and
    # one dp group, and the two groups intersect in that rank alone. A layout bug
    # that duplicates or drops a rank fails here rather than as a hung collective.
    m0 = Mesh(dp=2, tp=4, rank=0)
    assert m0.world == 8
    seen_tp, seen_dp = [], []
    for r in range(8):
        m = Mesh(dp=2, tp=4, rank=r)
        assert m.dp_rank * 4 + m.tp_rank == r, r  # cp=1, so rank decomposes exactly
        seen_tp.append(tuple(m.tp_group()))
        seen_dp.append(tuple(m.dp_group()))
        assert r in m.tp_group() and r in m.dp_group()
        assert set(m.tp_group()) & set(m.dp_group()) == {r}
    assert sorted(set(seen_tp)) == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert len(set(seen_dp)) == 4 and all(len(g) == 2 for g in set(seen_dp))
    assert sorted(r for g in set(seen_dp) for r in g) == list(range(8))

    for bad in (dict(cp=2), dict(tp=0), dict(tp=2, rank=2), dict(tp=2, rank=-1)):
        try:
            Mesh(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Mesh({bad}) must raise")

    print("tensor_parallel: gqa replication + alignment refusal + vocab pad + mesh OK")
