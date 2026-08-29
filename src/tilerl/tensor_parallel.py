"""Tensor parallelism: shard the config, then shard the weights to match.

The model code is untouched by design. Every shape it reshapes with comes from
``cfg`` (head counts, intermediate size), so dividing those by the world size
makes ``_full_attn`` / ``_gdn`` / ``_mlp`` operate on the local shard with no
edit — and the KV and GDN state pools, built from the same cfg, shard with it.

Column-parallel (split output rows) needs no communication; row-parallel
(split input columns) leaves each rank holding a PARTIAL sum, so the three
row-parallel projections all-reduce before the residual add. All three go
through ``Model._add_via``, which is the only place that changes.

Why TP=4 and not 8 on Qwen3.8-27B: ``num_kv_heads`` is 4. At TP=8 each KV head
lives on two ranks and the KV cache is stored twice, which costs more capacity
than the deduplicated weights buy back (344 GiB of effective KV across 8 cards
vs 528 for plain DP). TP=4 x DP=2 divides cleanly and reaches 666 GiB.

# ponytail: embed_tokens/lm_head are replicated (0.64 GiB in fp4), which skips
# a vocab-parallel all-gather. Shard them if the replica ever costs more than
# the collective would.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

__all__ = ["tp_config", "shard_params", "ShardPlan"]

#: keys sharded on dim 0 (output rows) -> the concatenated segments they hold.
#: A fused weight is [seg0 | seg1 | ...]; slicing it uniformly would hand a
#: rank the tail of one segment and the head of the next.
_COLUMN = {
    "q_proj": ("q",),
    "k_proj": ("k",),
    "v_proj": ("v",),
    "qkv": ("q", "k", "v"),
    "gate_proj": ("inter",),
    "up_proj": ("inter",),
    "gate_up": ("inter", "inter"),
    "in_proj_qkv": ("lq", "lk", "lv"),
    "in_proj_z": ("lv",),
    "in_proj_a": ("nvh",),
    "in_proj_b": ("nvh",),
    "conv1d": ("lq", "lk", "lv"),
    "dt_bias": ("nvh",),
    "a_log": ("nvh",),
}

#: keys sharded on dim 1 (input columns): each rank produces a partial sum.
_ROW = {"o_proj", "down_proj", "out_proj"}

#: fp8 block-scale granularity; a shard boundary must land on it.
_FP8_BLOCK = 128


def _segments(cfg) -> dict[str, int]:
    d = cfg.head_dim
    return {
        "q": cfg.num_attention_heads * d * (2 if cfg.full_attn_gated else 1),
        "k": cfg.num_kv_heads * d,
        "v": cfg.num_kv_heads * d,
        "inter": cfg.intermediate_size,
        "lq": cfg.linear_q_dim,
        "lk": cfg.linear_k_dim,
        "lv": cfg.linear_v_dim,
        "nvh": cfg.linear_num_value_heads,
    }


def tp_config(cfg, world: int):
    """``cfg`` with every sharded dimension divided by ``world``.

    hidden_size stays whole: it is the width of the replicated activation that
    enters and leaves every layer.
    """
    if world == 1:
        return cfg
    if cfg.num_kv_heads % world:
        raise ValueError(
            f"tp={world} does not divide num_kv_heads={cfg.num_kv_heads}; "
            "replicating KV heads costs more capacity than it buys (see module docstring)"
        )
    for name, val in (
        ("num_attention_heads", cfg.num_attention_heads),
        ("intermediate_size", cfg.intermediate_size),
        ("linear_num_value_heads", cfg.linear_num_value_heads),
        ("linear_num_key_heads", cfg.linear_num_key_heads),
    ):
        if val % world:
            raise ValueError(f"tp={world} does not divide {name}={val}")
    return replace(
        cfg,
        num_attention_heads=cfg.num_attention_heads // world,
        num_kv_heads=cfg.num_kv_heads // world,
        intermediate_size=cfg.intermediate_size // world,
        linear_num_value_heads=cfg.linear_num_value_heads // world,
        linear_num_key_heads=cfg.linear_num_key_heads // world,
    )


class ShardPlan:
    """How one parameter key is split. ``kind`` is 'col', 'row' or None."""

    __slots__ = ("kind", "segments")

    def __init__(self, kind: str | None, segments: tuple[int, ...] = ()):
        self.kind = kind
        self.segments = segments


def plan_for(key: str, cfg) -> ShardPlan:
    base = key.split(".")[-1]
    if base in _ROW:
        return ShardPlan("row")
    segs = _COLUMN.get(base)
    if segs is None:
        return ShardPlan(None)
    sizes = _segments(cfg)
    return ShardPlan("col", tuple(sizes[s] for s in segs))


def _slice_segments(t: torch.Tensor, segments, rank: int, world: int, dim: int, div: int = 1):
    """Take rank's share of each segment along ``dim`` and re-concatenate.

    ``div`` is the packing factor of that dim (2 for fp4's two-per-byte rows,
    16 for a per-16-column fp4 scale, 128 for an fp8 block scale).
    """
    parts, off = [], 0
    for seg in segments:
        n = seg // div
        step = n // world
        parts.append(t.narrow(dim, off + rank * step, step))
        off += n
    return torch.cat(parts, dim=dim).contiguous() if len(parts) > 1 else parts[0].contiguous()


def _dequant(params: dict, key: str, cfg) -> torch.Tensor | None:
    """bf16 view of a quantized weight, for shards that break the block scale."""
    from .model import unpack_fp4

    if key in params:
        return params[key]
    wq = params.get(key + ".wq")
    if wq is not None:
        return unpack_fp4(wq, params[key + ".scale"], params.get(key + ".oscale"))
    return None


def shard_params(params: dict, cfg, rank: int, world: int) -> dict:
    """``params`` for the FULL model -> this rank's shard. ``cfg`` is the FULL
    config (call before :func:`tp_config`)."""
    if world == 1:
        return params
    out: dict[str, Any] = {}
    handled: set[str] = set()
    for key in list(params):
        base = key.rsplit(".", 1)
        stem = base[0] if len(base) == 2 and base[1] in _QUANT_SUFFIX else key
        if stem in handled:
            continue
        plan = plan_for(stem, cfg)
        if plan.kind is None:
            for s in ("",) + _QUANT_SUFFIX_DOT:
                if stem + s in params:
                    out[stem + s] = params[stem + s]
            handled.add(stem)
            continue
        handled.add(stem)
        _shard_one(params, out, stem, plan, cfg, rank, world)
    return out


_QUANT_SUFFIX = ("wq", "scale", "oscale", "w8", "wscale", "lora_a", "lora_b")
_QUANT_SUFFIX_DOT = tuple("." + s for s in _QUANT_SUFFIX)


def _blocks_ok(n: int, world: int) -> bool:
    return (n // world) % _FP8_BLOCK == 0


def _shard_one(params, out, stem, plan, cfg, rank, world) -> None:
    dim = 0 if plan.kind == "col" else 1
    w8 = params.get(stem + ".w8")
    wq = params.get(stem + ".wq")
    dense = params.get(stem)

    # An fp8 block scale covers 128 rows/cols; a shard that lands inside a
    # block cannot slice it. in_proj_a/b are [48, hidden] - one block row - so
    # they take the dequantized path. It costs 47 MB over the whole model.
    if w8 is not None:
        n = w8.shape[dim]
        if _blocks_ok(n, world):
            out[stem + ".w8"] = _slice_segments(w8, plan.segments or (n,), rank, world, dim)
            ws = params[stem + ".wscale"]
            out[stem + ".wscale"] = _slice_segments(
                ws, plan.segments or (n,), rank, world, dim, div=_FP8_BLOCK
            )
            _carry_oscale(params, out, stem, plan, rank, world)
            if dense is not None:
                out[stem] = _slice_segments(dense, plan.segments or (n,), rank, world, dim)
            return
        dense = _dequant(params, stem, cfg) if dense is None else dense

    if wq is not None and dense is None:
        # fp4 packs two values per byte along dim 1 and carries a per-16-column
        # scale; both slice cleanly on this model's shapes.
        segs = plan.segments or (wq.shape[dim] * (2 if dim == 1 else 1),)
        out[stem + ".wq"] = _slice_segments(wq, segs, rank, world, dim, div=2 if dim == 1 else 1)
        sc = params[stem + ".scale"]
        out[stem + ".scale"] = _slice_segments(sc, segs, rank, world, dim, div=16 if dim == 1 else 1)
        _carry_oscale(params, out, stem, plan, rank, world)
        return

    if dense is None:
        return
    segs = plan.segments or (dense.shape[dim],)
    out[stem] = _slice_segments(dense, segs, rank, world, dim)


def _carry_oscale(params, out, stem, plan, rank, world) -> None:
    """The per-output-row epilogue scale follows a column shard and is
    replicated for a row shard: scale * sum(partials) == sum(scale * partial)."""
    osc = params.get(stem + ".oscale")
    if osc is None:
        return
    if plan.kind == "row":
        out[stem + ".oscale"] = osc
    else:
        out[stem + ".oscale"] = _slice_segments(osc, plan.segments, rank, world, 0)


if __name__ == "__main__":  # runnable check: segments stay whole, shapes halve
    from .config import qwen36_27b

    cfg = qwen36_27b()
    c4 = tp_config(cfg, 4)
    assert c4.num_kv_heads == 1 and c4.num_attention_heads == 6, c4
    assert c4.intermediate_size == 4352 and c4.linear_num_value_heads == 12
    assert c4.hidden_size == cfg.hidden_size  # activation width is replicated

    try:
        tp_config(cfg, 8)
    except ValueError as e:
        assert "num_kv_heads" in str(e), e
    else:
        raise AssertionError("tp=8 must be refused: num_kv_heads=4")

    # a fused [q|k|v] row shard must hand each rank its own q, k and v slice,
    # never the tail of one segment plus the head of the next.
    d = cfg.head_dim
    q, k = cfg.num_attention_heads * d * 2, cfg.num_kv_heads * d
    w = torch.arange(q + 2 * k, dtype=torch.float32).unsqueeze(1)
    plan = plan_for("layers.0.qkv", cfg)
    got = torch.cat([_slice_segments(w, plan.segments, r, 4, 0) for r in range(4)])
    assert sorted(got.flatten().tolist()) == list(range(q + 2 * k)), "lost rows"
    r0 = _slice_segments(w, plan.segments, 0, 4, 0).flatten().tolist()
    assert r0[: q // 4] == list(range(q // 4)), "rank 0 q segment"
    assert r0[q // 4] == q, "rank 0's k segment must start at the k boundary"
    print("tensor_parallel: cfg split + fused-segment shard OK")
