"""Tensor parallelism: shard the config, then shard the weights to match.

The model code is untouched by design. Every shape it reshapes with comes from
``cfg``, so dividing the head counts and ``intermediate_size`` by the world
size makes ``_full_attn`` / ``_gdn`` / ``_mlp`` run on the local shard with no
edit — and the KV and GDN state pools, built from the same cfg, shard with it.

Column-parallel (split output rows) needs no communication; row-parallel
(split input columns) leaves each rank holding a PARTIAL sum, so the three
row-parallel projections all-reduce before the residual add. All three go
through ``Model._add_via``, the only place in the model that changes.

Three rules taken from vLLM / SGLang / TensorRT-LLM rather than invented:

* **GQA with fewer KV heads than ranks replicates whole KV heads**, chosen at
  LOAD time (``rank // kv_replicas``), never at runtime. The attention kernel
  then never learns that replication happened. Nobody shards ``head_dim`` —
  it breaks RoPE pairing and the softmax denominator.
* **A quantization block must divide the shard, and that is a hard error**, not
  something to pad or requantize around. fp4 carries a scale per 16 input
  columns and packs two values per byte, so a row shard must be a multiple of
  32; an fp8 block scale covers 128.
* **Slice in the logical layout, swizzle afterwards.** ``Backend._served_fp4``
  twiddles the fp4 bytes lazily on first use, which is after this runs — keep
  it that way, a shard of twiddled bytes is not a twiddle of the shard.

Sharding runs BEFORE ``_fuse_projections``: fusing the already-local q/k/v
shards produces exactly the right layout, which is why only the two weights
the checkpoint itself stores concatenated need segment bookkeeping.

Capacity, for the 27B on 8 H20s — this model has 4 KV heads, so TP=8 stores
each KV head on two ranks and the cache twice. Concurrent requests at depth
8192: DP=8 826, TP=8 605, TP=4 x DP=2 **1042**. The duplication grows with
context while the weight saving is a fixed 23 GiB, so TP=8 loses further the
longer the prompt. vLLM's answer to that is decode context parallelism
(sharding KV along sequence); without it, TP=4 x DP=2 is this model's
configuration.

# ponytail: torch.distributed collectives CANNOT be captured in a CUDA graph
# (SGLang's capture table is explicit). TP therefore forfeits the decode graph,
# worth 7.9x at B=1, until Backend.all_reduce grows a pynccl-style path.
"""

from __future__ import annotations

from dataclasses import replace

import torch

__all__ = ["tp_config", "shard_params", "kv_replicas", "pad_vocab"]

#: keys split on dim 0 (output rows). Value is the segment list for a weight
#: the CHECKPOINT stores concatenated; () means one contiguous output range.
#: q/k/v and gate/up are absent because sharding runs before they are fused.
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

#: fp4 packs two values per byte along the input dim and carries a scale per
#: 16 input columns, so a row shard must clear 32 to slice both cleanly.
_FP4_ROW_ALIGN = 32
#: an fp8 block scale covers 128 rows and 128 columns.
_FP8_BLOCK = 128


def pad_vocab(vocab: int, world: int, to: int = 64) -> int:
    """Vocab rounded up so the TP divide is exact, the way every engine does
    it: pad first, trim ``logits[:, :vocab]`` after the gather."""
    step = to * world
    return -(-vocab // step) * step


def kv_replicas(total_kv_heads: int, world: int) -> tuple[int, int]:
    """``(kv_heads_per_rank, replicas)`` — SGLang/vLLM's rule verbatim.

    Fewer KV heads than ranks means each rank keeps ONE head and consecutive
    groups of ``replicas`` ranks hold the same one.
    """
    if world >= total_kv_heads:
        if world % total_kv_heads:
            raise ValueError(
                f"tp={world} must be a multiple of num_kv_heads={total_kv_heads} to replicate"
            )
        return 1, world // total_kv_heads
    if total_kv_heads % world:
        raise ValueError(f"tp={world} must divide num_kv_heads={total_kv_heads}")
    return total_kv_heads // world, 1


def tp_config(cfg, world: int):
    """``cfg`` with every sharded dimension divided by ``world``.

    ``hidden_size`` stays whole: it is the width of the replicated activation
    entering and leaving every layer.
    """
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
    """A shard boundary inside a quantization block is a hard error. Every
    production engine refuses here; none pads and none requantizes."""
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
    """Rank's share of each segment along ``dim``, re-concatenated.

    ``div`` is that dim's packing factor: 2 for fp4's two-per-byte columns, 16
    for a per-16-column fp4 scale, 128 for an fp8 block scale.
    """
    segs = segs or (t.shape[dim] * div,)
    parts, off = [], 0
    for seg in segs:
        n = seg // div
        step = n // world
        parts.append(t.narrow(dim, off + rank * step, step))
        off += n
    return torch.cat(parts, dim=dim).contiguous() if len(parts) > 1 else parts[0].contiguous()


def shard_params(params: dict, cfg, rank: int, world: int) -> dict:
    """``params`` for the FULL model -> this rank's shard.

    ``cfg`` is the FULL config, and ``params`` must be UNFUSED
    (``load_hf(fuse_projections=False)``): fusing afterwards concatenates the
    local q/k/v shards into exactly the layout the fused kernels want.
    """
    if world == 1:
        return params
    _, replicas = kv_replicas(cfg.num_kv_heads, world)
    out: dict[str, torch.Tensor] = {}
    for key, t in params.items():
        head, _, suf = key.rpartition(".")
        stem = head if suf in _QUANT_SUFFIX else key
        base = stem.rsplit(".", 1)[-1]
        if base in _ROW:
            kind, dim, segs = "row", 1, ()
        elif base in _COLUMN:
            kind, dim, segs = "col", 0, _segment_sizes(cfg, _COLUMN[base])
        else:
            out[key] = t  # norms, embed_tokens, biases: replicated
            continue
        # K and V replicate across the ranks that share a head; Q does not.
        r = rank // replicas if base in ("k_proj", "v_proj") else rank
        w = world // replicas if base in ("k_proj", "v_proj") else world
        if w == 1:
            out[key] = t
            continue
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

    # GQA replication is what makes tp=8 legal at all on 4 KV heads.
    assert kv_replicas(4, 4) == (1, 1)
    assert kv_replicas(4, 8) == (1, 2)
    assert kv_replicas(4, 2) == (2, 1)

    c8 = tp_config(cfg, 8)
    assert c8.num_kv_heads == 1 and c8.num_attention_heads == 3
    assert c8.hidden_size == cfg.hidden_size  # activation width is replicated

    # k/v must land on the same slice for both ranks of a replica pair.
    kv = torch.arange(cfg.num_kv_heads * cfg.head_dim, dtype=torch.float32).unsqueeze(1)
    p = {"layers.0.k_proj": kv}
    a = shard_params(p, cfg, 0, 8)["layers.0.k_proj"]
    b = shard_params(p, cfg, 1, 8)["layers.0.k_proj"]
    assert torch.equal(a, b), "ranks sharing a KV head must hold the same rows"
    c = shard_params(p, cfg, 2, 8)["layers.0.k_proj"]
    assert not torch.equal(a, c), "the next replica group must hold a different head"

    # a shard boundary inside a quantization block is refused, not padded.
    bad = {"layers.0.down_proj.w8": torch.zeros(64, 8 * 100)}
    try:
        shard_params(bad, cfg, 0, 8)
    except ValueError as e:
        assert "128-block" in str(e), e
    else:
        raise AssertionError("misaligned fp8 row shard must raise")

    assert pad_vocab(248320, 4) == 248320  # already a multiple of 64*4
    assert pad_vocab(1000, 8) == 1024
    print("tensor_parallel: gqa replication + alignment refusal + vocab pad OK")
