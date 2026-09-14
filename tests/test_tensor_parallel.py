"""Single-process checks for the tensor-parallel sharding rules.

Moved verbatim from tensor_parallel.py's __main__ selfcheck so the module no
longer imports model (which imports tensor_parallel): the only back edge in the
model <-> tensor_parallel cycle.
"""

from __future__ import annotations

import pytest
import torch

from tilerl.config import qwen36_27b, tiny
from tilerl.model import Model, add_lora, build_random
from tilerl.tensor_parallel import (
    Mesh,
    is_sharded,
    kv_replicas,
    pad_vocab,
    shard_dim,
    shard_params,
    tp_config,
    zigzag_key_positions,
    zigzag_positions,
)


def test_kv_replication_table_and_head_split():
    cfg = qwen36_27b()
    assert kv_replicas(4, 4) == (1, 1)
    assert kv_replicas(4, 8) == (1, 2)
    assert kv_replicas(4, 2) == (2, 1)
    c8 = tp_config(cfg, 8)
    assert c8.num_kv_heads == 1 and c8.num_attention_heads == 3
    assert c8.hidden_size == cfg.hidden_size


def test_shared_kv_rank_pair_holds_the_same_head():
    cfg = qwen36_27b()
    kv = torch.arange(cfg.num_kv_heads * cfg.head_dim, dtype=torch.float32).unsqueeze(1)
    p = {"layers.0.k_proj": kv}
    a = shard_params(p, cfg, 0, 8)["layers.0.k_proj"]
    b = shard_params(p, cfg, 1, 8)["layers.0.k_proj"]
    assert torch.equal(a, b), "ranks sharing a KV head must hold the same rows"
    c = shard_params(p, cfg, 2, 8)["layers.0.k_proj"]
    assert not torch.equal(a, c), "the next replica group must hold a different head"


def test_misaligned_fp8_row_shard_is_refused():
    cfg = qwen36_27b()
    bad = {"layers.0.down_proj.w8": torch.zeros(64, 8 * 100)}
    with pytest.raises(ValueError, match="128-block"):
        shard_params(bad, cfg, 0, 8)


def test_vocab_padding():
    assert pad_vocab(248320, 4) == 248320
    assert pad_vocab(1000, 8) == 1024


def test_is_sharded_agrees_with_shard_params_for_every_param():
    # LoRA pairs split on one side only; the answer must match what shard_params
    # did for EVERY param, and shard_dim must name the reduced axis.
    cfg = tiny()
    one = build_random(cfg, seed=0, keep_master=True)
    add_lora(one, rank=4)
    local = Model(tp_config(cfg, 2),
                  shard_params(build_random(cfg, seed=0, keep_master=True).params,
                               cfg, 0, 2),
                  materialized=True)
    add_lora(local, rank=4)
    n = 0
    for k, full in one.params.items():
        if k not in local.params:
            continue
        n += 1
        small = local.params[k].shape
        assert is_sharded(k) == (full.shape != small), (
            f"{k}: is_sharded={is_sharded(k)} but {tuple(full.shape)} -> {tuple(small)}")
        axes = [d for d in range(len(full.shape)) if full.shape[d] != small[d]]
        assert shard_dim(k) == (axes[0] if axes else None), (
            f"{k}: shard_dim={shard_dim(k)} but {tuple(full.shape)} -> {tuple(small)}")
    assert n >= 59, f"only {n} params compared"


def test_mesh_partitions_the_world_without_overlap():
    m0 = Mesh(dp=2, tp=4, rank=0)
    assert m0.world == 8
    seen_tp, seen_dp = [], []
    for r in range(8):
        m = Mesh(dp=2, tp=4, rank=r)
        assert m.dp_rank * 4 + m.tp_rank == r, r
        seen_tp.append(tuple(m.tp_group()))
        seen_dp.append(tuple(m.dp_group()))
        assert r in m.tp_group() and r in m.dp_group()
        assert set(m.tp_group()) & set(m.dp_group()) == {r}
    assert sorted(set(seen_tp)) == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert len(set(seen_dp)) == 4 and all(len(g) == 2 for g in set(seen_dp))
    assert sorted(r for g in set(seen_dp) for r in g) == list(range(8))
    for bad in (dict(tp=0), dict(tp=2, rank=2), dict(tp=2, rank=-1)):
        with pytest.raises(ValueError):
            Mesh(**bad)


def test_cp_groups_and_zigzag_are_contiguous_and_balanced():
    for r in range(8):
        m = Mesh(dp=1, tp=4, cp=2, rank=r)
        assert m.cp_group() == [r - m.cp_rank, r - m.cp_rank + 1], r
        assert set(m.cp_group()) & set(m.tp_group()) == {r}
    for cp in (2, 4):
        pos = [zigzag_positions(32, cp, r) for r in range(cp)]
        assert sorted(p.item() for c in pos for p in c) == list(range(32))
        work = [int((p[:, None] >= torch.arange(32)[None, :]).sum()) for p in pos]
        assert len(set(work)) == 1, f"cp={cp} zigzag is unbalanced: {work}"
        assert torch.equal(zigzag_key_positions(32, cp), torch.cat(pos))
    with pytest.raises(ValueError):
        zigzag_positions(6, 4, 0)
