"""Hermetic tests for the draft-head loader: the per-format RMSNorm convention
and the layer-index convention. Both failure modes are SILENT in the loader
(a scaled norm, or a wrong inferred depth), which is why they get a gate."""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
import torch
from safetensors.torch import load_file, save_file

from tilerl.config import tiny
from tilerl.model import build_random
from tilerl.spec import load_draft


def _draft_file(tmp_path, cfg, *, dspark: bool, norm_value: float = 0.25):
    """One-layer draft head on disk, in either checkpoint flavor."""
    hd = cfg.hidden_size
    q_rows = cfg.num_attention_heads * cfg.head_dim * (2 if cfg.full_attn_gated else 1)
    kv_rows = cfg.num_kv_heads * cfg.head_dim
    t = {
        "fc.weight": torch.zeros(hd, 2 * hd),
        "norm.weight": torch.full((hd,), norm_value),
        ("hidden_norm.weight" if dspark else "pre_fc_norm_hidden.weight"): torch.full(
            (hd,), norm_value
        ),
        "layers.0.input_layernorm.weight": torch.full((hd,), norm_value),
        "layers.0.post_attention_layernorm.weight": torch.full((hd,), norm_value),
        "layers.0.self_attn.q_proj.weight": torch.zeros(q_rows, hd),
        "layers.0.self_attn.k_proj.weight": torch.zeros(kv_rows, hd),
        "layers.0.self_attn.v_proj.weight": torch.zeros(kv_rows, hd),
        "layers.0.self_attn.o_proj.weight": torch.zeros(
            hd, cfg.num_attention_heads * cfg.head_dim
        ),
        "layers.0.self_attn.q_norm.weight": torch.ones(cfg.head_dim),
        "layers.0.self_attn.k_norm.weight": torch.ones(cfg.head_dim),
        "layers.0.mlp.gate_proj.weight": torch.zeros(cfg.intermediate_size, hd),
        "layers.0.mlp.up_proj.weight": torch.zeros(cfg.intermediate_size, hd),
        "layers.0.mlp.down_proj.weight": torch.zeros(hd, cfg.intermediate_size),
    }
    p = tmp_path / ("dspark.safetensors" if dspark else "nextn.safetensors")
    save_file(t, str(p))
    return p


def test_norm_fold_is_per_format(tmp_path):
    """The +1 fold belongs to Qwen NextN only.

    NextN norms are zero-centered (y = x*(1+w)); DSpark's are plain w*x —
    agent-infer loads them with load_vec_any and only q/k_norm with
    load_vec_minus_one (dspark.rs:580, 726). Folding a DSpark head scales every
    norm silently, with none of the anti-correlated-logits signature that made
    the reverse bug findable.
    """
    cfg = tiny()
    trunk = build_random(cfg, seed=0)

    nextn = load_draft(trunk, _draft_file(tmp_path, cfg, dspark=False))
    assert torch.allclose(nextn.params["norm"], torch.full_like(nextn.params["norm"], 1.25)), (
        "NextN norms must carry the folded +1"
    )

    dspark = load_draft(trunk, _draft_file(tmp_path, cfg, dspark=True))
    assert torch.allclose(dspark.params["norm"], torch.full_like(dspark.params["norm"], 0.25)), (
        "DSpark norms are plain w*x — folding corrupts them"
    )
    assert "pre_fc_norm_hidden" in dspark.params  # hidden_norm maps onto it


def test_layer_indices_must_be_zero_based(tmp_path):
    """An absolute-index MTP convention must fail loudly, naming the cause.

    Inferring depth as max(index)+1 turns one layer numbered 7 into an 8-layer
    head: the engine then allocates an 8-layer draft KV plane and the forward
    dies on a missing layers.0, pointing at the wrong thing.
    """
    cfg = tiny()
    trunk = build_random(cfg, seed=0)
    t = load_file(str(_draft_file(tmp_path, cfg, dspark=False)))
    p = tmp_path / "absidx.safetensors"
    save_file({k.replace("layers.0.", "layers.7."): v for k, v in t.items()}, str(p))
    with pytest.raises(RuntimeError, match="indexed"):
        load_draft(trunk, p)
