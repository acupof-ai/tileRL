"""Hermetic tests for checkpoint loading: HF roundtrip, num_layers truncation,
fp4 on-load quantization, and the loud-failure paths (missing tensor,
layer_types mismatch)."""

from __future__ import annotations

import json
import os

os.environ.setdefault("TILERL_TARGET", "cpu")

from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from tilerl.config import tiny
from tilerl.model import build_random, fp4_param_keys, load_hf, param_specs

#: param suffix -> HF suffix (reverse of model._LAYER_SUFFIXES)
_SIMPLE = {
    "input_norm": "input_layernorm.weight",
    "post_attn_norm": "post_attention_layernorm.weight",
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "in_proj_qkv": "linear_attn.in_proj_qkv.weight",
    "in_proj_z": "linear_attn.in_proj_z.weight",
    "in_proj_b": "linear_attn.in_proj_b.weight",
    "in_proj_a": "linear_attn.in_proj_a.weight",
    "conv1d": "linear_attn.conv1d.weight",
    "dt_bias": "linear_attn.dt_bias",
    "a_log": "linear_attn.A_log",
    "gdn_norm": "linear_attn.norm.weight",
    "out_proj": "linear_attn.out_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


def _hf_name(key: str) -> str:
    if key == "embed_tokens":
        return "model.language_model.embed_tokens.weight"
    if key == "final_norm":
        return "model.language_model.norm.weight"
    if key == "lm_head":
        return "lm_head.weight"
    _, layer, suffix = key.split(".", 2)
    return f"model.language_model.layers.{layer}.{_SIMPLE[suffix]}"


def _write_checkpoint(d, cfg, params) -> None:
    """Write a tiny Qwen-format safetensors checkpoint + config.json."""
    n = cfg.num_layers
    tensors = {
        _hf_name(k): t.contiguous() for k, t in params.items() if not k.endswith((".wq", ".scale"))
    }
    save_file(tensors, str(d / "model.safetensors"))
    layer_types = [
        "full_attention" if i in cfg.full_attn_layer_set else "linear_attention" for i in range(n)
    ]
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "num_hidden_layers": n,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_kv_heads,
                "head_dim": cfg.head_dim,
                "vocab_size": cfg.vocab_size,
                "rms_norm_eps": cfg.rms_eps,
                "tie_word_embeddings": cfg.tie_word_embeddings,
                "attn_output_gate": cfg.full_attn_gated,
                "layer_types": layer_types,
            }
        )
    )


def test_hf_roundtrip(tmp_path):
    cfg = tiny()
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)
    loaded = load_hf(cfg, str(tmp_path))
    for key, t in model.params.items():
        assert key in loaded.params, f"missing param {key} after load"
        assert torch.equal(loaded.params[key], t), f"param {key} changed in load"


def test_num_layers_truncation(tmp_path):
    """load_hf(num_layers=1) loads only layer 0 from a full checkpoint: the
    truncated model has exactly layer 0, and layers.1.* is skipped."""
    cfg = tiny()
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)  # full 2-layer checkpoint
    loaded = load_hf(cfg, str(tmp_path), num_layers=1)
    assert loaded.cfg.num_layers == 1
    assert set(loaded.params) == set(param_specs(loaded.cfg))
    assert "layers.0.input_norm" in loaded.params
    assert not any(k.startswith("layers.1.") for k in loaded.params)
    # out-of-range truncation is rejected, not silently clamped
    with pytest.raises(ValueError):
        load_hf(cfg, str(tmp_path), num_layers=99)


def test_missing_tensor_raises(tmp_path):
    cfg = tiny()
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)
    # drop one tensor and rewrite the shard
    tensors = {
        _hf_name(k): t
        for k, t in model.params.items()
        if not k.endswith((".wq", ".scale")) and k != "layers.1.in_proj_qkv"
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="missing"):
        load_hf(cfg, str(tmp_path))


def test_layer_types_mismatch_raises(tmp_path):
    cfg = tiny()
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)
    cfg_json = json.loads((tmp_path / "config.json").read_text())
    cfg_json["layer_types"] = ["full_attention", "full_attention"]
    (tmp_path / "config.json").write_text(json.dumps(cfg_json))
    with pytest.raises(RuntimeError, match="layer_types"):
        load_hf(cfg, str(tmp_path))


def test_fp4_on_load_and_forward(tmp_path):
    """fp4=True packs linears on load; the packed model forwards through the
    fp4 linear path end to end."""
    import numpy as np

    from tilerl.engine import SamplingParams, build_engine
    from tilerl.testing import RefBackend

    cfg = tiny()
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)
    loaded = load_hf(replace(cfg, fp4=True), str(tmp_path))
    for key in fp4_param_keys(loaded.cfg):
        assert f"{key}.wq" in loaded.params and f"{key}.scale" in loaded.params
    engine = build_engine(
        loaded.cfg,
        loaded,
        RefBackend(),
        num_blocks=8,
        num_slots=4,
        max_batch=4,
        max_total_tokens=512,
    )
    rid = engine.submit(
        list(range(1, 17)), SamplingParams(temperature=0.0, max_new_tokens=2, seed=0)
    )
    done = {}
    for _ in range(64):
        engine.step()
        done.update(engine.poll())
        if rid in done:
            break
    assert rid in done, "fp4 model did not generate"
    engine.shutdown()
