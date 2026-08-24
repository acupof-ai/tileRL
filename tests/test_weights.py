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


def test_nvfp4_modelopt_load(tmp_path):
    """ModelOpt NVFP4/FP8-block checkpoint (Qwen3.6 format): MLP linears load
    from weight_packed + f8 weight_scale + global scale, GDN linears from f8
    weight + inverse scale_inv, both dequantized to bf16 equal to a pure-torch
    reference computed in the test; model.visual.*/mtp.* tensors are ignored."""
    # Dimensions must be multiples of the quant blocks: 16 (NVFP4) and 128
    # (FP8 block), so the tiny config is widened accordingly.
    cfg = replace(
        tiny(),
        hidden_size=128,
        intermediate_size=128,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
    )
    model = build_random(cfg, seed=7)
    gen = torch.Generator().manual_seed(11)

    def e2m1_decode(nib):
        # OCP/MX e2m1 in formula form (independent of reference._E2M1_LUT):
        # e=0 -> {0, .5} subnormal; e>=1 -> 2^(e-1) * {1, 1.5}.
        e = ((nib >> 1) & 3).float()
        m = (nib & 1).float()
        mag = torch.where(e == 0, 0.5 * m, torch.pow(2.0, e - 1.0) * (1.0 + 0.5 * m))
        return (1.0 - 2.0 * (nib >> 3).float()) * mag

    def ref_nvfp4(packed, scale, gscale):
        n, k2 = packed.shape
        vals = torch.stack([e2m1_decode(packed & 0xF), e2m1_decode(packed >> 4)], dim=-1)
        vals = vals.reshape(n, k2 * 2)
        s = scale.float().repeat_interleave(16, dim=-1)
        return (vals * s * gscale.float()).to(torch.bfloat16)

    def ref_fp8_block(w, si, block=128):
        s = si.float().repeat_interleave(block, -1).repeat_interleave(block, -2)
        return (w.float() / s).to(torch.bfloat16)

    tensors, expected = {}, {}
    for key, t in model.params.items():
        hf = _hf_name(key)
        if key.endswith((".gate_proj", ".up_proj", ".down_proj")):
            n, k = t.shape
            packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
            scale = (torch.rand(n, k // 16, generator=gen) * 0.1 + 0.05).to(torch.float8_e4m3fn)
            gscale = torch.randn(1, generator=gen) * 0.1
            stem = hf.removesuffix(".weight")
            tensors[stem + ".weight_packed"] = packed
            tensors[stem + ".weight_scale"] = scale
            tensors[stem + ".weight_global_scale"] = gscale
            # activation quant: present in the checkpoint, read-and-ignored
            tensors[stem + ".input_global_scale"] = torch.randn(1, generator=gen) * 0.1
            expected[key] = ref_nvfp4(packed, scale, gscale)
        elif key.endswith((".in_proj_qkv", ".in_proj_z", ".out_proj")):
            n, k = t.shape
            w = (torch.randn(n, k, generator=gen) * 0.1).to(torch.float8_e4m3fn)
            si = (torch.rand(n // 128, k // 128, generator=gen) + 0.5).to(torch.bfloat16)
            tensors[hf] = w
            tensors[hf.removesuffix(".weight") + ".weight_scale_inv"] = si
            expected[key] = ref_fp8_block(w, si)
        else:
            tensors[hf] = t
            expected[key] = t
    # vision tower / MTP head: present in the checkpoint, ignored by load_hf
    tensors["model.visual.vision_tower.patch_embed.weight"] = torch.randn(
        4, 4, dtype=torch.bfloat16
    )
    tensors["mtp.layers.0.enorm.weight"] = torch.randn(128, dtype=torch.bfloat16)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    layer_types = [
        "full_attention" if i in cfg.full_attn_layer_set else "linear_attention"
        for i in range(cfg.num_layers)
    ]
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_6",
                "text_config": {
                    "hidden_size": cfg.hidden_size,
                    "intermediate_size": cfg.intermediate_size,
                    "num_hidden_layers": cfg.num_layers,
                    "num_attention_heads": cfg.num_attention_heads,
                    "num_key_value_heads": cfg.num_kv_heads,
                    "head_dim": cfg.head_dim,
                    "vocab_size": cfg.vocab_size,
                    "rms_norm_eps": cfg.rms_eps,
                    "tie_word_embeddings": cfg.tie_word_embeddings,
                    "attn_output_gate": cfg.full_attn_gated,
                    "layer_types": layer_types,
                },
                "quantization_config": {
                    "quant_method": "nvfp4",
                    "weight_block_size": [1, 16],
                },
            }
        )
    )
    loaded = load_hf(cfg, str(tmp_path))
    assert set(loaded.params) == set(param_specs(cfg))
    for key, exp in expected.items():
        assert torch.equal(loaded.params[key], exp), f"param {key} dequant mismatch"
