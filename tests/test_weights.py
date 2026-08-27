"""Hermetic tests for checkpoint loading: HF roundtrip, num_layers truncation,
fp4 on-load quantization, and the loud-failure paths (missing tensor,
layer_types mismatch, rope/tie fields the checkpoint contradicts)."""

from __future__ import annotations

import json
import os

os.environ.setdefault("TILERL_TARGET", "cpu")

from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from tilerl.config import tiny
from tilerl.model import build_random, fp4_param_keys, load_hf, param_specs, save_hf
from tilerl.ops import reference

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


def test_linear_head_dim_mismatch_raises():
    """kd != vd would run with a silently mis-shaped GDN state K axis (the
    pool shapes both trailing axes on the value dim). Every Qwen3.x ships
    kd == vd; arle hard-gates it too."""
    with pytest.raises(ValueError, match="linear_key_head_dim"):
        replace(tiny(), linear_key_head_dim=32)  # tiny: vd=16


def test_rope_and_tie_guards_raise(tmp_path):
    """Every rope/tie field the checkpoint can contradict must refuse to load.
    Each case below is served silently wrong without the guard — the last one
    is the 27B's own: config says tied, the checkpoint ships lm_head."""
    cfg = tiny()  # rope_theta 1e7, rotary_dim 16 of head_dim 16, tied
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)
    base = json.loads((tmp_path / "config.json").read_text())
    for patch, match in (
        ({"rope_theta": 1e6}, "rope_theta"),
        ({"rope_parameters": {"rope_theta": 1e6}}, "rope_theta"),  # Qwen3.5 spelling
        ({"partial_rotary_factor": 0.5}, "partial_rotary_factor"),
        ({"rope_parameters": {"partial_rotary_factor": 0.5}}, "partial_rotary_factor"),
        ({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "rope_scaling"),
        ({"rope_parameters": {"rope_type": "yarn", "factor": 4.0}}, "rope_scaling"),
        ({"tie_word_embeddings": False}, "tie_word_embeddings"),
    ):
        (tmp_path / "config.json").write_text(json.dumps(base | patch))
        with pytest.raises(RuntimeError, match=match):
            load_hf(cfg, str(tmp_path))
    # Multimodal layout: the top-level tie_word_embeddings wins over the
    # text_config copy, so a text_config-only lookup would no-op here.
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": base, "tie_word_embeddings": False})
    )
    with pytest.raises(RuntimeError, match="tie_word_embeddings"):
        load_hf(cfg, str(tmp_path))
    # Derived: cfg AND config.json both say tied, the tensors disagree.
    untied = replace(cfg, tie_word_embeddings=False)
    ckpt = tmp_path / "untied"
    ckpt.mkdir()
    _write_checkpoint(ckpt, untied, build_random(untied, seed=7).params)
    lying = json.loads((ckpt / "config.json").read_text()) | {"tie_word_embeddings": True}
    (ckpt / "config.json").write_text(json.dumps(lying))
    with pytest.raises(RuntimeError, match="lm_head"):
        load_hf(cfg, str(ckpt))


def test_fp4_on_load_and_forward(tmp_path):
    """fp4=True packs linears on load and (the default) keeps no bf16 master;
    the packed model still forwards through the fp4 linear path end to end."""
    import numpy as np

    from tilerl.engine import SamplingParams, build_engine
    from tilerl.testing import RefBackend

    cfg = tiny()
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)
    loaded = load_hf(replace(cfg, fp4=True), str(tmp_path))
    for key in fp4_param_keys(loaded.cfg):
        assert f"{key}.wq" in loaded.params and f"{key}.scale" in loaded.params
        assert key not in loaded.params, f"{key}: serving must ship no bf16 master"
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


def test_fp4_save_load_roundtrip(tmp_path):
    """save_hf writes a master-free fp4 model as the bytes it serves, so
    load_hf(save_hf(m)) re-quantizes nothing: .wq/.scale/.oscale come back
    bit-identical and the reloaded model generates the same greedy tokens.
    Untied, because an untied fp4 model serves lm_head packed too."""
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.testing import RefBackend

    cfg = replace(tiny(), fp4=True, tie_word_embeddings=False)
    model = build_random(cfg, seed=7)
    save_hf(model, tmp_path / "ckpt")
    loaded = load_hf(cfg, str(tmp_path / "ckpt"))

    for key in sorted(fp4_param_keys(cfg)):
        assert key not in loaded.params, f"{key}: reload must ship no bf16 master"
        for suffix in (".wq", ".scale", ".oscale"):
            a, b = model.params[key + suffix], loaded.params[key + suffix]
            assert a.dtype == b.dtype and torch.equal(a, b), f"{key}{suffix} is not bit-identical"

    def greedy(m) -> list[int]:
        engine = build_engine(
            m.cfg, m, RefBackend(), num_blocks=8, num_slots=4, max_batch=4, max_total_tokens=512
        )
        rid = engine.submit(
            list(range(1, 17)), SamplingParams(temperature=0.0, max_new_tokens=4, seed=0)
        )
        done: dict = {}
        for _ in range(64):
            engine.step()
            done.update(engine.poll())
            if rid in done:
                break
        engine.shutdown()
        return done[rid]

    assert greedy(loaded) == greedy(model)


def test_fused_projections_parity(tmp_path):
    """fuse_projections concats same-input fp4 projections into one GEMV; the
    fused model's logits match the unfused model's (concat + slice is lossless)."""
    import numpy as np

    from tilerl.ops.backend import get_backend
    from tilerl.testing import RefBackend
    from tilerl.train import _training_kv

    cfg = tiny()
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)
    unfused = load_hf(replace(cfg, fp4=True), str(tmp_path))
    fused = load_hf(replace(cfg, fp4=True), str(tmp_path), fuse_projections=True)
    assert "layers.0.gate_up.wq" in fused.params
    batch = np.random.default_rng(3).integers(3, cfg.vocab_size, size=(2, 16)).astype(np.int64)
    positions = np.arange(16, dtype=np.int64)
    backend = RefBackend()
    with torch.no_grad():
        kv = lambda m, be: _training_kv(m, 2, 16, device=be.device)
        y0 = unfused.forward(batch, positions, kv(unfused, backend), backend)
        y1 = fused.forward(batch, positions, kv(fused, backend), backend)
        # AGENTS.md parity gate: fused path through the TileLang kernels. .cpu()
        # because the tilelang backend returns device tensors (metal) and the
        # reference stays on CPU — same idiom as test_ops_parity._assert_close.
        tl = get_backend()
        y2 = fused.forward(batch, positions, kv(fused, tl), tl).cpu()
    assert torch.allclose(y0, y1, rtol=1e-2, atol=1e-2), (y0 - y1).abs().max()
    assert torch.allclose(y0, y2, rtol=1e-2, atol=1e-2), (y0 - y2).abs().max()


def test_nvfp4_modelopt_load(tmp_path):
    """ModelOpt NVFP4/FP8-block checkpoint (Qwen3.6 format): MLP linears load
    from weight_packed + f8 weight_scale + global scale (the stored global is
    its reciprocal, divided), GDN linears from f8 weight + per-128-block
    scale_inv (multiplied, despite the name). The GDN linears are kept native:
    the bf16 dequant is the recording-only master, .w8 the e4m3 weight and
    .wscale the f32 per-128-block scale (equal to a pure-torch reference
    computed in the test); model.visual.*/mtp.* tensors are ignored."""
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
        # ModelOpt stores the global scale's reciprocal: divide by it
        # (agent-infer quant_format.rs ScaleApply::Divide).
        n, k2 = packed.shape
        vals = torch.stack([e2m1_decode(packed & 0xF), e2m1_decode(packed >> 4)], dim=-1)
        vals = vals.reshape(n, k2 * 2)
        s = scale.float().repeat_interleave(16, dim=-1)
        return (vals * s / gscale.float()).to(torch.bfloat16)

    def ref_fp8_block(w, si, block=128):
        # The stored "scale_inv" is the per-block scale itself (multiplied,
        # despite the name — agent-infer quant_format.rs ScaleApply::Multiply).
        s = si.float().repeat_interleave(block, -1).repeat_interleave(block, -2)
        return (w.float() * s).to(torch.bfloat16)

    tensors, expected, expected_fp8 = {}, {}, {}
    for key, t in model.params.items():
        hf = _hf_name(key)
        if key.endswith((".gate_proj", ".up_proj", ".down_proj")):
            n, k = t.shape
            packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
            scale = (torch.rand(n, k // 16, generator=gen) * 0.1 + 0.05).to(torch.float8_e4m3fn)
            # Stored global is the reciprocal: large and strictly positive.
            gscale = torch.rand(1, generator=gen) * 1000 + 100
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
            si = (torch.rand(n // 128, k // 128, generator=gen) * 0.01 + 0.001).to(torch.bfloat16)
            tensors[hf] = w
            tensors[hf.removesuffix(".weight") + ".weight_scale_inv"] = si
            expected[key] = ref_fp8_block(w, si)
            expected_fp8[key] = (w, si.float())
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
    loaded = load_hf(cfg, str(tmp_path), keep_master=True)  # the dequants under test
    assert set(param_specs(cfg)) <= set(loaded.params)
    # the only extra params are the native-fp8 siblings
    extras = {k for k in loaded.params if k not in param_specs(cfg)}
    assert extras == {f"{k}.{s}" for k in expected_fp8 for s in ("w8", "wscale")}
    for key, exp in expected.items():
        assert torch.equal(loaded.params[key], exp), f"param {key} dequant mismatch"
    for key, (w, ws) in expected_fp8.items():
        assert torch.equal(loaded.params[key + ".w8"], w), f"param {key}.w8 mismatch"
        assert torch.equal(loaded.params[key + ".wscale"], ws), f"param {key}.wscale mismatch"


@pytest.mark.parametrize("fp4", [False, True])
def test_nvfp4_official_load(tmp_path, fp4):
    """Official NVFP4 checkpoint (nvidia/Qwen3.6-27B-NVFP4 naming): MLP
    linears load from u8 ``weight`` (e2m1 nibbles) + f8 ``weight_scale`` +
    scalar ``weight_scale_2``; GDN linears from f8 ``weight`` + scalar
    ``weight_scale`` (per-tensor FP8) and full-attn linears from the same with
    a per-channel [N,1] ``weight_scale``. Both FP8 forms stay 8-bit: .w8 the
    e4m3 weight, .wscale a ones grid, .oscale the per-output-row scale, and the
    bf16 master carries the full magnitude (oscale included). With ``fp4`` the
    MLP linears are served as the checkpoint's own bytes — .wq byte-identical,
    .scale at the checkpoint's block 16 — instead of being dequantized and
    re-packed. ``input_scale`` siblings are read-and-ignored."""
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
        # OCP/MX e2m1 in formula form (independent of reference._E2M1_LUT).
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

    tensors, expected, expected_fp8, packed_fp4 = {}, {}, {}, {}
    for key, t in model.params.items():
        hf = _hf_name(key)
        stem = hf.removesuffix(".weight")
        if key.endswith((".gate_proj", ".up_proj", ".down_proj")):
            n, k = t.shape
            packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
            scale = (torch.rand(n, k // 16, generator=gen) * 0.1 + 0.05).to(torch.float8_e4m3fn)
            gscale = torch.randn(1, generator=gen) * 0.1
            tensors[hf] = packed
            tensors[stem + ".weight_scale"] = scale
            tensors[stem + ".weight_scale_2"] = gscale
            # activation quant: present in the checkpoint, read-and-ignored
            tensors[stem + ".input_scale"] = torch.randn(1, generator=gen) * 0.1
            expected[key] = ref_nvfp4(packed, scale, gscale)
            packed_fp4[key] = packed
        elif key.endswith(
            (
                ".in_proj_qkv",
                ".in_proj_z",
                ".out_proj",
                ".q_proj",
                ".k_proj",
                ".v_proj",
                ".o_proj",
            )
        ):
            n, k = t.shape
            w = (torch.randn(n, k, generator=gen) * 0.1).to(torch.float8_e4m3fn)
            # full-attn linears carry a per-channel [N,1] scale, GDN a scalar
            per_channel = key.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj"))
            scale = (
                torch.rand(n, 1, generator=gen) * 0.1 + 0.5
                if per_channel
                else (torch.randn(1, generator=gen) * 0.1 + 0.5)
            )
            tensors[hf] = w
            tensors[stem + ".weight_scale"] = scale
            tensors[stem + ".input_scale"] = torch.randn(1, generator=gen) * 0.1
            oscale = scale.float().reshape(-1).expand(n).contiguous()
            expected[key] = (w.float() * oscale.reshape(-1, 1)).to(torch.bfloat16)
            expected_fp8[key] = (w, torch.ones(((n + 127) // 128), ((k + 127) // 128)), oscale)
        else:
            tensors[hf] = t
            expected[key] = t
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
                "quantization_config": {"quant_method": "nvfp4", "weight_block_size": [1, 16]},
            }
        )
    )
    loaded = load_hf(replace(cfg, fp4=fp4), str(tmp_path), keep_master=True)
    assert set(param_specs(cfg)) <= set(loaded.params)
    extras = {k for k in loaded.params if k not in param_specs(cfg)}
    fp8_extras = {f"{k}.{s}" for k in expected_fp8 for s in ("w8", "wscale", "oscale")}
    assert fp8_extras <= extras and (fp4 or extras == fp8_extras)
    for key, exp in expected.items():
        got = loaded.params[key]
        if fp4 and key in packed_fp4:  # master regenerated from the served bytes
            assert torch.allclose(got.float(), exp.float(), rtol=1e-2, atol=1e-2), key
        else:
            assert torch.equal(got, exp), f"param {key} dequant mismatch"
    for key, (w, ws, os_) in expected_fp8.items():
        assert torch.equal(loaded.params[key + ".w8"], w), f"param {key}.w8 mismatch"
        assert torch.equal(loaded.params[key + ".wscale"], ws), f"param {key}.wscale mismatch"
        assert torch.equal(loaded.params[key + ".oscale"], os_), f"param {key}.oscale mismatch"
        # the 8-bit served path and the bf16 master must agree (the tape only
        # ever sees the master, the kernel only ever sees w8/wscale/oscale)
        x = torch.randn(2, w.shape[1], generator=gen)
        served = reference.linear_fp8(x, w, ws, os_)
        assert torch.allclose(served, x @ loaded.params[key].float().t(), rtol=1e-2, atol=1e-2)
    if not fp4:
        return
    for key, packed in packed_fp4.items():
        n, k = param_specs(cfg)[key]
        scale = loaded.params[key + ".scale"]
        assert torch.equal(loaded.params[key + ".wq"], packed), f"{key}.wq is not the ckpt bytes"
        assert scale.shape == (n, k // 16), f"{key}.scale lost the checkpoint block"
        # the w4a8 kernel dequantizes into e4m3 (max 448): 6*scale must fit
        assert 6 * scale.max() <= 448, f"{key}.scale saturates the e4m3 dequant"
    lean = load_hf(replace(cfg, fp4=True), str(tmp_path))  # keep_master defaults off
    assert not ((set(packed_fp4) | set(expected_fp8)) & set(lean.params))  # no masters


def test_awq_load(tmp_path):
    """AWQ-int4 checkpoint (autoawq GEMM convention): linears load from
    ``qweight`` (int32, 8 int4 per int32 for 8 consecutive output features)
    + per-group ``scales`` + ``qzeros``, dequantized to bf16 equal to a
    pure-torch reference computed in the test; group_size comes from
    ``quantization_config``. in_proj_b/a stay bf16 (out_features=2 < 8)."""
    # head_dim=32 so o_proj's in_features (hq*head_dim) is a multiple of the
    # AWQ group 128; hidden/intermediate 128 for the same reason.
    cfg = replace(
        tiny(),
        hidden_size=128,
        intermediate_size=128,
        head_dim=32,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
    )
    model = build_random(cfg, seed=7)
    gen = torch.Generator().manual_seed(11)
    group_size = 128

    def ref_awq(qweight, scales, qzeros):
        k, n8 = qweight.shape
        shifts = torch.arange(8, dtype=torch.int64) * 4
        q = ((qweight.long().unsqueeze(-1) >> shifts) & 0xF).float().reshape(k, n8 * 8)
        z = ((qzeros.long().unsqueeze(-1) >> shifts) & 0xF).float()
        z = z.reshape(k // group_size, n8 * 8).repeat_interleave(group_size, dim=0)
        return ((q - z) * scales.float()).t().to(torch.bfloat16)

    tensors, expected = {}, {}
    for key, t in model.params.items():
        hf = _hf_name(key)
        stem = hf.removesuffix(".weight")
        if key.endswith(
            (
                ".q_proj",
                ".k_proj",
                ".v_proj",
                ".o_proj",
                ".in_proj_qkv",
                ".in_proj_z",
                ".out_proj",
                ".gate_proj",
                ".up_proj",
                ".down_proj",
            )
        ):
            out_f, in_f = t.shape
            q = torch.randint(0, 16, (in_f, out_f), generator=gen, dtype=torch.int64)
            z = torch.randint(0, 16, (in_f // group_size, out_f), generator=gen, dtype=torch.int64)
            scales = (torch.rand(in_f // group_size, out_f, generator=gen) * 0.1 + 0.01).to(
                torch.bfloat16
            )
            shifts = (torch.arange(out_f) % 8 * 4).to(torch.int64)
            qweight = (
                (q << shifts.view(1, -1)).reshape(in_f, out_f // 8, 8).sum(dim=-1).to(torch.int32)
            )
            qzeros = (
                (z << shifts.view(1, -1))
                .reshape(in_f // group_size, out_f // 8, 8)
                .sum(dim=-1)
                .to(torch.int32)
            )
            tensors[stem + ".qweight"] = qweight
            tensors[stem + ".scales"] = scales
            tensors[stem + ".qzeros"] = qzeros
            expected[key] = ref_awq(qweight, scales, qzeros)
        else:
            tensors[hf] = t
            expected[key] = t
    save_file(tensors, str(tmp_path / "model.safetensors"))
    layer_types = [
        "full_attention" if i in cfg.full_attn_layer_set else "linear_attention"
        for i in range(cfg.num_layers)
    ]
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
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
                "quantization_config": {"quant_method": "awq", "group_size": group_size, "bits": 4},
            }
        )
    )
    loaded = load_hf(cfg, str(tmp_path))
    assert set(loaded.params) == set(param_specs(cfg))
    for key, exp in expected.items():
        assert torch.equal(loaded.params[key], exp), f"param {key} dequant mismatch"


def test_mlx_affine_load(tmp_path):
    """MLX affine-4bit checkpoint (``language_model.`` prefix, uint32-packed
    int4 weights + bf16 scales/biases, group 64): linears dequantize to
    bf16 equal to a pure-torch reference computed in the test; embed_tokens
    and conv1d stay bf16 (MLX does not quantize them)."""
    # linear_*_head_dim=64 so out_proj/in_proj_z in_features (64/128) are
    # multiples of the MLX group 64 (tiny's default 32 is not).
    cfg = replace(tiny(), linear_key_head_dim=64, linear_value_head_dim=64)
    model = build_random(cfg, seed=7)
    gen = torch.Generator().manual_seed(11)
    group_size = 64

    def ref_mlx(w, scales, biases):
        out_f, k8 = w.shape
        shifts = torch.arange(8, dtype=torch.int64) * 4
        q = ((w.long().unsqueeze(-1) >> shifts) & 0xF).float().reshape(out_f, k8 * 8)
        s = scales.float().repeat_interleave(group_size, dim=-1)
        b = biases.float().repeat_interleave(group_size, dim=-1)
        return (s * q + b).to(torch.bfloat16)

    tensors, expected = {}, {}
    for key, t in model.params.items():
        hf = _hf_name(key).removeprefix("model.")  # -> language_model.*
        base = hf.removesuffix(".weight")
        if key.endswith(
            (
                ".q_proj",
                ".k_proj",
                ".v_proj",
                ".o_proj",
                ".in_proj_qkv",
                ".in_proj_z",
                ".in_proj_b",
                ".in_proj_a",
                ".out_proj",
                ".gate_proj",
                ".up_proj",
                ".down_proj",
            )
        ):
            out_f, in_f = t.shape
            q = torch.randint(0, 16, (out_f, in_f), generator=gen, dtype=torch.int64)
            scales = (torch.randn(out_f, in_f // group_size, generator=gen) * 0.1).to(
                torch.bfloat16
            )
            biases = (torch.randn(out_f, in_f // group_size, generator=gen) * 0.01).to(
                torch.bfloat16
            )
            shifts = (torch.arange(in_f) % 8 * 4).to(torch.int64)
            packed = (
                (q << shifts.view(1, -1)).reshape(out_f, in_f // 8, 8).sum(dim=-1).to(torch.uint32)
            )
            tensors[hf] = packed
            tensors[base + ".scales"] = scales
            tensors[base + ".biases"] = biases
            expected[key] = ref_mlx(packed, scales, biases)
        else:
            tensors[hf] = t
            expected[key] = t
    save_file(tensors, str(tmp_path / "model.safetensors"))
    layer_types = [
        "full_attention" if i in cfg.full_attn_layer_set else "linear_attention"
        for i in range(cfg.num_layers)
    ]
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
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
                "quantization": {"group_size": group_size, "bits": 4},
            }
        )
    )
    loaded = load_hf(cfg, str(tmp_path))
    assert set(loaded.params) == set(param_specs(cfg))
    for key, exp in expected.items():
        assert torch.equal(loaded.params[key], exp), f"param {key} dequant mismatch"
