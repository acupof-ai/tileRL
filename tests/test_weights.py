"""Checkpoint loading: HF roundtrip, truncation, every quant format, and the
loud-failure paths."""

from __future__ import annotations

import itertools
import json
import os

os.environ.setdefault("TILERL_TARGET", "cpu")

from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file
from tilerl_kernels import reference

from tilerl.config import tiny
from tilerl.model import build_random, fp4_param_keys, load_hf, param_specs, save_hf

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


#: zero-centered norm keys: load_hf folds +1 back in
_ZC = ("input_norm", "post_attn_norm", "q_norm", "k_norm")


_AWQ_LINEARS = (
    ".q_proj", ".k_proj", ".v_proj", ".o_proj", ".in_proj_qkv", ".in_proj_z", ".out_proj",
    ".gate_proj", ".up_proj", ".down_proj",
)


def _is_zc(key):
    return key == "final_norm" or key.endswith(_ZC)


def _load_norm(key, t):
    return (t.float() + 1.0).to(t.dtype) if _is_zc(key) else t


def _e2m1_decode(nib):
    # e2m1 in formula form, independent of reference._E2M1_LUT
    e = ((nib >> 1) & 3).float()
    m = (nib & 1).float()
    mag = torch.where(e == 0, 0.5 * m, torch.pow(2.0, e - 1.0) * (1.0 + 0.5 * m))
    return (1.0 - 2.0 * (nib >> 3).float()) * mag


def _ref_nvfp4(packed, scale, gs):
    n, k2 = packed.shape
    vals = torch.stack([_e2m1_decode(packed & 0xF), _e2m1_decode(packed >> 4)], dim=-1)
    return (vals.reshape(n, k2 * 2) * scale.float().repeat_interleave(16, dim=-1) * gs).to(
        torch.bfloat16
    )


def _hf_name(key: str) -> str:
    if key == "embed_tokens":
        return "model.language_model.embed_tokens.weight"
    if key == "final_norm":
        return "model.language_model.norm.weight"
    if key == "lm_head":
        return "lm_head.weight"
    _, layer, suffix = key.split(".", 2)
    return f"model.language_model.layers.{layer}.{_SIMPLE[suffix]}"


def _write_config(d, cfg, model_type="qwen3_5_text", nested=False, **extra) -> None:
    text = {
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
        "layer_types": [
            "full_attention" if i in cfg.full_attn_layer_set else "linear_attention"
            for i in range(cfg.num_layers)
        ],
    }
    body = {"model_type": model_type, **extra}
    body.update({"text_config": text} if nested else text)
    (d / "config.json").write_text(json.dumps(body))


def _write_checkpoint(d, cfg, params) -> None:
    """A tiny Qwen-format safetensors checkpoint + config.json (norms un-folded on disk)."""
    tensors = {
        _hf_name(k): ((t.float() - 1.0).to(t.dtype) if _is_zc(k) else t).contiguous()
        for k, t in params.items()
        if not k.endswith((".wq", ".scale"))
    }
    save_file(tensors, str(d / "model.safetensors"))
    _write_config(d, cfg)


def test_hf_roundtrip(tmp_path):
    cfg = tiny()
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)
    loaded = load_hf(cfg, str(tmp_path))
    for key, t in model.params.items():
        assert key in loaded.params, f"missing param {key} after load"
        assert torch.equal(loaded.params[key], t), f"param {key} changed in load"


def test_num_layers_truncation(tmp_path):
    cfg = tiny()
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)
    loaded = load_hf(cfg, str(tmp_path), num_layers=1)
    assert loaded.cfg.num_layers == 1
    assert set(loaded.params) == set(param_specs(loaded.cfg))
    assert "layers.0.input_norm" in loaded.params
    assert not any(k.startswith("layers.1.") for k in loaded.params)
    with pytest.raises(ValueError):
        load_hf(cfg, str(tmp_path), num_layers=99)


def test_missing_tensor_raises(tmp_path):
    cfg = tiny()
    model = build_random(cfg, seed=7)
    _write_checkpoint(tmp_path, cfg, model.params)
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
    """kd != vd would silently mis-shape the GDN state pool (shaped on vd)."""
    with pytest.raises(ValueError, match="linear_key_head_dim"):
        replace(tiny(), linear_key_head_dim=32)


def test_rope_and_tie_guards_raise(tmp_path):
    """Every rope/tie field the checkpoint can contradict refuses to load; the
    last case is the 27B's own (config says tied, the checkpoint ships lm_head)."""
    cfg = tiny()
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
    # multimodal layout: the top-level tie_word_embeddings wins over text_config
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": base, "tie_word_embeddings": False})
    )
    with pytest.raises(RuntimeError, match="tie_word_embeddings"):
        load_hf(cfg, str(tmp_path))
    untied = replace(cfg, tie_word_embeddings=False)
    ckpt = tmp_path / "untied"
    ckpt.mkdir()
    _write_checkpoint(ckpt, untied, build_random(untied, seed=7).params)
    lying = json.loads((ckpt / "config.json").read_text()) | {"tie_word_embeddings": True}
    (ckpt / "config.json").write_text(json.dumps(lying))
    with pytest.raises(RuntimeError, match="lm_head"):
        load_hf(cfg, str(ckpt))


def test_fp4_on_load_and_forward(tmp_path):
    """fp4=True packs linears on load, keeps no bf16 master, and still generates."""
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
    """load_hf(save_hf(m)) re-quantizes nothing: .wq/.scale/.oscale bit-identical,
    same greedy tokens. Untied so lm_head is packed too."""
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


def test_sm70_chunks_cover_every_row_exactly_once():
    """The sm70 GEMV ladder chunks M into 1/2/4/8/32-row launches. A slicing bug
    is invisible where it runs — only M that does not divide 32 exposes it, and the
    sm70 branch never executes on the CPU target the parity tests use — so gate the
    arithmetic directly: chunks must tile [0, M) with no gap, no overlap, and each
    chunk's rung must be the smallest that holds its rows."""
    from tilerl_kernels.backend import _sm70_chunks

    for m in (1, 2, 3, 5, 8, 9, 16, 32, 33, 40, 64, 100, 512):
        chunks = _sm70_chunks(m)
        assert [o for o, _, _ in chunks] == list(
            itertools.accumulate([r for _, r, _ in chunks][:-1], initial=0)
        ), f"M={m}: offsets do not follow the row counts"
        assert sum(r for _, r, _ in chunks) == m, f"M={m}: chunks do not cover M"
        for _, r, rung in chunks:
            assert rung >= r, f"M={m}: rung {rung} cannot hold {r} rows"
            assert rung in (1, 2, 4, 8, 32), f"M={m}: {rung} is not a ladder rung"
            smaller = [w for w in (1, 2, 4, 8, 32) if w >= r]
            assert rung == min(smaller), f"M={m}: {r} rows took rung {rung}, not {min(smaller)}"
    # The whole point of the per-chunk rung: 40 rows is 32+8, not 32+32.
    assert [rung for _, _, rung in _sm70_chunks(40)] == [32, 8]


def test_the_mma8_row_cap_is_never_above_the_plans_padded_M():
    """Both mma8 branches feed the kernel exactly ``_MX`` rows, and the fp8 one takes
    them from an ``x2`` the plan already padded to ``Mp = _snap_mma_tile(M, 128)``.
    Mp is 16 for EVERY M in 2.._MX, so `_pad2d(x2, _MX, Kp)` asked to SHRINK 16 rows
    to 8 -- which main got away with because F.pad crops on a negative pad, and which
    the guard in `_pad2d` correctly refuses. Measured on sm90: 1191 calls at M=8 in a
    single B=8 tick, so this was every speculative config and dense B=8.

    Gate the arithmetic, not the kernel: neither mma8 branch can execute on the CPU
    target where the parity tests live, and the fp4 twin does NOT pre-pad x2, so the
    two branches disagree about what x2's row count is when they slice it."""
    from tilerl_kernels.backend import _MX, _snap_mma_tile

    for m in range(2, _MX + 1):
        mp = _snap_mma_tile(m, 128)
        assert mp >= _MX, (
            f"M={m}: the plan pads x2 to {mp} rows but the mma8 kernel wants {_MX}; "
            "if Mp could fall below _MX the slice would silently drop real rows"
        )
        # The real rows are the first m of Mp, so taking _MX of them keeps all of them.
        assert m <= _MX <= mp, f"M={m}: {m} <= {_MX} <= {mp} is what makes the slice exact"


def test_the_mma8_N_rung_can_be_narrower_than_the_plans_and_the_drop_is_exact():
    """The mma8 branches re-target wq/scale from the plan's ``Np`` to ``Np32``, and the
    two rungs DO NOT ORDER: Np = _round_up(N, bN) with bN from the plan's N tile (64 on
    the fp4 decode arm) overshoots Np32 = _round_up(N, 32) whenever bN's rounding jumps
    past 32. sm90 reported it as `_pad2d: (64, 256) exceeds the target [32, 256]` -- the
    fp4 twin of the fp8 row bug, and my published claim that "the fp4 branch does not
    pre-pad, which is why only fp8 breaks" was wrong: it pre-pads on the N axis.

    Two things to gate, and the first is why the second is needed: that Np > Np32 really
    happens (or `_fit_rows` is dead code), and that dropping those rows equals never
    padding them -- rows between N and either rung are zero, so the two agree elementwise.
    """
    import torch
    from tilerl_kernels.backend import _fit_rows, _pad2d, _round_up

    over = [(n, _round_up(n, 64), _round_up(n, 32))
            for n in (24, 40, 96, 130) if _round_up(n, 64) > _round_up(n, 32)]
    assert over, "no N has Np > Np32, so the mma8 re-target can never over-ask"

    for n, np_, np32 in over:
        plane = torch.arange(n * 8, dtype=torch.float32).reshape(n, 8)
        # what the branch does: pad to the plan's rung, then re-target to the narrow one
        got = _fit_rows(_pad2d(plane, np_, 8), np32, 8)
        assert got.shape == (np32, 8), f"N={n}: {tuple(got.shape)} is not [{np32}, 8]"
        assert torch.equal(got, _pad2d(plane, np32, 8)), (
            f"N={n}: dropping rows off the [{np_}, 8] plane differs from padding the "
            f"[{n}, 8] one to {np32} -- the dropped rows were not all zero pad"
        )


def test_fp4_save_widens_f16_scale_plane(tmp_path):
    """A backend may serve the block scales narrowed (sm70 does, at f16); the NVFP4
    format's are f32, so save_hf must widen them or the written checkpoint is
    unreadable by anything that trusts the format. The dtype is all save_hf reacts
    to, so narrowing the planes directly keeps this hermetic."""
    cfg = replace(tiny(), fp4=True, tie_word_embeddings=False)
    model = build_random(cfg, seed=7)
    for key in fp4_param_keys(cfg):
        model.params[key + ".scale"] = model.params[key + ".scale"].to(torch.float16)
    save_hf(model, tmp_path / "ckpt")
    loaded = load_hf(cfg, str(tmp_path / "ckpt"))
    for key in fp4_param_keys(cfg):
        sc = loaded.params[key + ".scale"]
        assert sc.dtype == torch.float32, f"{key}.scale saved as {sc.dtype}, not the format's f32"
        assert torch.equal(sc, model.params[key + ".scale"].float())  # widening is exact


def test_fused_projections_parity(tmp_path):
    """fuse_projections concats same-input fp4 projections; logits match unfused."""
    import numpy as np
    from tilerl_kernels.backend import get_backend

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
        tl = get_backend()
        y2 = fused.forward(batch, positions, kv(fused, tl), tl).cpu()
        y3 = unfused.forward(batch, positions, kv(unfused, tl), tl).cpu()
    assert torch.allclose(y0, y1, rtol=1e-2, atol=1e-2), (y0 - y1).abs().max()
    # The fusion, on the served backend. NOT torch.equal: one fused gemm and two
    # separate ones reduce in a different order, which is bit-identical on cpu and
    # 9.5e-06 apart on metal. A broken fusion is off by orders of magnitude.
    assert torch.allclose(y2, y3, rtol=0, atol=1e-4), (y2 - y3).abs().max()
    if not tl.target.startswith("cuda"):
        # sm90 sends these 32 rows to w4a8, whose e4m3 activations sit 19.8% off the
        # f32 reference at this M -- no tolerance there separates a broken fusion from
        # the format floor. test_ops_parity covers that path against its own reference.
        assert torch.allclose(y0, y2, rtol=1e-2, atol=1e-2), (y0 - y2).abs().max()


_NVFP4 = {"quantization_config": {"quant_method": "nvfp4", "weight_block_size": [1, 16]}}
#: tiny widened to multiples of the quant blocks (16 for NVFP4, 128 for FP8 block)
_WIDE_CFG = replace(
    tiny(), hidden_size=128, intermediate_size=128, linear_key_head_dim=64, linear_value_head_dim=64
)


def test_nvfp4_modelopt_load(tmp_path):
    """ModelOpt checkpoint: MLP linears from weight_packed + f8 weight_scale +
    global scale (stored as its reciprocal), GDN linears kept native from f8
    weight + per-128-block scale_inv (multiplied, despite the name);
    model.visual.*/mtp.* ignored."""
    cfg = _WIDE_CFG
    model = build_random(cfg, seed=7)
    gen = torch.Generator().manual_seed(11)

    tensors, expected, expected_fp8 = {}, {}, {}
    for key, t in model.params.items():
        hf = _hf_name(key)
        if key.endswith((".gate_proj", ".up_proj", ".down_proj")):
            n, k = t.shape
            packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
            scale = (torch.rand(n, k // 16, generator=gen) * 0.1 + 0.05).to(torch.float8_e4m3fn)
            gscale = torch.rand(1, generator=gen) * 1000 + 100
            stem = hf.removesuffix(".weight")
            tensors[stem + ".weight_packed"] = packed
            tensors[stem + ".weight_scale"] = scale
            tensors[stem + ".weight_global_scale"] = gscale
            tensors[stem + ".input_global_scale"] = torch.randn(1, generator=gen) * 0.1
            expected[key] = _ref_nvfp4(packed, scale, 1.0 / gscale.float())
        elif key.endswith((".in_proj_qkv", ".in_proj_z", ".out_proj")):
            n, k = t.shape
            w = (torch.randn(n, k, generator=gen) * 0.1).to(torch.float8_e4m3fn)
            si = (torch.rand(n // 128, k // 128, generator=gen) * 0.01 + 0.001).to(torch.bfloat16)
            tensors[hf] = w
            tensors[hf.removesuffix(".weight") + ".weight_scale_inv"] = si
            s = si.float().repeat_interleave(128, -1).repeat_interleave(128, -2)
            expected[key] = (w.float() * s).to(torch.bfloat16)
            expected_fp8[key] = (w, si.float())
        else:
            tensors[hf] = t
            expected[key] = _load_norm(key, t)
    tensors["model.visual.vision_tower.patch_embed.weight"] = torch.randn(
        4, 4, dtype=torch.bfloat16
    )
    tensors["mtp.layers.0.enorm.weight"] = torch.randn(128, dtype=torch.bfloat16)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    _write_config(tmp_path, cfg, "qwen3_6", nested=True, **_NVFP4)
    loaded = load_hf(cfg, str(tmp_path), keep_master=True)
    assert set(param_specs(cfg)) <= set(loaded.params)
    extras = {k for k in loaded.params if k not in param_specs(cfg)}
    assert extras == {f"{k}.{s}" for k in expected_fp8 for s in ("w8", "wscale")}
    for key, exp in expected.items():
        assert torch.equal(loaded.params[key], exp), f"param {key} dequant mismatch"
    for key, (w, ws) in expected_fp8.items():
        # uint8 view: torch 2.5.1 has no `equal_cpu` for Float8_e4m3fn, which failed
        # three loader tests on the V100 pod while passing on a newer torch. fp8 is a
        # bit pattern, so the view compares exactly what the dtype compare would.
        assert torch.equal(loaded.params[key + ".w8"].view(torch.uint8),
                           w.view(torch.uint8)), f"param {key}.w8 mismatch"
        assert torch.equal(loaded.params[key + ".wscale"], ws), f"param {key}.wscale mismatch"


@pytest.mark.parametrize("fp4", [False, True])
def test_nvfp4_official_load(tmp_path, fp4):
    """Official NVFP4 checkpoint: MLP linears from u8 weight + f8 weight_scale +
    scalar weight_scale_2; GDN linears per-tensor FP8, full-attn linears
    per-channel [N,1]. FP8 stays 8-bit (.w8, ones .wscale, per-row .oscale);
    with ``fp4`` the MLP bytes are served as-is (.wq byte-identical, block 16)."""
    cfg = _WIDE_CFG
    model = build_random(cfg, seed=7)
    gen = torch.Generator().manual_seed(11)

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
            tensors[stem + ".input_scale"] = torch.randn(1, generator=gen) * 0.1
            expected[key] = _ref_nvfp4(packed, scale, gscale.float())
            packed_fp4[key] = packed
        elif key.endswith(
            (".in_proj_qkv", ".in_proj_z", ".out_proj", ".q_proj", ".k_proj", ".v_proj", ".o_proj")
        ):
            n, k = t.shape
            w = (torch.randn(n, k, generator=gen) * 0.1).to(torch.float8_e4m3fn)
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
            expected[key] = _load_norm(key, t)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    _write_config(tmp_path, cfg, "qwen3_6", nested=True, **_NVFP4)
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
        # uint8 view: torch 2.5.1 has no `equal_cpu` for Float8_e4m3fn, which failed
        # three loader tests on the V100 pod while passing on a newer torch. fp8 is a
        # bit pattern, so the view compares exactly what the dtype compare would.
        assert torch.equal(loaded.params[key + ".w8"].view(torch.uint8),
                           w.view(torch.uint8)), f"param {key}.w8 mismatch"
        assert torch.equal(loaded.params[key + ".wscale"], ws), f"param {key}.wscale mismatch"
        assert torch.equal(loaded.params[key + ".oscale"], os_), f"param {key}.oscale mismatch"
        # the served 8-bit path and the bf16 master agree
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
        assert 6 * scale.max() <= 448, f"{key}.scale saturates the e4m3 dequant"
    lean = load_hf(replace(cfg, fp4=True), str(tmp_path))
    assert not ((set(packed_fp4) | set(expected_fp8)) & set(lean.params))  # no masters


def test_awq_load(tmp_path):
    """AutoAWQ GEMM checkpoint: qweight + per-group scales + qzeros -> bf16;
    in_proj_b/a stay bf16 (out_features=2 < 8)."""
    cfg = replace(_WIDE_CFG, head_dim=32)  # every in_features a multiple of the group
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
        if key.endswith(_AWQ_LINEARS):
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
            expected[key] = _load_norm(key, t)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    _write_config(
        tmp_path, cfg,
        quantization_config={"quant_method": "awq", "group_size": group_size, "bits": 4},
    )
    loaded = load_hf(cfg, str(tmp_path))
    assert set(loaded.params) == set(param_specs(cfg))
    for key, exp in expected.items():
        assert torch.equal(loaded.params[key], exp), f"param {key} dequant mismatch"


def test_mlx_affine_load(tmp_path):
    """MLX affine-4bit checkpoint (language_model. prefix, uint32-packed int4 +
    bf16 scales/biases, group 64); embed_tokens and conv1d stay bf16."""
    cfg = replace(tiny(), linear_key_head_dim=64, linear_value_head_dim=64)  # group 64
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
        hf = _hf_name(key).removeprefix("model.")
        base = hf.removesuffix(".weight")
        if key.endswith(_AWQ_LINEARS + (".in_proj_b", ".in_proj_a")):
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
            expected[key] = _load_norm(key, t)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    _write_config(tmp_path, cfg, quantization={"group_size": group_size, "bits": 4})
    loaded = load_hf(cfg, str(tmp_path))
    assert set(loaded.params) == set(param_specs(cfg))
    for key, exp in expected.items():
        assert torch.equal(loaded.params[key], exp), f"param {key} dequant mismatch"
