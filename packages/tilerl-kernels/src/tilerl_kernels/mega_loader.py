"""Load indexed tensors one at a time onto sm70, retaining checkpoint precision."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from . import reference


def load_sm70(cfg, source, backend):
    from tilerl.model import (
        Model,
        _fuse_projections,
        _native_fp4,
        _pack_fp4_bounded,
        _param_key_for,
        _validate_hf_config,
        fp4_param_keys,
        param_specs,
    )
    root = Path(source)
    _validate_hf_config(cfg, json.loads((root / "config.json").read_text()), source)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    specs, packed_keys, params = param_specs(cfg), fp4_param_keys(cfg), {}

    def read(name):
        with safe_open(str(root / index[name]), framework="pt", device="cpu") as file:
            return file.get_tensor(name)

    for name in sorted(index):
        if name.endswith(".weight_packed"):
            stem = name.removesuffix(".weight_packed")
            key = _param_key_for(stem + ".weight")
            if key not in specs:
                continue
            packed = read(name)
            wq, scale, oscale = _native_fp4(packed, read(stem + ".weight_scale"),
                                           read(stem + ".weight_global_scale"), divide=True)
            value = wq.to(backend.device)
            value.copy_(reference.twiddle_fp4_f16(value))
            value._tl_layout = "tw-f16"
            params[key + ".wq"] = value
            params[key + ".scale"] = scale.to(backend.device, torch.float16)
            params[key + ".oscale"] = oscale.to(backend.device)
        elif name.endswith(".weight") and name.removesuffix(".weight") + ".weight_scale" in index:
            stem = name.removesuffix(".weight")
            key = _param_key_for(name)
            if key not in specs:
                continue
            value = read(name)
            assert value.dtype == torch.float8_e4m3fn, (name, value.dtype)
            n, k = value.shape
            scale = read(stem + ".weight_scale").float().reshape(-1).expand(n).contiguous()
            params[key + ".w8"] = value.to(backend.device)
            params[key + ".wscale"] = torch.ones((n+127)//128, (k+127)//128, device=backend.device)
            params[key + ".oscale"] = scale.to(backend.device)
        else:
            key = _param_key_for(name)
            if key not in specs:
                continue
            value = read(name)
            if tuple(value.shape) != specs[key] and key.endswith(".conv1d"):
                value = value.reshape(specs[key])
            assert tuple(value.shape) == specs[key], (name, value.shape, specs[key])
            if key == "final_norm" or key.endswith(("input_norm", "post_attn_norm", "q_norm", "k_norm")):
                value = (value.float() + 1).to(value.dtype)
            if key in packed_keys:
                wq, scale = _pack_fp4_bounded(value)
                scale, oscale = reference.renorm_fp4_scale(scale)
                value = wq.to(backend.device)
                value.copy_(reference.twiddle_fp4_f16(value))
                value._tl_layout = "tw-f16"
                params[key + ".wq"] = value
                params[key + ".scale"] = scale.to(backend.device, torch.float16)
                params[key + ".oscale"] = oscale.to(backend.device)
            else:
                dtype = torch.float16 if key == "embed_tokens" else torch.float32
                params[key] = value.to(backend.device, dtype)
        if key in specs:
            print(f"loaded {key}", flush=True)
    missing = [k for k in specs if not any(k+s in params for s in ("", ".wq", ".w8"))]
    if missing:
        raise ValueError(f"Missing checkpoint tensors: {missing}")
    _fuse_projections(cfg, params)
    for name, value in params.items():
        if name.endswith(".wq"):
            value._tl_layout = "tw-f16"
    return Model(cfg, params, materialized=True)
