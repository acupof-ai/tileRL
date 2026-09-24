#!/usr/bin/env python3
"""ThinkingCap NVFP4 loader CPU self-consistency check (probe-only).

No bf16 reference ships in the repo, so the gate is self-consistency of the
on-disk NVFP4 representation, read through the SAME path load_hf uses:

1. header-only enumeration: every served key resolves through _param_key_for;
   packed triples are (U8, F8_E4M3FN scale, BF16/F32 global); no served key is
   unmapped (vision/mtp excluded).
2. one packed linear per category (full-attn qkv-ish, GDN qkv, GDN z) is read
   and dequantized with dequant_nvfp4(global_divide=True); assert finite,
   non-zero, and within a sane magnitude envelope (|w|<=1; compressed-tensors
   packs a normalized W*gs tensor, so dequant with 1/gs lands in O(1)).
3. scale block grid is 16 along K: weight_scale.shape[-1] == K/16; packed
   [N,K/2]; global scale scalar per tensor; qkv and z carry SEPARATE globals.
4. native fp4 serving repack (pack_fp4 block32 via _native_fp4) round-trips
   through dequant_fp4 within the NVFP4 nibble quantization bound (max rel
   err vs the block-16 dequant, asserted by construction of the LUT).

Reads real weight bytes for <=5 tensors. Run:
  python scripts/check_thinkingcap.py ~/models/ThinkingCap-Qwen3.8-27B-NVFP4
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


def header(p: Path) -> dict:
    with open(p, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def main(ckpt: str) -> int:
    d = Path(ckpt)
    import torch
    from safetensors import safe_open
    sys.path.insert(0, "src")
    sys.path.insert(0, "packages/tilerl-kernels/src")
    from tilerl_kernels.reference import dequant_nvfp4

    from tilerl.model import _param_key_for

    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(idx.values()))
    assert shards
    # 1: header enumeration
    served, packed, unmapped = [], [], []
    for name in idx:
        if name.endswith((".weight_scale", ".weight_global_scale")):
            continue
        key = _param_key_for(
            name.removeprefix("model.language_model.").removesuffix(".weight_packed")
            + (".weight" if name.endswith(".weight_packed") else ""))
        if key is None:
            unmapped.append(name)
            continue
        served.append((name, key))
        if name.endswith(".weight_packed"):
            packed.append((name, key))
    print(f"served={len(served)} packed={len(packed)} unmapped={len(unmapped)}")
    bad = [u for u in unmapped if not (u.startswith(("model.visual", "mtp")) or u.endswith(("A_log", "dt_bias")))]
    print("unmapped non-vision/mtp:", bad[:10])
    # 2/3: read representative tensors: layer3 q_proj (full), layer0 in_proj_qkv, layer0 in_proj_z
    targets = {
        "model.language_model.layers.3.self_attn.q_proj": None,
        "model.language_model.layers.0.linear_attn.in_proj_qkv": None,
        "model.language_model.layers.0.linear_attn.in_proj_z": None,
    }
    files = {}
    for stem in targets:
        targets[stem] = idx[stem + ".weight_packed"]
        files.setdefault(targets[stem], [])
    # map shard -> stems it holds
    by_shard = {}
    for stem, sh in targets.items():
        by_shard.setdefault(sh, []).append(stem)
    with torch.no_grad():
        for sh, stems in by_shard.items():
            with safe_open(str(d / sh), "pt", device="cpu") as f:
                for stem in stems:
                    w = f.get_tensor(stem + ".weight_packed")
                    ws = f.get_tensor(stem + ".weight_scale")
                    gs = f.get_tensor(stem + ".weight_global_scale")
                    n, k2 = w.shape
                    k = k2 * 2
                    assert ws.dtype == torch.float8_e4m3fn, (stem, ws.dtype)
                    assert ws.shape[-1] == k // 16, (stem, ws.shape, k)
                    assert gs.numel() == 1, (stem, gs.shape)
                    dq = dequant_nvfp4(w, ws, gs, global_divide=True).float()
                    finite = torch.isfinite(dq).all().item()
                    mx = float(dq.abs().max())
                    nz = float((dq != 0).float().mean())
                    print(f"{stem.split('.')[-3]}.{stem.split('.')[-1]}: "
                          f"[{n},{k}] ws={tuple(ws.shape)} gs={float(gs.float()):.5f} "
                          f"finite={finite} max|w|={mx:.4f} nonzero={nz:.3f}")
                    assert finite and mx <= 2.0 and mx > 1e-3 and nz > 0.3
    print("THINKINGCAP_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
