#!/usr/bin/env python3
"""ThinkingCap NVFP4 vs the bf16 base — dequant correctness on a real reference.

ThinkingCap is a fine-tune of Qwen3.8-27B (same dims), so dequantizing its
NVFP4 must recover a weight that points in the SAME direction as the base
bf16 tensor and carries the same norm:

  cosine   flat dot / norms — near 1 if the nibble layout, scales and global
           scale are applied right; a scrambled/transposed layout reads ~0.
  normr    ||w_tc||_F / ||w_base||_F — ~1 for a fine-tune; using the global
           scale's reciprocal the wrong way blows this up or shrinks it by
           orders of magnitude.

Samples full-attn attention, GDN in_proj_qkv/in_proj_z (the two with separate
global scales), an MLP projection, and every MTP tensor (aux shard vs the
base's mtp.* bf16). Pure CPU, reads only the sampled tensors' bytes.

  python scripts/check_thinkingcap.py \\
      ~/models/ThinkingCap-Qwen3.8-27B-NVFP4 ~/models/Qwen3.8-27B
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "src")
sys.path.insert(0, "packages/tilerl-kernels/src")
from tilerl_kernels.reference import dequant_nvfp4  # noqa: E402

from tilerl.model import _param_key_for  # noqa: E402

COS_MIN = 0.90   # NVFP4 fine-tune vs bf16 base: direction must survive
NR_LO, NR_HI = 0.5, 2.0


def header(p: Path) -> dict:
    with open(p, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def index(d: Path) -> dict:
    return json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]


def metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    # f64 chunked accumulation: a single f32 reduction over ~9e7 elements
    # pushes dot/(nx*ny) off [-1,1] by ~2% even when a == b (observed 1.017).
    import math
    a, b = a.float().flatten(), b.float().flatten()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    num = sa = sb = 0.0
    for i in range(0, n, 1 << 20):
        ca = a[i:i + (1 << 20)].double()
        cb = b[i:i + (1 << 20)].double()
        num += float((ca * cb).sum())
        sa += float((ca * ca).sum())
        sb += float((cb * cb).sum())
    cos = num / (math.sqrt(sa * sb) + 1e-30)
    nr = math.sqrt(sa / (sb + 1e-30))
    return cos, nr


def open_tensor(d: Path, wm: dict, name: str):
    with safe_open(str(d / wm[name]), "pt", device="cpu") as f:
        return f.get_tensor(name)


def main(tc: str, base: str) -> int:
    tcd, bd = Path(tc), Path(base)
    wm, wmb = index(tcd), index(bd)

    # --- header enumeration: every served key resolves, shard dtypes sane ---
    served = packed = 0
    unmapped_bad = []
    for name in wm:
        if name.endswith((".weight_scale", ".weight_global_scale")):
            continue
        cand = (name.removesuffix(".weight_packed") + ".weight"
                if name.endswith(".weight_packed") else name)
        key = _param_key_for(cand.removeprefix("model.language_model."))
        if key is None:
            if not (name.startswith(("model.visual", "mtp"))
                    or name.endswith(("A_log", "dt_bias"))):
                unmapped_bad.append(name)
            continue
        served += 1
        packed += int(name.endswith(".weight_packed"))
    print(f"served={served} packed={packed} bad_unmapped={unmapped_bad[:5]}")
    assert not unmapped_bad

    # base tensors live under model.language_model.<stem>.weight; mtp is mtp.*
    def base_name(stem: str) -> str:
        return stem if stem.startswith("mtp.") else f"model.language_model.{stem}.weight"

    # --- quantized trunk samples ---
    qstems = [
        "model.language_model.layers.3.self_attn.q_proj",   # full-attn
        "model.language_model.layers.7.self_attn.o_proj",
        "model.language_model.layers.3.mlp.gate_proj",      # MLP
        "model.language_model.layers.0.linear_attn.in_proj_qkv",  # GDN, own gs
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.4.linear_attn.in_proj_qkv",
    ]
    fails = []
    neg_checked = False
    for si, stem in enumerate(qstems):
        w = open_tensor(tcd, wm, stem + ".weight_packed")
        ws = open_tensor(tcd, wm, stem + ".weight_scale")
        gs = open_tensor(tcd, wm, stem + ".weight_global_scale")
        k = w.shape[1] * 2
        assert ws.dtype == torch.float8_e4m3fn
        assert ws.shape[-1] == k // 16, (stem, ws.shape, k)
        assert gs.numel() == 1
        dq = dequant_nvfp4(w, ws, gs, global_divide=True).float()
        bn = base_name(stem.removeprefix("model.language_model."))
        bf = open_tensor(bd, wmb, bn).float()
        cos, nr = metrics(dq, bf)
        tag = stem.split(".")[-1] + f".L{stem.split('.')[2]}"
        print(f"{tag:>18}: cos={cos:.4f} normr={nr:.4f} max|w|={float(dq.abs().max()):.3f}")
        if not (cos >= COS_MIN and NR_LO <= nr <= NR_HI):
            fails.append((tag, cos, nr))
        # Negative control on the first layer: flipping global_divide (multiply
        # by gs instead of dividing) must push normr out of the gate band.
        if si == 0:
            wrong = dequant_nvfp4(w, ws, gs, global_divide=False).float()
            _, nr_wrong = metrics(wrong, bf)
            neg_checked = not (NR_LO <= nr_wrong <= NR_HI)
            print(f"{tag:>18}: NEG global_divide flipped normr={nr_wrong:.4e} "
                  f"(must be outside [{NR_LO},{NR_HI}])")

    if not neg_checked:
        print("FAILURES: negative control did not go red")
        return 1

    # --- MTP: aux bf16 vs base mtp bf16, direct (no dequant) ---
    aux = sorted(k for k in wm if wm[k] == "model-base-aux.safetensors")
    assert len(aux) == 15, len(aux)
    for name in aux:
        a = open_tensor(tcd, wm, name)
        b = open_tensor(bd, wmb, name)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        cos, nr = metrics(a, b)
        tag = "mtp." + name.removeprefix("mtp.")
        print(f"{tag:>42}: cos={cos:.4f} normr={nr:.4f}")
        if not (cos >= COS_MIN and NR_LO <= nr <= NR_HI):
            fails.append((name, cos, nr))

    if fails:
        print("FAILURES:", fails)
        return 1
    print("THINKINGCAP_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
