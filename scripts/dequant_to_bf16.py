"""Dequantize the NVFP4/FP8 checkpoint to a plain bf16 HF checkpoint so other
engines (sglang on Hopper has no w4a4 NVFP4 path) can run the SAME model on the
same card. NVFP4 linears: e2m1 * f8 block scale / global (ModelOpt reciprocal);
FP8 block linears: f8 * scale_inv. Everything else copied; quant siblings and
config.quantization_config dropped. CPU only, row-chunked (lm_head is 248320
rows). ~54 GB out.

  python scripts/dequant_to_bf16.py /data00/Qwen3.8-27B-NVFP4 /work/Qwen3.8-27B-bf16
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tilerl.ops.reference import dequant_fp8, dequant_nvfp4  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
SHARD = 5 * 2**30
_SKIP = (".weight_scale", ".weight_global_scale", ".weight_scale_inv", ".input_global_scale", ".input_scale")


shard, shard_bytes, n_shard, index = {}, 0, 0, {}


def flush():
    global shard, shard_bytes, n_shard
    if not shard:
        return
    name = f"model-{n_shard:05d}.safetensors"
    save_file(shard, str(dst / name), metadata={"format": "pt"})
    for k in shard:
        index[k] = name
    print(f"  wrote {name} ({shard_bytes / 2**30:.1f} GiB, {len(shard)} tensors)", flush=True)
    shard, shard_bytes, n_shard = {}, 0, n_shard + 1


with safe_open(str(src / "model.safetensors"), "pt", device="cpu") as f:
    keys = list(f.keys())
    for k in keys:
        if k.endswith(_SKIP):
            continue
        t = f.get_tensor(k)
        if k.endswith(".weight_packed"):
            stem = k.removesuffix(".weight_packed")
            sc, gs = f.get_tensor(stem + ".weight_scale"), f.get_tensor(stem + ".weight_global_scale")
            t = torch.cat([  # row-chunked: dequant_fp4's int64 temporaries are ~30 GiB on lm_head at once
                dequant_nvfp4(t[i : i + 2048], sc[i : i + 2048], gs, global_divide=True)
                for i in range(0, t.shape[0], 2048)
            ])
            k = stem + ".weight"
        elif k.endswith(".weight") and (k.removesuffix(".weight") + ".weight_scale_inv") in keys:
            t = dequant_fp8(t, f.get_tensor(k.removesuffix(".weight") + ".weight_scale_inv")).to(torch.bfloat16)
        shard[k] = t.contiguous()
        shard_bytes += t.numel() * t.element_size()
        if shard_bytes >= SHARD:
            flush()
flush()
(dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}, indent=1))
cfg = json.loads((src / "config.json").read_text())
cfg.pop("quantization_config", None)
(cfg.get("text_config") or {}).pop("quantization_config", None)
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "preprocessor_config.json"):
    if (src / name).exists():
        shutil.copy(src / name, dst / name)
print("done:", len(index), "tensors")
