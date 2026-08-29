"""Load my quantized 27B via load_hf(keep_master=True) for 1 layer, and compare
the STE master (regenerated from served .wq/.scale/.oscale) against the ORIGINAL
bf16 weight. If load_hf reconstructs a weight far from the original, the
quantize->save->load round-trip is lossy/wrong (target-independent). If it
matches, the served path is faithful and the 220 is elsewhere."""
import json
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, "src")
sys.path.insert(0, "packages/tilerl-kernels/src")
from tilerl import config as config_mod
from tilerl import model as model_mod

fp4_dir = "/data00/home/chenkailun.c/models/Qwen3.8-27B-NVFP4"
bf16_dir = "/data00/home/chenkailun.c/models/Qwen3.8-27B"

cfg = config_mod.qwen38_27b()
m = model_mod.load_hf(cfg, fp4_dir, num_layers=2, keep_master=True)
print("loaded 2 layers with master", flush=True)

# in_proj_qkv of layer 0 (GDN): master is params['layers.0.in_proj_qkv'] (bf16)
key = "layers.0.in_proj_qkv"
master = m.params.get(key)
print("master present:", master is not None, "dtype", None if master is None else master.dtype, flush=True)

# original bf16
bidx = json.load(open(bf16_dir + "/model.safetensors.index.json"))["weight_map"]
hf = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
orig = load_file(bf16_dir + "/" + bidx[hf])[hf].float()
if master is not None:
    mf = master.float()
    print("shapes master", tuple(mf.shape), "orig", tuple(orig.shape), flush=True)
    rel = (mf - orig).abs().max().item() / (orig.abs().max().item() + 1e-9)
    print(f"master vs original bf16: rel={rel:.4e}", flush=True)
    print("orig rownorms[:3]", orig[:3].norm(dim=1).tolist(), flush=True)
    print("mast rownorms[:3]", mf[:3].norm(dim=1).tolist(), flush=True)

# Also: is embed_tokens served correctly (right key, right dtype)?
print("embed key present:", "embed_tokens" in m.params, flush=True)
et = m.params.get("embed_tokens")
if et is not None:
    print("embed dtype", et.dtype, "shape", tuple(et.shape), flush=True)
# lm_head served fp4?
print("lm_head.wq present:", "lm_head.wq" in m.params, "  lm_head (bf16) present:", "lm_head" in m.params, flush=True)
