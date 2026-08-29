import json
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, "packages/tilerl-kernels/src")
from tilerl_kernels.reference import dequant_fp4

d = "/data00/home/chenkailun.c/models/Qwen3.8-27B-NVFP4"
idx = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
print("lm_head shard:", idx.get("lm_head.wq"), flush=True)
t = load_file(d + "/" + idx["lm_head.wq"])
wq, sc, osc = t["lm_head.wq"], t["lm_head.scale"].float(), t["lm_head.oscale"].float()
print("shapes wq", tuple(wq.shape), "sc", tuple(sc.shape), "osc", tuple(osc.shape), flush=True)
print("blk = K//sc.shape[1] =", (wq.shape[1] * 2) // sc.shape[1], flush=True)
print("osc range", round(osc.min().item(), 6), round(osc.max().item(), 6), flush=True)
deq = dequant_fp4(wq, sc) * osc[:, None]
print("deq nan", bool(torch.isnan(deq).any()), "zerorows", int((deq.abs().sum(1) == 0).sum()), flush=True)
print("deq rownorm range", round(deq.norm(dim=1).min().item(), 4), round(deq.norm(dim=1).max().item(), 4), flush=True)

# Also check embed_tokens dtype/health
emb_sh = idx["model.language_model.embed_tokens.weight"]
et = load_file(d + "/" + emb_sh)["model.language_model.embed_tokens.weight"]
print("embed dtype", et.dtype, "shape", tuple(et.shape), "nan", bool(torch.isnan(et.float().float()).any()), flush=True)
