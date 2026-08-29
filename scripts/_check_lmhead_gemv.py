"""sm70 backend.linear_fp4 on the REAL lm_head weight vs reference — the last
decode step, exercised through the full backend dispatch (_served_fp4, _plan,
the sm70 GEMV), no attention compile needed. If this diverges, the fp4 GEMV
mishandles a real (vs synthetic) weight; if it matches, lm_head is not the
cause of the 220 collapse."""
import json
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, "packages/tilerl-kernels/src")
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend

d = "/data00/home/chenkailun.c/models/Qwen3.8-27B-NVFP4"
idx = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
t = load_file(d + "/" + idx["lm_head.wq"])
wq, sc, osc = t["lm_head.wq"], t["lm_head.scale"].float(), t["lm_head.oscale"].float()
N, K = wq.shape[0], wq.shape[1] * 2
print("lm_head", N, "x", K, flush=True)

torch.manual_seed(0)
x = torch.randn(1, K, dtype=torch.float32) * 0.5  # a plausible hidden row

ref = reference.linear_fp4(x, wq, sc, osc)  # [1,N] f32

be = get_backend()
y = be.linear_fp4(x.to(be.device), wq.to(be.device), sc.to(be.device),
                  oscale=osc.to(be.device)).float().cpu()

rel = (y - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
print(f"backend.linear_fp4 vs ref on real lm_head: rel={rel:.3e} pass={rel < 1e-2}", flush=True)
print("ref argmax", int(ref.argmax()), "backend argmax", int(y.argmax()), flush=True)
print("ref top5", torch.topk(ref[0], 5).indices.tolist(), flush=True)
print("backend top5", torch.topk(y[0], 5).indices.tolist(), flush=True)
