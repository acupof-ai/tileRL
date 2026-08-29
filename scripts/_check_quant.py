"""Sanity-check the quantize_nvfp4.py output: dequant one real fp4 linear and
compare to the original bf16 weight. If my quantizer is wrong, this shows it
without running the model."""
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, "packages/tilerl-kernels/src")
from tilerl_kernels.reference import dequant_fp4

bf16_dir = sys.argv[1]  # original bf16
fp4_dir = sys.argv[2]   # my quantized output

# Find a linear weight present in shard 1 of both.
import glob
bf16_sh = sorted(glob.glob(f"{bf16_dir}/model-*.safetensors"))[0]
fp4_sh = sorted(glob.glob(f"{fp4_dir}/model-*.safetensors"))[0]
bt = load_file(bf16_sh)
ft = load_file(fp4_sh)

# A fp4 linear: stem with .wq in the fp4 output.
stem = None
for k in ft:
    if k.endswith(".wq"):
        stem = k[:-3]
        break
print("checking stem:", stem)
orig = bt[stem + ".weight"].float()  # [N,K] bf16 original
wq = ft[stem + ".wq"]
scale = ft[stem + ".scale"].float()
oscale = ft[stem + ".oscale"].float()
print("shapes: orig", tuple(orig.shape), "wq", tuple(wq.shape), "scale", tuple(scale.shape), "oscale", tuple(oscale.shape))
print("oscale range:", oscale.min().item(), "..", oscale.max().item())

# Dequant: w = oscale[:,None] * dequant_fp4(wq, scale)
deq = dequant_fp4(wq, scale) * oscale[:, None]
rel = (deq - orig).abs().max().item() / (orig.abs().max().item() + 1e-9)
print(f"dequant vs original: max_err={((deq-orig).abs().max()).item():.4f} rel={rel:.4e}")
print("orig[0,:6]:", orig[0, :6].tolist())
print("deq [0,:6]:", deq[0, :6].tolist())
print("orig row norms (first 5):", orig[:5].norm(dim=1).tolist())
print("deq  row norms (first 5):", deq[:5].norm(dim=1).tolist())
print("NaN in deq:", torch.isnan(deq).any().item(), "all-zero rows:", (deq.abs().sum(1) == 0).sum().item())
