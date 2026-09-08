"""What does one svdvals cost on the card, and which tensors is the Sigma gate iterating?

The gate OOMed asking for 9.47 GiB (f64 of the 248320x5120 embedding) with 91.65 of 95.22
GiB already held. Before redesigning it, two numbers:
  1. the 2D parameter inventory -- names, shapes, elements, f32 and f64 bytes
  2. wall time for svdvals on the largest few, since a gate that takes an hour per arm is
     not a gate even if it fits
"""
import os
import time
from collections import Counter

import torch

os.environ.setdefault("TILERL_TARGET", "cuda")

from tilerl.cli import _build_model  # noqa: E402

cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
two_d = {k: v for k, v in model.params.items() if v.dim() == 2}
tot = sum(v.numel() for v in two_d.values())
print(f"2D params: {len(two_d)} of {len(model.params)}, {tot/1e9:.2f}G elements")
print(f"card: {torch.cuda.memory_allocated()/2**30:.1f} GiB allocated\n")

rows = sorted(two_d.items(), key=lambda kv: -kv[1].numel())
print(f"{'name':52} {'shape':>20} {'Melem':>9} {'f32 GiB':>8} {'f64 GiB':>8}")
for k, v in rows[:8]:
    print(f"{k:52} {str(tuple(v.shape)):>20} {v.numel()/1e6:9.1f} "
          f"{4*v.numel()/2**30:8.2f} {8*v.numel()/2**30:8.2f}")
print(f"... {len(rows)-8} more, smallest {rows[-1][1].numel()/1e6:.1f} Melem\n")

# Distinct shapes: the cost is per shape class, and a per-class count prices the whole sweep.
classes = Counter(tuple(v.shape) for v in two_d.values())
print("distinct 2D shapes and their counts:")
for shape, n in sorted(classes.items(), key=lambda kv: -kv[0][0]*kv[0][1]):
    print(f"  {str(shape):>22} x{n:3}  {n*shape[0]*shape[1]/1e9:6.2f}G elem total")

print("\nsvdvals wall time, one tensor per shape class, f32 on the card:")
for shape, n in sorted(classes.items(), key=lambda kv: -kv[0][0]*kv[0][1]):
    k, v = next((k, v) for k, v in two_d.items() if tuple(v.shape) == shape)
    free = torch.cuda.mem_get_info()[0] / 2**30
    need = 4 * v.numel() / 2**30
    if need > free - 2:
        print(f"  {str(shape):>22} SKIPPED: needs {need:.2f} GiB, {free:.2f} free")
        continue
    torch.cuda.synchronize()
    t = time.perf_counter()
    try:
        s = torch.linalg.svdvals(v.detach().float())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        print(f"  {str(shape):>22} {dt:7.2f} s  x{n:3} = {n*dt:8.1f} s for the class "
              f"(smin {s.min().item():.3e} smax {s.max().item():.3e})")
        del s
    except Exception as e:
        print(f"  {str(shape):>22} FAILED {type(e).__name__}: {str(e)[:60]}")
    torch.cuda.empty_cache()
