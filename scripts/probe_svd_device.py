"""Correct the 1698 s figure: was it measured on the card or on the host?

`probe_svd_cost.py` printed `card: 0.0 GiB allocated` immediately after `_build_model`, which
says the params were still on the HOST when it timed svdvals -- `v.detach().float()` of a CPU
tensor stays on CPU, and the `torch.cuda.synchronize()` around it is then a no-op timing
nothing. So the per-class times may be host times reported as card times.

This prints the device before timing and times both, so the two are separated by measurement
rather than by argument. `--limit` keeps it to the largest few classes: the point is the
host/card RATIO, not another full census.
"""
import argparse
import os
import time
from collections import Counter

import torch

os.environ.setdefault("TILERL_TARGET", "cuda")

from tilerl.cli import _build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=4, help="largest N shape classes to time")
a = ap.parse_args()

cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
two_d = {k: v for k, v in model.params.items() if v.dim() == 2}
devs = Counter(str(v.device) for v in two_d.values())
print(f"2D params: {len(two_d)}, devices: {dict(devs)}")
print(f"cuda allocated after build: {torch.cuda.memory_allocated()/2**30:.2f} GiB")
print(f"cuda available: {torch.cuda.mem_get_info()[0]/2**30:.2f} GiB free of "
      f"{torch.cuda.mem_get_info()[1]/2**30:.2f}\n")

classes = Counter(tuple(v.shape) for v in two_d.values())
big = sorted(classes.items(), key=lambda kv: -kv[0][0] * kv[0][1])[:a.limit]

print(f"{'shape':>22} {'n':>4} {'host s':>9} {'card s':>9} {'ratio':>7} "
      f"{'host class s':>13} {'card class s':>13}")
tot_h = tot_c = 0.0
for shape, n in big:
    k, v = next((k, v) for k, v in two_d.items() if tuple(v.shape) == shape)
    src = v.detach().float()

    t = time.perf_counter()
    torch.linalg.svdvals(src.cpu())
    hs = time.perf_counter() - t

    free = torch.cuda.mem_get_info()[0] / 2**30
    need = 4 * v.numel() / 2**30 * 3   # the matrix plus svdvals' workspace, roughly
    if need > free - 2:
        print(f"{str(shape):>22} {n:>4} {hs:9.2f} {'SKIP':>9} {'-':>7} "
              f"{n*hs:13.1f} {'needs ' + format(need, '.1f') + ' GiB':>13}")
        tot_h += n * hs
        continue
    g = src.cuda()
    torch.cuda.synchronize()
    t = time.perf_counter()
    torch.linalg.svdvals(g)
    torch.cuda.synchronize()
    cs = time.perf_counter() - t
    del g
    torch.cuda.empty_cache()
    tot_h += n * hs
    tot_c += n * cs
    print(f"{str(shape):>22} {n:>4} {hs:9.2f} {cs:9.2f} {hs/max(cs,1e-9):7.2f} "
          f"{n*hs:13.1f} {n*cs:13.1f}")

print(f"\nthese {a.limit} classes: host {tot_h:.0f} s, card {tot_c:.0f} s")
print("The 1698 s previously reported was measured with params on the host and a no-op "
      "synchronize, so it prices the HOST path. Whichever number is larger is the one a "
      "census would actually cost, since the gate reads params wherever they live.")
