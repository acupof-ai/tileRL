"""Microbench: does a CUDA graph pool reuse freed graph memory across captures?

If yes, a filler capture can poison the pool's free list and the forward's
next capture allocates from poisoned memory. If no, every capture gets fresh
cudaMalloc zero pages and the uninitialized-pool-memory hypothesis is dead.

No model needed; run on any free card.
"""
import gc

import torch

dev = "cuda"
pool = torch.cuda.graph_pool_handle()
BIG = 1 << 30  # 1 GiB
SMALL = 256 << 20  # 256 MiB


def filler(value, destroy=True):
    gf = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf, pool=pool):
        buf = torch.empty(BIG, dtype=torch.uint8, device=dev)
        buf.fill_(value)
        del buf
    if destroy:
        del gf
        gc.collect()


def reader():
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, pool=pool):
        t = torch.empty(SMALL, dtype=torch.uint8, device=dev)
        out = t.clone()
        del t
    gr.replay()
    torch.cuda.synchronize()
    frac = (out == 0xFF).float().mean().item()
    del gr
    gc.collect()
    return frac


# Baseline: no filler, fresh pool -> expect 0.0 (zero pages).
print(f"no filler:            reader 0xFF fraction = {reader():.4f}")

# Filler 0xFF, destroy graph + gc -> does the reader see 0xFF?
filler(0xFF)
print(f"filler(0xFF)+destroy: reader 0xFF fraction = {reader():.4f}")

# Filler 0xFF, keep graph alive -> reader should NOT see 0xFF (memory held).
gf_keep = torch.cuda.CUDAGraph()
with torch.cuda.graph(gf_keep, pool=pool):
    buf = torch.empty(BIG, dtype=torch.uint8, device=dev)
    buf.fill_(0xFF)
    del buf
print(f"filler(0xFF)+kept:    reader 0xFF fraction = {reader():.4f}")

# Within-capture free+reuse: allocate big, fill, free, alloc small in SAME capture.
g_same = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_same, pool=pool):
    p = torch.empty(BIG, dtype=torch.uint8, device=dev)
    p.fill_(0xFF)
    del p
    t = torch.empty(SMALL, dtype=torch.uint8, device=dev)
    out_same = t.clone()
    del t
g_same.replay()
torch.cuda.synchronize()
print(f"same-capture free:    reader 0xFF fraction = "
      f"{(out_same == 0xFF).float().mean().item():.4f}")
