"""Per-call host cost of our GDN core vs fla's: wall clock minus profiler GPU time at one
layer's shapes. fla is 42 ms cheaper on the GPU per prefill yet 6.7% slower end to end."""

from __future__ import annotations

import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

from tilerl_kernels import reference as R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--hk", type=int, default=16)
    ap.add_argument("--hv", type=int, default=48)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    b, t = 1, args.seq
    qn = torch.randn(b, t, args.hk, args.dk, device=dev)
    qn = qn / qn.norm(dim=-1, keepdim=True) / args.dk ** 0.5
    kn = torch.randn(b, t, args.hk, args.dk, device=dev)
    kn = kn / kn.norm(dim=-1, keepdim=True)
    v = torch.randn(b, t, args.hv, args.dk, device=dev)
    gt = -torch.rand(b, t, args.hv, device=dev) * 0.5
    bt = torch.rand(b, t, args.hv, device=dev)
    st = torch.randn(b, args.hv, args.dk, args.dk, device=dev) * 0.1

    def measure(name, fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / args.iters * 1e3
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        gpu = sum(e.time_range.elapsed_us() for e in prof.events()
                  if e.device_type.name == "CUDA") / args.iters / 1e3
        print(f"  {name:>22}: wall {wall:7.3f} ms   gpu {gpu:7.3f} ms   "
              f"host {wall - gpu:7.3f} ms")
        return wall - gpu

    print(f"one GDN layer, T={t}, {args.hk} key / {args.hv} value heads")
    h_ours = measure("ours (chunk_core)", lambda: R.gdn_chunk_core(qn, kn, v, gt, bt, st, 64))
    h_fla = measure("fla", lambda: R.gdn_chunk_core_fla(qn, kn, v, gt, bt, st, 64))
    print(f"\nper-layer host delta: {h_fla - h_ours:+.3f} ms"
          f"   x48 layers: {(h_fla - h_ours) * 48:+.1f} ms")


if __name__ == "__main__":
    main()
