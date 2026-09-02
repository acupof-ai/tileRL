"""Times fla's `chunk_gated_delta_rule_fwd_h` alone at our shapes, against the 10.76 ms one
layer of the 2026-08-25 port's state scan."""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--hv", type=int, default=48)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda")
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    b, t, h, dk, dv = 1, args.seq, args.hv, args.dk, args.dv
    k = torch.nn.functional.normalize(
        torch.randn(b, t, h, dk, device=dev, dtype=torch.float32), dim=-1).bfloat16()
    w = torch.randn(b, t, h, dk, device=dev, dtype=torch.bfloat16) * 0.1
    u = torch.randn(b, t, h, dv, device=dev, dtype=torch.bfloat16)
    g = (-torch.rand(b, t, h, device=dev, dtype=torch.float32) * 0.5).cumsum(1)
    h0 = torch.randn(b, h, dk, dv, device=dev, dtype=torch.float32) * 0.1

    def run():
        return chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g, initial_state=h0, output_final_state=True,
            chunk_size=args.chunk, save_new_value=True)

    run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            run()
        torch.cuda.synchronize()
    us = sum(e.time_range.elapsed_us() for e in prof.events()
             if e.device_type.name == "CUDA") / args.iters
    n = t // args.chunk
    print(f"fla inter-chunk state scan, T={t} ({n} chunks), {h} heads, DK=DV={dk}")
    print(f"  one layer: {us:.1f} us     48 layers: {us * 48 / 1e3:.2f} ms")
    print(f"  the 2026-08-25 port's kernel B: 10760 us for one layer")
    print(f"  ratio: {10760 / us:.0f}x")


if __name__ == "__main__":
    main()
