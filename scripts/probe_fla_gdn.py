"""What does fla's chunked gated-delta actually cost at OUR shapes?

Our `gdn_chunk_fused` runs a T-long scalar scan in one kernel: 63 ms of a 229 ms
prefill, 0.13% of the tensor pipe. fla (what sglang runs) does the same maths as
four kernels with the chunk interior as matmuls. Every estimate of the gap so
far has been mine; this is the number.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_fla_gdn.py
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--layers", type=int, default=48, help="GDN layers in the 27B")
    ap.add_argument("--hv", type=int, default=48, help="value heads")
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device("cuda")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    b, t, h, dk, dv = 1, args.seq, args.hv, args.dk, args.dv
    q = torch.randn(b, t, h, dk, device=dev, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(b, t, h, dk, device=dev, dtype=torch.float32), dim=-1).bfloat16()
    v = torch.randn(b, t, h, dv, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(b, t, h, device=dev, dtype=torch.float32) * 0.5
    beta = torch.rand(b, t, h, device=dev, dtype=torch.bfloat16)

    def run():
        return chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, scale=dk ** -0.5,
                                      initial_state=None, output_final_state=False)

    run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            run()
        torch.cuda.synchronize()
    by: dict[str, list] = {}
    tot = 0.0
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        us = e.time_range.elapsed_us() / args.iters
        r = by.setdefault(e.name[:46], [0, 0.0])
        r[0] += 1
        r[1] += us
        tot += us
    print(f"fla chunk_gated_delta_rule, b=1 t={t} hv={h} dk={dk} dv={dv}")
    print(f"{'kernel':<46} {'n':>4} {'us':>9}")
    for name, (c, us) in sorted(by.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"{name:<46} {c // args.iters:>4} {us:>9.1f}")
    print(f"{'ONE LAYER total':<46} {'':>4} {tot:>9.1f} us")
    print(f"\n{args.layers} GDN layers: {tot * args.layers / 1e3:.1f} ms"
          f"   (ours today: 63 ms; chunked-matmul roofline: 9.9 ms)")


if __name__ == "__main__":
    main()
