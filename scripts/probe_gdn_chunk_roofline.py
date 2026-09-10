"""CUDA timing for the chunked GDN's eight matmuls (chunk 64, batched over 48 value
heads) via torch.bmm; no correctness. The flop structure is the cost model's single
source — kernel_cost._GDN_CHUNK_MATMULS — this probe only measures us/TFLOP-s/%peak
on a card. The serial kernel does 63 ms at 1.2% of bf16 tensor peak.

    CUDA_VISIBLE_DEVICES=<card> TILERL_TARGET=cuda python3 scripts/probe_gdn_chunk_roofline.py
"""

from __future__ import annotations

import argparse

import torch
from tilerl_kernels.backend import get_backend
from torch.profiler import ProfilerActivity, profile

from tilerl.kernel_cost import _PREFILL_CHUNK


#: (label, M, N, K) roles of the eight _gdn_chunk_fwd matmuls; dk/dv are
#: substituted from cfg so the probe and the prefill declaration share dims.
def _named_matmul_dims(dk, dv, n):
    return (
        ("KK^T", n, n, dk), ("M@bV", n, n, dv), ("M@beK", n, n, dk),
        ("W@S", n, dk, dv), ("QK^T", n, n, dk), ("P@S", n, dk, dv),
        ("A@d", n, n, dv), ("R^T@d", dk, n, dv),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heads", type=int, default=48, help="value heads (the batch dim)")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    b = get_backend()
    assert b.device.type == "cuda", "needs TILERL_TARGET=cuda"
    dev = b.device

    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        return sum(e.time_range.elapsed_us() for e in prof.events()
                   if e.device_type.name == "CUDA") / args.iters

    from tilerl import config as _cfg
    cfg = _cfg.qwen38_27b()
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    n = _PREFILL_CHUNK
    named = _named_matmul_dims(dk, dv, n)

    print(f"batched over {args.heads} heads; dk=dv={dk}, chunk {n}; "
          f"peak bf16 ~148 TFLOP/s on H20")
    print(f"{'matmul':>18} {'us':>8} {'GFLOP':>8} {'TFLOP/s':>9} {'% peak':>7}")
    tot_us = tot_fl = 0.0
    for name, m, mm, k in named:
        A = torch.randn(args.heads, m, k, device=dev, dtype=torch.bfloat16)
        B = torch.randn(args.heads, k, mm, device=dev, dtype=torch.bfloat16)
        us = timed(lambda: torch.bmm(A, B))
        fl = 2.0 * args.heads * m * mm * k
        tot_us += us
        tot_fl += fl
        print(f"{name:>18} {us:>8.1f} {fl / 1e9:>8.3f} {fl / us / 1e6:>9.2f} "
              f"{100 * fl / us / 1e6 / 148:>6.1f}%")
    print(f"{'per chunk per layer':>18} {tot_us:>8.1f} {tot_fl / 1e9:>8.3f} "
          f"{tot_fl / tot_us / 1e6:>9.2f} {100 * tot_fl / tot_us / 1e6 / 148:>6.1f}%")
    # T=512 -> 8 chunks, 48 GDN layers.
    ms = tot_us * (512 // _PREFILL_CHUNK) * 48 / 1e3
    print(f"\nT=512, {512 // _PREFILL_CHUNK} chunks x 48 GDN layers: {ms:.1f} ms  "
          f"(serial gdn_chunk_fused today: 63 ms)")


if __name__ == "__main__":
    main()
