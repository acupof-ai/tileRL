"""Is the GDN prefill kernel SM-limited? Vary the block count in ONE launch.

The kernel's grid is (num_value_heads, batch), so a batch of 2 doubles the
blocks inside a single launch — which the engine never does, because it admits
one prefill row per tick. If per-launch time grows much slower than the block
count, the SMs were idle at B=1 (48 blocks, 78 SMs) and splitting the value
dimension across more blocks is worth a kernel. If it grows linearly, the
kernel is not SM-limited and that lever is dead.

  python scripts/occ_gdn.py --gpu 7
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--t", type=int, default=512)
    ap.add_argument("--batches", default="1,2,3,4")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    import torch

    from tilerl.config import qwen38_27b
    from tilerl.ops.backend import get_backend

    backend = get_backend()
    cfg = qwen38_27b()
    nkh, nvh = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    kd, vd, ker = cfg.linear_key_head_dim, cfg.linear_value_head_dim, cfg.linear_conv_kernel_dim
    qkv = 2 * nkh * kd + nvh * vd
    print(f"nvh={nvh} -> {nvh} blocks per batch row; H20 has 78 SMs")
    print(f"  {'B':>3} {'blocks':>7} {'ms/launch':>10} {'vs B=1':>8} {'ideal':>7}")
    base = None
    for b in (int(x) for x in args.batches.split(",")):
        t = args.t
        g = torch.randn
        q, k = g(b, t, nkh * kd) * 0.1, g(b, t, nkh * kd) * 0.1
        v, z = g(b, t, nvh * vd) * 0.1, g(b, t, nvh * vd) * 0.1
        kw = dict(
            conv1d_weight=g(qkv, ker) * 0.1, dt_bias=g(nvh), a_log=g(nvh) * 0.1,
            norm_weight=torch.ones(vd), conv_window=g(b, ker - 1, qkv) * 0.1,
            seq_q_lens=torch.full((b,), t, dtype=torch.int32, device=backend.device),
        )
        st = g(b, nvh, kd, vd) * 0.01

        def run():
            backend.linear_attn_chunk(q, k, v, g(b, t, nvh) * 0.0, g(b, t, nvh), st, z=z, **kw)

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        for _ in range(10):
            run()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 10
        base = base or ms
        print(f"  {b:>3} {nvh * b:>7} {ms:>10.3f} {ms / base:>8.2f}x {b:>6}x")
    print("  sub-linear => SMs were idle, a V split adds real parallelism")


if __name__ == "__main__":
    main()
