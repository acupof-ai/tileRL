"""A/B the fp8 decode GEMV: flat (GROUP=1) vs grouped prefetch, real 27B
GDN shapes, synthetic e4m3 weights. Run: python scripts/ab_fp8_gemv.py --gpu 6"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=6)
ap.add_argument("--groups", default="1,2,4,8")
ap.add_argument("--nparts", default="4,8")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("TILERL_TARGET", "cuda")

import torch  # noqa: E402

import benchkit as bk  # noqa: E402
from tilerl.config import qwen38_27b  # noqa: E402
from tilerl_kernels import kernels_linear  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

cfg = qwen38_27b()
h = cfg.hidden_size
nvh, kd, vd = cfg.linear_num_value_heads, cfg.linear_num_key_heads * cfg.linear_key_head_dim, cfg.linear_num_value_heads * cfg.linear_value_head_dim
qd = kd
shapes = {"in_proj(fused)": (qd + kd + vd + vd + 2 * nvh, h), "out_proj": (h, vd)}
backend = get_backend()
bw = bk.__dict__.get("hbm_gbs", lambda: 3254.0)()
for name, (N, K) in shapes.items():
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    w8 = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    ws = torch.rand(-(-N // 128), -(-K // 128), device="cuda") + 0.5
    ref = (x.float() @ (w8.float() * ws.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]).T,)
    arms = []
    for g in (int(v) for v in args.groups.split(",")):
        k = kernels_linear.make_linear_fp8_gemv(backend.target, GROUP=g)
        for np_ in (int(v) for v in args.nparts.split(",")):
            arms.append((f"G{g}-np{np_}", lambda k=k, np_=np_: (k(x, w8, ws, torch.ones(N, device="cuda"), 32, np_),)))
    rows = bk.ab(f"fp8 gemv {name} N={N} K={K} (roof {1e3 * (N * K + 2 * K) / bw / 1e9:.4f} ms)", arms, ref)
