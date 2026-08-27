"""A/B the fp4 decode GEMV's split-K (blocks = N/4 * k_split) on the real
27B fp4 shapes. Run: python scripts/ab_fp4_ksplit.py --gpu 7"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=7)
ap.add_argument("--splits", default="1,2,4,8,16")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("TILERL_TARGET", "cuda")

import torch  # noqa: E402

import benchkit as bk  # noqa: E402
from tilerl.config import qwen38_27b  # noqa: E402
from tilerl.ops import kernels_linear, reference  # noqa: E402
from tilerl.ops.backend import get_backend  # noqa: E402

cfg = qwen38_27b()
h, inter, hd = cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads * cfg.head_dim
shapes = {"gate_up": (2 * inter, h), "down": (h, inter), "o_proj": (h, hd),
          "qkv": (2 * hd + 2 * cfg.num_kv_heads * cfg.head_dim, h)}
backend = get_backend()
k = kernels_linear.make_linear_fp4_gemv(backend.target)
blk = 16
for name, (N, K) in shapes.items():
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    wq = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    sc = (torch.rand(N, K // blk, device="cuda") + 0.5).to(torch.float8_e4m3fn).float()
    ref = (reference.dequant_fp4(wq, sc).float() @ x.float().T).T
    arms = [(f"ks{ks}", lambda ks=ks: (k(x, wq, sc, 32, 4, blk, ks).sum(0, keepdim=True),))
            for ks in (int(v) for v in args.splits.split(","))]
    roof = (N * K * 0.75 + 2 * K) / 3254e9 * 1e3
    bk.ab(f"fp4 gemv {name} N={N} K={K} (roof {roof:.4f} ms)", arms, (ref,))
