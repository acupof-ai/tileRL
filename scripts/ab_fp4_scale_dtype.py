"""A/B the fp4 decode GEMV's Scale dtype (f32 shipped vs bf16 vs e4m3): the
scale stream is 1/3 of the kernel's bytes at block 16. Generates kernel
variants by rewriting the Scale literal (tilelang reads dtypes from source).
Run: python scripts/ab_fp4_scale_dtype.py --gpu 7"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=7)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("TILERL_TARGET", "cuda")

import torch  # noqa: E402

import benchkit as bk  # noqa: E402
from tilerl.config import qwen38_27b  # noqa: E402
from tilerl.ops import reference  # noqa: E402
from tilerl.ops.backend import get_backend  # noqa: E402

SRC = (HERE.parent / "src/tilerl/ops/kernels_linear.py").read_text()
VAR = HERE / "_fp4_scale_variants"
VAR.mkdir(exist_ok=True)


def variant(dtype: str):
    src = SRC.replace('Scale: T.Tensor((N, K // block), "float32")', f'Scale: T.Tensor((N, K // block), "{dtype}")')
    src = src.replace("from .kernels_mma import", "from tilerl.ops.kernels_mma import").replace("from . import", "from tilerl.ops import")
    assert src != SRC or dtype == "float32"
    p = VAR / f"kl_{dtype}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(f"kl_{dtype}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.make_linear_fp4_gemv


cfg = qwen38_27b()
h, inter = cfg.hidden_size, cfg.intermediate_size
shapes = {"gate_up": (2 * inter, h), "down": (h, inter), "o_proj": (h, cfg.num_attention_heads * cfg.head_dim)}
backend = get_backend()
blk = 16
for name, (N, K) in shapes.items():
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    wq = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    s32 = (torch.rand(N, K // blk, device="cuda") + 0.5).to(torch.float8_e4m3fn).float()  # e4m3-exact
    ref = (reference.dequant_fp4(wq, s32).float() @ x.float().T).T
    arms = []
    for dt, tdt in (("float32", torch.float32), ("bfloat16", torch.bfloat16), ("float8_e4m3fn", torch.float8_e4m3fn)):
        k = variant(dt)(backend.target)
        sc = s32.to(tdt)
        arms.append((dt, lambda k=k, sc=sc: (k(x, wq, sc, 32, 4, blk).float(),)))
    roof = (N * K * 0.5 + 2 * K) / 3254e9 * 1e3
    bk.ab(f"fp4 gemv {name} N={N} K={K} (roof w/o scales {roof:.4f} ms)", arms, (ref,))
