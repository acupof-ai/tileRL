"""Split-KV decode attention vs the dense MMA kernel at long KV depth (sm90).
Run: python scripts/parity_attn_decode.py --gpu 7 --len 32768"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=7)
ap.add_argument("--len", type=int, default=32768)
ap.add_argument("--batch", type=int, default=2)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("TILERL_TARGET", "cuda")

import torch  # noqa: E402

import benchkit as bk  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

backend = get_backend()
B, H, HKV, D, BS = args.batch, 32, 8, 256, 16
n = args.len
nb = -(-n // BS) * B
k = torch.randn(nb, HKV, BS, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(nb, HKV, BS, D, device="cuda", dtype=torch.bfloat16)
bt = torch.arange(nb, device="cuda", dtype=torch.int32).reshape(B, -1)
lens = torch.tensor([n, n - 777], device="cuda", dtype=torch.int32)[:B]
q = torch.randn(B, 1, H, D, device="cuda", dtype=torch.bfloat16)
scale = D**-0.5
qlens = torch.ones(B, dtype=torch.int32, device="cuda")
ref = backend._kernel("paged_attention")(q, k, v, bt, lens, qlens, scale, BS, 16, 128)[:, :1].float()
out = backend._paged_attention_decode(q, k, v, bt, lens, qlens, scale).float()
print(f"len {n}: relerr {bk.relerr(out, ref):.3e}")
ms_ref = bk.timeit(lambda: backend._kernel("paged_attention")(q, k, v, bt, lens, qlens, scale, BS, 16, 128), 20)
ms_new = bk.timeit(lambda: backend._paged_attention_decode(q, k, v, bt, lens, qlens, scale), 20)
print(f"dense {ms_ref:.3f} ms  split-kv {ms_new:.3f} ms  ({ms_ref / ms_new:.1f}x)")
