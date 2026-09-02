"""Parity for the batched-decode tensor-core GEMMs (fp4, fp8) at M=2..8 vs the
dequantized reference; a compile defect shows here in seconds, not 20 min into the harness.
  CUDA_VISIBLE_DEVICES=7 python scripts/parity_mma8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from tilerl_kernels.backend import get_backend  # noqa: E402
from tilerl_kernels.reference import dequant_fp4, dequant_fp8, pack_fp4  # noqa: E402


def quant_fp8(w):  # per-128-block, the loader's native layout
    n, k = w.shape
    ns, ks = -(-n // 128), -(-k // 128)
    padded = w.float().new_zeros(ns * 128, ks * 128)
    padded[:n, :k] = w.float()
    blocks = padded.reshape(ns, 128, ks, 128)
    bmax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    w8 = (blocks / (bmax / 448.0)).reshape(ns * 128, ks * 128)[:n, :k]
    return w8.to(torch.float8_e4m3fn).contiguous(), (bmax / 448.0).reshape(ns, ks).contiguous()

bk = get_backend()
N, K = 512, 1024
torch.manual_seed(0)
w = torch.randn(N, K) * 0.1
wq, sc = pack_fp4(w)
w8, wsc = quant_fp8(w)
worst = 0.0
for M in (2, 4, 8):
    x = torch.randn(M, K)
    for name, y, ref in (
        ("fp4", bk.linear_fp4(x, wq, sc), x @ dequant_fp4(wq, sc).t()),
        ("fp8", bk.linear_fp8(x, w8, wsc), x @ dequant_fp8(w8, wsc).t()),
    ):
        ref = ref.to(y.device, torch.float32)
        rel = ((y - ref).norm() / ref.norm()).item()
        worst = max(worst, rel)
        print(f"  linear_{name}[M={M}]  relerr {rel:.4e}")
print(f"worst {worst:.4e}")
assert worst < 1e-2, f"mma8 parity failed: {worst:.4e}"
print("PARITY OK")
