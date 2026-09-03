"""Parity for the fused frozen-base backward (dequant + gemm_nn, block 16).
The kernel decodes the served (twiddled) bytes while the reference untwiddles
first, so a wrong slot map shows up here.
  CUDA_VISIBLE_DEVICES=7 python scripts/parity_dequant_fp4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from tilerl_kernels import reference  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

bk = get_backend()
torch.manual_seed(0)
N, K = 512, 1024
w = torch.randn(N, K) * 0.1
wq, sc = reference.pack_fp4(w, block=16)
ref = reference.dequant_fp4(wq.clone(), sc)  # clone: _served_fp4 rewrites in place
wq_dev, sc_dev = wq.to(bk.device), sc.to(bk.device)

worst = 0.0
for M in (8, 64, 512):  # decode-shaped through training-shaped
    grad = torch.randn(M, N)
    gx = bk.linear_frozen_bwd(grad.to(bk.device), wq_dev, sc_dev).float().cpu()
    rel = ((gx - grad @ ref).norm() / (grad @ ref).norm()).item()
    worst = max(worst, rel)
    print(f"  linear_frozen_bwd[M={M}]  relerr {rel:.4e}")
assert worst < 1e-2, f"frozen backward parity failed: {worst:.4e}"
print("PARITY OK")
