"""Parity for the fp4 dequant kernel and the frozen-base backward it serves.

The kernel decodes the SERVED (twiddled) bytes; the reference untwiddles first.
If the slot map is wrong the two disagree immediately — which is the only cheap
way to check a bit permutation.

  CUDA_VISIBLE_DEVICES=7 python scripts/parity_dequant_fp4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from tilerl.ops import reference  # noqa: E402
from tilerl.ops.backend import get_backend  # noqa: E402

bk = get_backend()
torch.manual_seed(0)
N, K = 512, 1024
w = torch.randn(N, K) * 0.1
wq, sc = reference.pack_fp4(w)
ref = reference.dequant_fp4(wq.clone(), sc)  # clone: _served_fp4 rewrites in place

wq_dev, sc_dev = wq.to(bk.device), sc.to(bk.device)
got = bk._kernel("dequant_fp4_bf16")(
    bk._served_fp4(wq_dev), bk._const_f32(sc_dev), bk._ones(N), K // sc.shape[1]
).float().cpu()
rel = ((got - ref).norm() / ref.norm()).item()
print(f"  dequant_fp4_bf16   relerr {rel:.4e}")
assert rel < 1e-2, f"dequant parity failed: {rel:.4e}"

grad = torch.randn(8, N)
gx = bk.linear_frozen_bwd(grad.to(bk.device), wq_dev, sc_dev).float().cpu()
gx_ref = grad @ ref
rel = ((gx - gx_ref).norm() / gx_ref.norm()).item()
print(f"  linear_frozen_bwd  relerr {rel:.4e}")
assert rel < 1e-2, f"frozen backward parity failed: {rel:.4e}"
print("PARITY OK")
