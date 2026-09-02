"""Parity for the sm70 fp16-twiddle GEMV (linear_fp4, M=1) vs the f32 reference.

The kernel decodes the SERVED (fp16-twiddled) bytes; the reference dequants the
natural bytes. A wrong slot map / shift / rebias shows up here immediately —
the cheap check on a bit permutation before the 27B end-to-end.

  CUDA_VISIBLE_DEVICES=0 TILERL_TARGET=cuda python scripts/parity_sm70_gemv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src"))

import torch  # noqa: E402

from tilerl_kernels import reference  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

bk = get_backend()
assert bk.arch == "sm70", f"this parity is for sm70, got {bk.arch}"
torch.manual_seed(0)
g = torch.Generator().manual_seed(1)

worst = 0.0
# K=512/1024 exercise the K-tail (G=1); K=2048+ exercises the GROUP=4 path.
for (N, K) in [(64, 512), (256, 1024), (512, 2048), (128, 4096)]:
    w = torch.randn(N, K, generator=g) * 0.1
    wq, sc = reference.pack_fp4(w, block=16)
    ref = reference.dequant_fp4(wq, sc)  # natural bytes -> [N, K] f32
    x = torch.randn(1, K, generator=g)
    y = bk.linear_fp4(x.to(bk.device), wq.to(bk.device), sc.to(bk.device)).float().cpu()
    expect = x @ ref.T
    rel = ((y - expect).norm() / expect.norm()).item()
    worst = max(worst, rel)
    print(f"  linear_fp4[M=1, N={N:>4}, K={K:>5}]  relerr {rel:.4e}")

print(f"\nworst {worst:.4e}  gate 1e-2  {'PASS' if worst < 1e-2 else 'FAIL'}")
sys.exit(1 if worst >= 1e-2 else 0)
