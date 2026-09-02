"""Parity for the sm70 M-row ladder through the real dispatch site.

backend.linear_fp4 now packs X to f16 and calls the *h rungs; this checks the
whole path — rung selection, padding, round-up of odd widths — against the f32
dequant reference at the real projection shapes.

  scripts/v100.sh '/usr/bin/python3 scripts/parity_sm70_gemv_m.py'
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src"))

import torch
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend

bk = get_backend()
assert bk.arch == "sm70", f"this parity is for sm70, got {bk.arch}"
g = torch.Generator().manual_seed(1)

worst = 0.0
for N, K in [(17408, 5120), (12288, 5120), (5120, 17408)]:
    w = torch.randn(N, K, generator=g) * 0.1
    wq, sc = reference.pack_fp4(w, block=16)
    ref = reference.dequant_fp4(wq, sc)
    wq_d = reference.twiddle_fp4_f16(wq).to(bk.device)
    wq_d._tl_layout = "f16_twiddle"
    sc_d = sc.to(bk.device)
    for M in (1, 2, 3, 4, 5, 8):
        x = torch.randn(M, K, generator=g)
        y = bk.linear_fp4(x.to(bk.device), wq_d, sc_d).float().cpu()
        expect = x @ ref.T
        rel = ((y - expect).norm() / expect.norm()).item()
        worst = max(worst, rel)
        print(f"  N={N:>5} K={K:>5} M={M}  relerr {rel:.4e}")

print(f"\nworst {worst:.4e}  gate 1e-2  {'PASS' if worst < 1e-2 else 'FAIL'}")
sys.exit(1 if worst >= 1e-2 else 0)
