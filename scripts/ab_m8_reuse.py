"""Does the sm70 M-row fp4 GEMV actually reuse the decoded weight tile?

profile_verify_replay.py measured linear_fp4_gemv_sm70_m at 507.5 us for 8 rows
against the M=1 kernel's 64.5 us — 63.4 us/row, i.e. no reuse at all, which is
what makes a W>1 verify replay 271 ms against 40.9 ms at W=1. But
wins/2026-08-30-sm70-fp16-twiddle-gemv.md measured 1.79x for 8 rows at
N=K=4864, where reuse clearly does work. This sweeps both shapes to find where
it stops.

  TILERL_TARGET=cuda python3 scripts/ab_m8_reuse.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src")
)

import torch
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend

# 4864 is the shape the win entry measured; the rest are the 27B's real
# projections (q, gate/up, down) that the verify replay actually runs.
SHAPES = [(4864, 4864), (12288, 5120), (17408, 5120), (5120, 17408)]


def bench(fn, n: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def main() -> None:
    backend = get_backend()
    dev = backend.device
    print(f"arch={backend.arch}")
    hdr = ("N", "K", "M=1 us", "M=8 us", "ratio", "us/row", "reuse?")
    print("{:>6s} {:>6s} {:>9s} {:>9s} {:>7s} {:>8s} {:>7s}".format(*hdr))
    for N, K in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16) * 0.02
        wq, sc = reference.pack_fp4(w)
        sc, osc = reference.renorm_fp4_scale(sc)
        p = backend.materialize({"w.wq": wq, "w.scale": sc, "w.oscale": osc})
        x1 = torch.randn(1, K, device=dev)
        x8 = torch.randn(8, K, device=dev)
        a = bench(lambda: backend.linear_fp4(x1, p["w.wq"], p["w.scale"], oscale=p["w.oscale"]))
        b = bench(lambda: backend.linear_fp4(x8, p["w.wq"], p["w.scale"], oscale=p["w.oscale"]))
        # Reuse working means 8 rows cost far less than 8 separate passes.
        verdict = "yes" if b / a < 4 else "NO"
        print(f"{N:6d} {K:6d} {a:9.1f} {b:9.1f} {b / a:7.2f} {b / 8:8.1f} {verdict:>7s}")
    print("\nratio ~1.8 = tile reused across rows; ~8 = re-decoded per row")


if __name__ == "__main__":
    main()
