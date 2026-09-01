"""ncu target: the M-row fp4 GEMV at the shape where reuse works and one where it does not.

ab_m8_reuse.py showed linear_fp4_gemv_sm70_m scales 1.65x for 8 rows at
N=K=4864 but 7.5x at 17408x5120, where M=8 costs more per row than M=1.
Weight re-decode, occupancy, convert count and X re-reads are all ruled out by
arithmetic, so this exists to be run under ncu:

  ncu --set full --kernel-name regex:linear_fp4_gemv_sm70_m \
      --launch-count 2 python3 scripts/ncu_m8_gemv.py

One launch per shape, nothing else on the GPU, so the two profiles line up
side by side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src")
)

import torch
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend

SHAPES = {"good": (4864, 4864), "bad": (17408, 5120)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--only", choices=sorted(SHAPES), help="profile just one shape")
    args = ap.parse_args()

    backend = get_backend()
    dev = backend.device
    names = [args.only] if args.only else sorted(SHAPES)
    for name in names:
        N, K = SHAPES[name]
        w = torch.randn(N, K, dtype=torch.bfloat16) * 0.02
        wq, sc = reference.pack_fp4(w)
        sc, osc = reference.renorm_fp4_scale(sc)
        p = backend.materialize({"w.wq": wq, "w.scale": sc, "w.oscale": osc})
        x = torch.randn(args.m, K, device=dev)
        # Warm the JIT and the twiddle outside the profiled launch.
        backend.linear_fp4(x, p["w.wq"], p["w.scale"], oscale=p["w.oscale"])
        torch.cuda.synchronize()
        print(f"profiling {name}: N={N} K={K} M={args.m}", flush=True)
        backend.linear_fp4(x, p["w.wq"], p["w.scale"], oscale=p["w.oscale"])
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
