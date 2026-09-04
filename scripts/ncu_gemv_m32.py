"""One M=32 GEMV launch, sized for ncu. Confirms issue-bound before redesigning.

errors/2026-09-02-npartition-is-not-the-m32-lever.md derived ~15% of issue peak
from an instruction count off the extern source (436 instr per 16 weight elems at
M=32, 64 X loads feeding 8 useful FMA pairs). That is arithmetic, not a
measurement, and the two levers left (stage X in SMEM, partial M unroll) are both
real kernel work with numerics risk -- so confirm the diagnosis first.

What would refute it: high memory throughput (it is bandwidth-bound after all),
or low occupancy (it is occupancy-bound and the fix is geometry, not issue slots).

  scripts/v100.sh run nc '/usr/local/cuda-12.4/bin/ncu --set detailed \
      --kernel-name regex:linear_fp4_gemv --launch-count 1 \
      /usr/bin/python3 scripts/ncu_gemv_m32.py'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

os.environ.setdefault("TILERL_TARGET", "cuda")
from tilerl_kernels import kernels_linear  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

N, K = 34816, 5120  # gate_up: the largest single shape, 64 launches/token
BLK, M, NP = 32, 32, 4


def main() -> None:
    be = get_backend()
    dev = be.device
    k = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, xh=True, sh=True)
    g = torch.Generator(device="cpu").manual_seed(N * 131 + K)
    wq = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, generator=g).to(dev)
    sc = (torch.rand(N, K // BLK, generator=g) * 1.9 + 0.0025).to(dev).half()
    osc = torch.ones(N, device=dev)
    res = torch.zeros(M, N, device=dev)
    x = torch.randn(M, K, generator=g).to(dev).half()
    k(x, wq, sc, osc, res, 32, NP, BLK)  # warm: JIT, so ncu profiles the real launch
    torch.cuda.synchronize()
    k(x, wq, sc, osc, res, 32, NP, BLK)
    torch.cuda.synchronize()
    print(f"gate_up {N}x{K}, M={M}, np={NP}: 2 launches (1 warm, 1 profiled)")


if __name__ == "__main__":
    main()
