"""Does the sm70 M-path GEMV lose efficiency where X stops fitting L1?

Task #26's surviving hypothesis after n_partition and the FMA chain both died at
1.01x: X[M,K] f16 fits L2 but not the 128 KB L1 at M=32, so the per-tile row reads
hit L2 at ~200 cycles on the FMA critical path. ncu would answer this directly but
the pod denies performance counters to non-root (ERR_NVGPUCTRPERM), so the test has
to come from a prediction that only this mechanism makes.

It makes a sharp one. X's working set is 2*M*K bytes, so the L1 knee sits at

    M_knee = 131072 / (2*K)

which is M~12.8 at K=5120 and M~51 at K=1280. **A capacity limit's knee moves with
K; a schedule limit's does not.** Sweeping M at two K values separates them with no
kernel change at all -- if efficiency falls at the same M for both, X capacity is
not the mechanism and SMEM staging is the wrong next move.

Reported as achieved TFLOPS (2*M*N*K / t), which normalizes the two shapes against
each other and against the 31.3 TFLOPS packed-f16 peak.

There is a second, harder ceiling in the same arithmetic, and it is worth reading
off this table too: the extern loads 32 B of X per row per tile and issues 8
fma.rn.f16x2 = 32 flops, so the kernel is structurally **1 flop per X byte** at
every M. L1/SMEM bandwidth is 128 B/cycle/SM * 80 * 1.53 GHz = 15.7 TB/s, hence
15.7 TFLOPS = 50% of FMA peak even with a perfect L1 hit rate. If the sweep tops
out near 15 TFLOPS rather than 31, SMEM staging cannot pay either -- SMEM has the
same 128 B/cycle -- and the lever is issuing fewer, wider X loads instead.

  scripts/v100.sh run l1 '/usr/bin/python3 -u scripts/ab_gemv_l1_knee.py'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

os.environ.setdefault("TILERL_TARGET", "cuda")
import benchkit as bk  # noqa: E402
from tilerl_kernels import kernels_linear  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

#: (N, K, label). Same N so the weight stream per output column matches; K differs
#: 4x, which moves the predicted L1 knee 4x if capacity is what binds.
SHAPES = [(5120, 5120, "K=5120"), (5120, 1280, "K=1280")]
MS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
BLK, NP = 32, 4
L1_BYTES = 128 * 1024
FMA_PEAK = 31.3  # TFLOPS, 80 SM * 64 cores * 2 flop * 1.53 GHz * 2 (half2)
L1_BW = 15.7  # TB/s, 128 B/cycle/SM * 80 SM * 1.53 GHz -- and at 1 flop/X-byte,
#: also the TFLOPS ceiling, i.e. 50% of FMA_PEAK is unreachable by construction.
#: A number above either peak means the instrument is wrong, not the kernel fast:
#: bk.timeit returns MILLISECONDS, and reading it as us gave 5687 TFLOPS (182x
#: V100's peak) on the first run of this script.


def main() -> None:
    be = get_backend()
    dev = be.device
    for N, K, label in SHAPES:
        knee = L1_BYTES / (2 * K)
        print(f"\n# {label}  N={N}  predicted L1 knee at M={knee:.1f} "
              f"(X = 2*M*{K} B vs {L1_BYTES // 1024} KB)")
        print(f"{'M':>4} {'X KB':>7} {'us':>9} {'TFLOPS':>8} {'% FMA':>7} {'X TB/s':>8} {'% L1bw':>7}")
        g = torch.Generator(device="cpu").manual_seed(N * 131 + K)
        wq = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, generator=g).to(dev)
        sc = (torch.rand(N, K // BLK, generator=g) * 1.9 + 0.0025).to(dev).half()
        osc = torch.ones(N, device=dev)
        for M in MS:
            k = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, xh=True, sh=True)
            res = torch.zeros(M, N, device=dev)
            x = torch.randn(M, K, generator=g).to(dev).half()
            ms = bk.timeit(lambda: k(x, wq, sc, osc, res, 32, NP, BLK))  # MILLISECONDS
            tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12
            # Every output column re-reads all of X, so X traffic is N*M*K*2 bytes
            # -- 1 flop per X byte, which is why these two columns track each other.
            xtb = N * M * K * 2 / (ms * 1e-3) / 1e12
            print(f"{M:>4} {2 * M * K / 1024:>7.0f} {ms * 1000:>9.1f} {tf:>8.2f} "
                  f"{100 * tf / FMA_PEAK:>6.1f}% {xtb:>8.2f} {100 * xtb / L1_BW:>6.1f}%")

    print("\nRead it this way:")
    print("  knee moves with K  -> X does not fit L1; SMEM staging is the lever")
    print("  same knee for both -> capacity is not the cap; staging X will not pay")
    print(f"  X TB/s near {L1_BW} -> the 1 flop/X-byte structure is the cap, and SMEM")
    print("    has the same 128 B/cycle port, so the lever is fewer X loads per flop")
    print("    (one thread over several output columns), not moving X to SMEM")
    print(f"  anything over {FMA_PEAK} TFLOPS or {L1_BW} TB/s -> the instrument is wrong")


if __name__ == "__main__":
    main()
