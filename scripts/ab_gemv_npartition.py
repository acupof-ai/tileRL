"""Does raising n_partition cut the M=32 GEMV's X re-reads?

At M=32 each block reads all of X[M,K] and the grid is ceildiv(N, n_partition),
so X is re-read N/n_partition times: 2.85 GB against 0.10 GB of weights for
gate_up, 28x. The M=1 path does not have this problem (X is 0.9x W), which is
why the same kernel is at 83% of bandwidth peak there and 24% of FLOP peak here.

Raising n_partition should cut X traffic proportionally. Falsifiable: if the
kernel is issue-bound rather than X-bound, nothing moves.

n_partition is a runtime arg, so this needs no kernel change -- but it IS the
thread-block y-dim, so threads = 32 * n_partition and 32 is the ceiling (1024).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
os.environ.setdefault("TILERL_TARGET", "cuda")
import benchkit as bk
from tilerl_kernels import kernels_linear
from tilerl_kernels.backend import get_backend

SHAPES = [(34816, 5120, 64, "gate_up"), (5120, 17408, 64, "down"),
          (16384, 5120, 48, "qkvz"), (5120, 6144, 48, "gdn out")]
NPARTS = (4, 8, 16, 32)
BLK, M = 32, 32


def main() -> None:
    be = get_backend()
    dev = be.device
    k = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, xh=True, sh=True)
    print(f"# sm70 fp4 GEMV, M={M}: n_partition sweep (threads = 32 x np)")
    print("# X is re-read ceildiv(N,np) times; W once. Predicted: time falls with np.")
    hdr = "".join(f"{f'np={p}':>11}" for p in NPARTS)
    print(f"{'shape':>20} {'W MB':>7} {'X MB @4':>9}{hdr}   {'best':>6}")
    tot = {p: 0.0 for p in NPARTS}
    for N, K, cnt, label in SHAPES:
        # The kernel writes Y[m, bx*np + ni] with no n<N guard, so N must be a
        # multiple of every np under test or the largest reads/writes past the
        # end. Every real N here divides 32; assert rather than pad, so a shape
        # that does not is a visible skip and not silent OOB.
        assert all(N % p == 0 for p in NPARTS), f"{label}: N={N} not a multiple of {NPARTS}"
        g = torch.Generator(device="cpu").manual_seed(N * 131 + K)
        wq = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, generator=g).to(dev)
        sc = (torch.rand(N, K // BLK, generator=g) * 1.9 + 0.0025).to(dev).half()
        osc = torch.ones(N, device=dev)
        res = torch.zeros(M, N, device=dev)
        x = torch.randn(M, K, generator=g).to(dev).half()
        w_mb = (N * K / 2 + N * (K // BLK) * 2) / 1e6
        us, ref = {}, None
        for p in NPARTS:
            f = lambda p=p: k(x, wq, sc, osc, res, 32, p, BLK)
            y = f()
            if ref is None:
                ref = y.clone()
            else:
                # A different n_partition must not change the result.
                assert torch.equal(y, ref), f"{label} np={p}: output changed"
            us[p] = bk.timeit(f)
            tot[p] += us[p] * cnt
        best = min(us, key=us.get)
        row = "".join(f"{us[p] * 1000:>11.1f}" for p in NPARTS)
        print(f"{label + f' {N}x{K}':>20} {w_mb:>7.1f} {(-(-N // 4) * M * K * 2) / 1e6:>9.0f}{row}   "
              f"np={best:<3} {us[NPARTS[0]] / us[best]:.2f}x")
    print(f"\nper-pass total (ms), weighted by launches/token:")
    for p in NPARTS:
        print(f"  np={p:>2}: {tot[p]:>8.1f}   {tot[NPARTS[0]] / tot[p]:>5.2f}x vs np=4")


if __name__ == "__main__":
    main()
