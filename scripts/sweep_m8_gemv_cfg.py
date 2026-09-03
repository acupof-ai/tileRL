"""Sweep the sm70 M-row fp4 GEMV's compile-time knobs against occupancy.

ncu says the kernel is pinned at 255 registers/thread with 12.2% occupancy and
6.8% DRAM at every shape — starved of parallelism, not bandwidth. M, GROUP,
reduce_thread and n_partition are all factory/call arguments, so the register
live-set can be cut without touching the kernel source.

M is the row count a speculative verify needs: W=2..8. The shipped config is
M=8, GROUP=4, reduce_thread=32. A verify tick runs 497 of these calls, so
us/row here multiplies by ~497 into the tick.

  TILERL_TARGET=cuda python3 scripts/sweep_m8_gemv_cfg.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src")
)

import torch
from tilerl_kernels import kernels_linear, reference
from tilerl_kernels.backend import _pad2d, get_backend

# gate/up is the widest projection and the one the verify tick runs most.
N, K = 17408, 5120
BLK = 16


def bench(fn, n: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8, help="live rows (verify width)")
    args = ap.parse_args()
    backend = get_backend()
    dev = backend.device
    tgt = backend.target

    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.02
    wq, sc = reference.pack_fp4(w)
    sc, osc = reference.renorm_fp4_scale(sc)
    p = backend.materialize({"w.wq": wq, "w.scale": sc, "w.oscale": osc})
    _, _, Np, Kp, _, bN = backend._plan("linear_fp4", 1, N, K)
    wq1 = _pad2d(p["w.wq"], Np, Kp // 2)
    sc1 = _pad2d(p["w.scale"], Np, Kp // BLK)
    osc1 = backend._const_f32(p["w.oscale"], Np)

    ref = None
    print(f"N={N} K={K} rows={args.rows}  shipped: M=8 GROUP=4 rt=32 npart={bN}")
    print(f"{'M':>3s} {'GRP':>4s} {'rt':>4s} {'npart':>6s} {'us':>9s} {'us/row':>8s} "
          f"{'vs ship':>8s} {'maxerr':>9s}")
    grid = itertools.product((8, 4), (4, 2, 1), (32, 16), (bN, 8))
    for M, G, rt, npart in grid:
        if args.rows > M:
            continue
        try:
            k = kernels_linear.make_linear_fp4_gemv_sm70_m(tgt, M=M, GROUP=G)
            x = torch.randn(M, Kp, device=dev)
            res = torch.zeros(M, Np, device=dev)
            y = k(x, wq1, sc1, osc1, res, rt, npart, BLK)
            us = bench(lambda: k(x, wq1, sc1, osc1, res, rt, npart, BLK))
        except Exception as exc:  # a config the kernel rejects is data, not a crash
            print(f"{M:3d} {G:4d} {rt:4d} {npart:6d}   {type(exc).__name__}: {str(exc)[:40]}")
            continue
        got = y[: args.rows, :N].float()
        if ref is None:
            ref = got.clone()
            err = 0.0
        else:
            err = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
        star = " *" if err < 1e-2 else " BAD"
        print(f"{M:3d} {G:4d} {rt:4d} {npart:6d} {us:9.1f} {us / args.rows:8.1f} "
              f"{us / 1019.3:8.2f} {err:9.2e}{star}")

    print("\nshipped M=8 measured 1019.3 us here; a verify tick runs ~497 such calls")


if __name__ == "__main__":
    main()
