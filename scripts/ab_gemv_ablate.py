"""Price the sm70 GEMV's M=32 gap by ABLATION -- the instrument left after ncu.

Five mechanisms are excluded by A/B (FMA chain, n_partition, L1 capacity, L1
bandwidth, occupancy) and M=32 still sits at 17.6% of FMA peak, 35% of the
L1-bandwidth ceiling, 40% of the issue ceiling. ncu would say where the cycles go
but the pod denies performance counters to non-root (ERR_NVGPUCTRPERM).

Ablation needs no counters. Each variant keeps the instruction count and the load
count IDENTICAL and removes one suspect's cost, so the time delta prices that
suspect directly. Every one returns WRONG NUMBERS -- that is what makes them
measurements rather than candidates, and none can ship.

  abl=1  X_REUSE    every row reads row 0's X. Same 2 loads + 8 FMA per row, but
                    an L1 hit after the first row -> the cost of X's traffic and
                    latency, the thing SMEM staging would have addressed.
  abl=2  NO_SCALE   drop the per-tile widen + scale (HADD2.F32 + FADD/FFMA, 837 of
                    2936 instructions = 28.5% by SASS count) -> the scale tail.
  abl=3  NO_DECODE  skip tl_fp4_decode8_f16, feed raw words to the FMAs -> the fp4
                    dequant (LOP3/SHF/PRMT).

Read the deltas against each other, not against a roofline. If none of the three
moves M=32 much, the cost is in something none of them touch -- the serial
dependence of the m loop on one accumulator register, or the LDG issue rate -- and
that is worth knowing too. If one dominates, it is the lever, and its A/B is a
real kernel change measured against the same threshold discipline.

  scripts/v100.sh run abl '/usr/bin/python3 -u scripts/ab_gemv_ablate.py'
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

#: (N, K, launches/token, label) -- the real 27B decode shapes.
SHAPES = [(34816, 5120, 64, "gate_up"), (5120, 17408, 64, "down"),
          (16384, 5120, 48, "qkvz"), (5120, 6144, 48, "gdn out"),
          (14336, 5120, 16, "qkv"), (5120, 6144, 16, "attn o")]
MS = (1, 8, 32)
#: (id, label, factory kwargs). NCOLS2 is not an `abl` value: it graduated into the
#: shipping `ncols` parameter, so the harness reaches it the same way the dispatch
#: does. Building it via a stale `abl=6` would have silently compiled the BASE
#: kernel and reported 1.00x (errors/2026-09-03-the-ab-measured-abl-not-ncols.md).
ABLATIONS = [(1, "X_REUSE", {"abl": 1}), (2, "NO_SCALE", {"abl": 2}),
             (3, "NO_DECODE", {"abl": 3}), (4, "PIPELINE", {"abl": 4}),
             (5, "SMEM", {"abl": 5}), (6, "NCOLS2", {"ncols": 2})]
#: 4, 5 and 6 are CORRECT variants, so they are both discriminators and candidate
#: fixes; the harness checks their relerr. Measured so far: PIPELINE 0.99x (reorder
#: -- the cost is throughput, not latency), SMEM 0.67x (swaps LDG for LDS and keeps
#: the 1 load : 4 FMA ratio). NCOLS2 is the only variant that RAISES the ratio, to
#: 1 : 16.
#: NCOLS2 THRESHOLD, committed before reading M=32: accept at >=1.25x with no M=1
#: regression. Below that, the loads-per-FMA family is closed -- this is its fourth
#: attempt and a ratio change that does not pay leaves nothing else in it.
#: Confirm from the cubin that HFMA2-per-load actually doubled before believing any
#: timing: SMEM taught that a variant can do what it claims and still lose.
PIPE_ACCEPT = 1.15
SMEM_ACCEPT = 1.30
NCOLS_ACCEPT = 1.25
CORRECT_ABL = (4, 5, 6)  # variants that keep the arithmetic: their relerr is checked
BLK, NP = 32, 4
FMA_PEAK = 31.3


def main() -> None:
    be = get_backend()
    dev = be.device
    print("# sm70 fp4 GEMV ablations -- ALL RETURN WRONG NUMBERS, timing only")
    print("# gain > 1 means the ablated work costs that much; 1.00 means it is free")
    tot: dict[tuple[int, int], float] = {}
    for M in MS:
        base = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, xh=True, sh=True)
        cands = [(a, lbl, kernels_linear.make_linear_fp4_gemv_sm70_m(
            be.target, M=M, xh=True, sh=True, **kw)) for a, lbl, kw in ABLATIONS]
        print(f"\n{'shape':>20} {'M':>3} {'base us':>9} " +
              " ".join(f"{lbl:>10}" for _, lbl, _ in ABLATIONS))
        for N, K, cnt, label in SHAPES:
            g = torch.Generator(device="cpu").manual_seed(N * 131 + K)
            wq = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, generator=g).to(dev)
            sc = (torch.rand(N, K // BLK, generator=g) * 1.9 + 0.0025).to(dev).half()
            osc = torch.ones(N, device=dev)
            res = torch.zeros(M, N, device=dev)
            x = torch.randn(M, K, generator=g).to(dev).half()
            u0 = bk.timeit(lambda: base(x, wq, sc, osc, res, 32, NP, BLK))
            gains = []
            for a, _, k in cands:
                # abl=4 and 5 keep the arithmetic, so their output must match:
                # check it, because a variant that changes numbers is a bug.
                if a in CORRECT_ABL:
                    rel = bk.relerr(k(x, wq, sc, osc, res, 32, NP, BLK),
                                    base(x, wq, sc, osc, res, 32, NP, BLK))
                    if rel:
                        print(f"  !! abl={a} relerr {rel:.2e} at {label} M={M} "
                              f"-- must keep the arithmetic")
                u = bk.timeit(lambda k=k: k(x, wq, sc, osc, res, 32, NP, BLK))
                gains.append(u0 / u)
                tot[(M, a)] = tot.get((M, a), 0.0) + u * cnt
            tot[(M, 0)] = tot.get((M, 0), 0.0) + u0 * cnt
            print(f"{label + f' {N}x{K}':>20} {M:>3} {u0 * 1000:>9.1f} " +
                  " ".join(f"{x_:>9.2f}x" for x_ in gains))

    print("\nper-pass total over these shapes (ms), weighted by launches/token:")
    for M in MS:
        b = tot[(M, 0)]
        row = "  ".join(f"{lbl} {b / tot[(M, a)]:.2f}x" for a, lbl, _ in ABLATIONS)
        print(f"  M={M:>2}: base {b:>7.1f}   {row}")

    pipe = tot[(32, 0)] / tot[(32, 4)]
    smem = tot[(32, 0)] / tot[(32, 5)]
    ncols = tot[(32, 0)] / tot[(32, 6)]
    reuse = tot[(32, 0)] / tot[(32, 1)]
    print(f"\nX_REUSE bounds X-related cost at {reuse:.2f}x (it also deletes 85% of the")
    print("LDGs -- a loop-invariant address is hoistable -- so read it as a ceiling).")
    for lbl, g, acc in (("PIPELINE", pipe, PIPE_ACCEPT), ("SMEM", smem, SMEM_ACCEPT),
                        ("NCOLS2", ncols, NCOLS_ACCEPT)):
        print(f"  {lbl:<9} {g:.2f}x, {100 * (g - 1) / (reuse - 1):>5.1f}% of that headroom"
              f"   threshold {acc}x -> {'ACCEPT' if g >= acc else 'reject'}")
    print("\nPIPELINE ~1.0x: the cost is throughput, not latency (reorder changes nothing).")
    print("SMEM 0.67x: swapping LDG for LDS keeps 1 load : 4 FMA, and adds barriers.")
    print("NCOLS2 is the only variant that raises the ratio (to 1 : 16). If it does")
    print("not pay, the loads-per-FMA family is closed -- read HFMA2-per-load off the")
    print("cubin first, because a variant can do what it claims and still lose.")


if __name__ == "__main__":
    main()
