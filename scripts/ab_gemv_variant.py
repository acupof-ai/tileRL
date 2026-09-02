"""A/B one sm70 GEMV variant against the shipped kernel at M=1, 8 and 32.

The M=32 path runs at 17.6% of packed-f16 FLOP peak while the SAME kernel is at
83% of bandwidth peak at M=1, so a candidate fix has to be measured at all three:
M=1 is decode and must not regress, M=8 is a speculative verify, M=32 is what
prefill runs. Five candidates have died here:

- n_partition (errors/2026-09-02-npartition-is-not-the-m32-lever.md), 1.01x
- splitting the 8-deep FMA accumulator chain
  (errors/2026-09-02-fma-chain-is-not-the-m32-cap.md), 1.01x
- L1 capacity / SMEM staging: no K-dependent knee, never written
  (errors/2026-09-03-smem-staging-rejected-one-flop-per-x-byte.md)
- L1 bandwidth: X reads at 35% of the port, same entry
- occupancy via min_blocks=4: 4x the warps, 1.00x at M=32
  (errors/2026-09-03-occupancy-is-not-the-gemv-cap.md)

Ceilings, all measured, none binding: FMA peak 31.3 TFLOPS (17.6% of it), L1
bandwidth 15.7 (35%), instruction issue 13.6 at 2.29 inst/HFMA2 (40%). Every
mechanism nameable from static analysis is excluded, so the next attempt needs a
different INSTRUMENT rather than another hypothesis -- ncu, which this pod denies
to non-root (ERR_NVGPUCTRPERM, scripts/ncu_gemv_m32.py). `scripts/sass_gemv_mix.py`
is what static analysis can still reach: registers, spills, opcode mix.

To use: add a keyword flag to make_linear_fp4_gemv_sm70_m, name it in VARIANT,
and run. With VARIANT empty this compares the shipped kernel against itself,
which is the run-to-run noise floor -- worth having, since the dead candidates
came in inside 2% and that only means something against a known floor.

Commit the accept/reject threshold BEFORE reading the M=32 number. Three attempts
have landed at 1.00-1.01x, and "basically break-even, maybe ship it" is easy to
talk yourself into afterwards.

  scripts/v100.sh run sp '/usr/bin/python3 -u scripts/ab_gemv_variant.py'
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

SHAPES = [(34816, 5120, 64, "gate_up"), (5120, 17408, 64, "down"),
          (16384, 5120, 48, "qkvz"), (5120, 6144, 48, "gdn out"),
          (14336, 5120, 16, "qkv"), (5120, 6144, 16, "attn o")]
#: M=1 is decode and must not regress; 8 is a spec verify; 32 is prefill.
MS = (1, 8, 32)
#: Factory kwargs for the candidate. Empty = shipped vs itself = the noise floor.
#: min_blocks=4 was measured and REJECTED at 1.00x (M=32), 0.84x (M=8) --
#: errors/2026-09-03-occupancy-is-not-the-gemv-cap.md. Five mechanisms are now
#: excluded: FMA chain, n_partition, L1 capacity, L1 bandwidth, occupancy.
VARIANT: dict[str, object] = {}
BLK, NP = 32, 4


def main() -> None:
    be = get_backend()
    dev = be.device
    base = dict(xh=True, sh=True)
    what = ", ".join(f"{k}={v}" for k, v in VARIANT.items()) or "NOISE FLOOR (self vs self)"
    print(f"# sm70 fp4 GEMV: shipped vs {what}")
    print(f"{'shape':>20} {'M':>3} {'base us':>9} {'cand us':>9} {'gain':>7} {'relerr':>9}")
    tot: dict[int, list[float]] = {m: [0.0, 0.0] for m in MS}
    for M in MS:
        k0 = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, **base)
        k1 = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, **base, **VARIANT)
        for N, K, cnt, label in SHAPES:
            g = torch.Generator(device="cpu").manual_seed(N * 131 + K)
            wq = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, generator=g).to(dev)
            sc = (torch.rand(N, K // BLK, generator=g) * 1.9 + 0.0025).to(dev).half()
            osc = torch.ones(N, device=dev)
            res = torch.zeros(M, N, device=dev)
            x = torch.randn(M, K, generator=g).to(dev).half()
            f0 = lambda: k0(x, wq, sc, osc, res, 32, NP, BLK)  # noqa: E731
            f1 = lambda: k1(x, wq, sc, osc, res, 32, NP, BLK)  # noqa: E731
            rel = bk.relerr(f1(), f0())
            u0, u1 = bk.timeit(f0), bk.timeit(f1)
            tot[M][0] += u0 * cnt
            tot[M][1] += u1 * cnt
            print(f"{label + f' {N}x{K}':>20} {M:>3} {u0 * 1000:>9.1f} {u1 * 1000:>9.1f} "
                  f"{u0 / u1:>6.2f}x {rel:>9.2e}")
    print("\nper-pass total over these shapes (ms), weighted by launches/token:")
    for M in MS:
        a, b = tot[M]
        print(f"  M={M:>2}: {a:>8.1f} -> {b:>8.1f}   {a / b:>5.2f}x")
    print("\nM=32 is the discriminating row: at M=1 the tile does one row per weight")
    print("load, so it is bandwidth-bound and latency work cannot show up there.")


if __name__ == "__main__":
    main()
