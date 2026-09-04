"""Does an f16 scale plane speed up the sm70 GEMV, and by how much?

Two questions in one run, because the pod takes one job at a time:

1. The f16-scale A/B. At the checkpoint's real block of 32, the f32 scale plane
   is 3.20 GB of the 16.04 GB streamed per dense token (20%); f16 halves it and
   moves the roofline 56.1 -> 62.3 tok/s. This measures whether the kernel
   converts those bytes into time.
2. Per-shape achieved GB/s against the 900 GB/s HBM peak. Which shapes sit below
   the average says whether the GEMV's gap to its own byte roofline is a tail
   (one huge-N shape) or uniform (occupancy / issue rate).

Shapes are DERIVED from param_specs + _projection_groups (see ``shapes``), not
written down: serving fuses projections, and the hand-written table this replaced
had three wrong rows while still reproducing the byte total.

The absolute times carry an eager-launch floor (~60 us regardless of shape) that
the graph-captured decode path does not pay, and that floor dominates the small-N
rows — read the RATIO here and take per-shape cost from prof_decode_budget.py.
In the captured graph the GEMV runs at 746 GB/s, 83% of peak; this harness read
58% off the same kernel (errors/2026-09-02-per-shape-gap-was-a-wrong-shape-
table.md).

  scripts/v100.sh run sc '/usr/bin/python3 -u scripts/ab_scale_f16.py'
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


def shapes() -> list[tuple[int, int, int, str]]:
    """(N, K, launches/token, label) for every fp4 GEMV a dense token issues.

    Derived from param_specs + _projection_groups rather than restated: serving
    runs with fuse_projections, so a hand-written table is wrong the moment a
    group changes. The hand-written one was wrong in three rows at once and
    still passed a byte-total assert (errors/2026-09-02-per-shape-gap-was-a-
    wrong-shape-table.md).
    """
    from collections import Counter

    from tilerl.config import qwen38_27b
    from tilerl.model import _projection_groups, param_specs

    cfg = qwen38_27b()
    specs = param_specs(cfg)
    fused = {}  # member key -> its group's key, so members are counted once
    for i in range(cfg.num_layers):
        for key, group in _projection_groups(cfg, i):
            rows = sum(specs[m][0] for m in group)
            specs[key] = (rows, specs[group[0]][1])
            fused.update(dict.fromkeys(group, key))

    # Only 2D fp4 linears stream nibbles; norms/conv1d/embed are not GEMVs.
    skip = ("norm", "conv1d", "dt_bias", "a_log", "embed_tokens")
    hits = Counter()
    for key, shape in specs.items():
        if key in fused or len(shape) != 2 or key.endswith(skip):
            continue
        hits[(shape[0], shape[1], key.split(".")[-1])] += 1
    return sorted(
        ((n, k, c, lab) for (n, k, lab), c in hits.items()), key=lambda r: -r[0] * r[1] * r[2]
    )


BLK = 32  # the served checkpoint's scale block
M = 1  # decode


def main() -> None:
    be = get_backend()
    dev = be.device
    k32 = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, xh=True)
    k16 = kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, xh=True, sh=True)

    print(f"# sm70 fp4 GEMV, M={M}, block={BLK}: f32 vs f16 block scales")
    print(
        f"{'shape':>22} {'x':>4} {'GB/tok':>7} {'f32 us':>8} {'f16 us':>8} "
        f"{'gain':>6} {'GB/s':>6} {'roof%':>6} {'relerr':>9}"
    )
    t32 = t16 = b32 = b16 = nib_tot = 0.0
    SHAPES = shapes()
    for N, K, cnt, label in SHAPES:
        g = torch.Generator(device="cpu").manual_seed(N * 131 + K)
        wq = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, generator=g).to(dev)
        # Match the served magnitudes (2.5e-3..1.99) so f16 is exercised in its
        # real range, not near a denormal it never sees.
        s32 = (torch.rand(N, K // BLK, generator=g) * 1.9 + 0.0025).to(dev)
        s16 = s32.half()
        osc = torch.ones(N, device=dev)
        res = torch.zeros(M, N, device=dev)
        x = torch.randn(M, K, generator=g).to(dev).half()

        f32 = lambda: k32(x, wq, s32, osc, res, 32, 4, BLK)  # noqa: E731
        f16 = lambda: k16(x, wq, s16, osc, res, 32, 4, BLK)  # noqa: E731
        y32, y16 = f32(), f16()
        # bk.relerr (max abs error over ref's abs-max), NOT a clamped per-element
        # ratio: dividing by y.abs().clamp(min=1e-3) grows with N — 0.21 at
        # N=1024, 21.8 at 248320, because more rows means more near-zero outputs
        # — and read as a broken kernel when the real error was 1.9e-04.
        rel = bk.relerr(y16, y32)
        u32, u16 = bk.timeit(f32), bk.timeit(f16)

        nib, spl = N * K / 2, N * (K // BLK)
        by32, by16 = nib + spl * 4, nib + spl * 2
        t32 += u32 * cnt
        t16 += u16 * cnt
        b32 += by32 * cnt
        b16 += by16 * cnt
        nib_tot += nib * cnt
        gbs = by16 / u16 / 1e6  # the f16 path is the candidate
        print(
            f"{label + f' {N}x{K}':>22} {cnt:>4} {by32 * cnt / 1e9:>7.2f} "
            f"{u32 * 1000:>8.1f} {u16 * 1000:>8.1f} {u32 / u16:>5.2f}x "
            f"{gbs:>6.0f} {100 * gbs / 900:>5.0f}% {rel:>9.2e}"
        )

    # Assert the DISTRIBUTION, not just the total: the per-shape rows are the
    # output, and a table can reproduce 12.81 GB while getting three rows wrong
    # (the errors cancelled — attn o 168 MB short, gdn out 252 MB long). The
    # launch count is what caught it: 401 written vs 305 in the profiler.
    launches = sum(c for _, _, c, _ in SHAPES)
    assert abs(nib_tot / 1e9 - 12.81) < 0.02, f"nibbles {nib_tot / 1e9:.2f} GB != 12.81"
    assert launches == 305, f"{launches} GEMV launches/token, profiler measures 305"
    print(
        f"\nper-token GEMV: {t32:.2f} -> {t16:.2f} ms ({t32 / t16:.2f}x), "
        f"weights {b32 / 1e9:.2f} -> {b16 / 1e9:.2f} GB"
    )
    print(f"  f32 {b32 / t32 / 1e6:.0f} GB/s, f16 {b16 / t16 / 1e6:.0f} GB/s of 900 peak")


if __name__ == "__main__":
    main()
