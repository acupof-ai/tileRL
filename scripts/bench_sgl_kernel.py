"""sgl-kernel fp4/fp8 GEMM on tileRL's decode shapes: external baseline for
scripts/bench_gemv_micro.py. PENDING-REMOTE (CUDA-only wheel, never run here).
Each side is timed in its own byte convention (tileRL fp4 0.5625 B/elem,
sgl fp4 ~0.5625, fp8 1); never average the two.
    uv add --dev sgl-kernel && uv run python scripts/bench_sgl_kernel.py
"""

from __future__ import annotations

import sys

SHAPES = [  # (name, N, K) — the kill-criterion shapes
    ("gate_up", 34816, 5120),
    ("down", 5120, 17408),
]
M = 1
ITERS = 50
WARMUP = 10


def _have_sgl_kernel():
    try:
        import sgl_kernel  # noqa: F401

        return True
    except ImportError:
        return False


def _candidate_fp4_ops():  # the fp4 entry point has moved across releases

    cands = []
    for path in (
        "sgl_kernel.ops.fp4_scaled_mm",
        "sgl_kernel.ops.cutlass_scaled_mm",
        "sgl_kernel.ops.fp4_gemm",
    ):
        mod, _, attr = path.rpartition(".")
        try:
            fn = getattr(__import__(mod, fromlist=[attr]), attr)
            cands.append((path, fn))
        except (ImportError, AttributeError):
            pass
    return cands


def bench_fp8():
    import torch
    from sgl_kernel.ops import cutlass_scaled_mm

    print(f"\n== fp8 (cutlass_scaled_mm, M={M}) ==")
    for name, n, k in SHAPES:
        a = torch.randn(M, k, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
        sa = torch.ones(M, device="cuda", dtype=torch.float32)
        sb = torch.ones(n, device="cuda", dtype=torch.float32)
        for _ in range(WARMUP):
            cutlass_scaled_mm(a, b, sa, sb, torch.bfloat16)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            cutlass_scaled_mm(a, b, sa, sb, torch.bfloat16)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000 / ITERS
        bytes_ = M * n * k * 1.0  # fp8 = 1 B/elem
        print(f"  {name:8s} {us:8.1f} µs   {bytes_ / us / 1e6:5.2f} TB/s (1 B/elem)")


def bench_fp4():

    cands = _candidate_fp4_ops()
    if not cands:
        print("\n== fp4: no known entry point found ==")
        print('   inspect: python -c "import sgl_kernel.ops as o; print([x for x in dir(o)])"')
        print("   then add the entry to _candidate_fp4_ops in this script.")
        return
    path, fn = cands[0]
    print(f"\n== fp4 ({path}, M={M}) ==")
    print("   NOTE: the weight layout below is a best-effort guess (packed uint8 +")
    print("   e4m3 scales). If the call signature differs, adapt this arm — the")
    print("   shapes and timing loop are the part that must not change.")
    # ponytail: fp4 arm unwritten, wire the chosen entry point with bench_fp8's event loop on the pod
    print("   fp4 arm not wired — see NOTE. The fp8 arm above is the stable baseline.")


def main() -> int:
    if not _have_sgl_kernel():
        print("sgl_kernel not importable. On the pod: uv add --dev sgl-kernel")
        print("(CUDA-only — this script is pending-remote by design.)")
        return 0
    import torch

    print(f"torch {torch.__version__} · {torch.cuda.get_device_name(0)}")
    bench_fp8()
    bench_fp4()
    print("\nCompare against: scripts/bench_gemv_micro.py (same shapes, same process")
    print("convention) and the Marlin row in docs/experience/2026-08-27-tilelang-vs-native.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
