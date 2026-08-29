"""sm70 fp4 GEMV parity under WIDE oscale — the arm the random-weight test misses.

The shipped parity used randn*0.3: rows same-scale, oscale ~constant. Real
27B weights span orders of magnitude across output rows, so renorm_fp4_scale
puts a wide dynamic range into oscale. If the GEMV mis-applies oscale, uniform
rows hide it and wide rows expose it (near-uniform logits -> id 220).

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70b \
    TILERL_TARGET=cuda PYTHONPATH=packages/tilerl-kernels/src:src \
    python scripts/_sm70_oscale_parity.py
"""
import sys

import torch

from tilerl_kernels import kernels_linear, reference


def _round_up(x, m):
    return (x + m - 1) // m * m


def run(N, K, row_scale_spread, block=32, reduce_thread=32, n_partition=4):
    torch.manual_seed(0)
    dev = "cuda"
    # Per-row magnitude spanning `row_scale_spread` orders of magnitude — what a
    # real weight matrix looks like, unlike randn*0.3.
    exps = torch.linspace(-row_scale_spread, row_scale_spread, N)
    w = torch.randn(N, K, dtype=torch.float32) * (10.0 ** exps)[:, None]
    wq, scale = reference.pack_fp4(w, block=block)
    scale, oscale = reference.renorm_fp4_scale(scale)
    print(f"  oscale range: {oscale.min():.3e} .. {oscale.max():.3e} (span {oscale.max()/oscale.min():.1e})")
    x = torch.randn(1, K, dtype=torch.float32) * 0.5

    ref = reference.linear_fp4(x, wq, scale, oscale)  # [1,N] f32

    Np = _round_up(N, n_partition)
    Kp = _round_up(K, reduce_thread * 16)
    xq = torch.zeros(1, Kp, dtype=torch.float32, device=dev)
    xq[0, :K] = x[0]
    wqp = torch.zeros(Np, Kp // 2, dtype=torch.uint8, device=dev)
    wqp[:N, : K // 2] = wq.to(dev)
    scp = torch.zeros(Np, Kp // block, dtype=torch.float32, device=dev)
    scp[:N, : K // block] = scale.to(dev)
    oscp = torch.zeros(Np, dtype=torch.float32, device=dev)
    oscp[:N] = oscale.to(dev)
    res = torch.zeros(1, Np, dtype=torch.float32, device=dev)

    k = kernels_linear.make_linear_fp4_gemv_sm70("cuda")
    y = k(xq, wqp, scp, oscp, res, reduce_thread, n_partition, block)[:, :N].cpu()

    rel = (y - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    ok = rel < 1e-2
    print(f"N={N} K={K} spread={row_scale_spread}: rel={rel:.3e} pass={ok}")
    return ok


if __name__ == "__main__":
    print("device", torch.cuda.get_device_capability())
    ok = True
    for spread in (0, 2, 4, 6):  # 0 = the shipped test's regime; 6 = real-weight-like
        ok &= run(5120, 5120, spread)
    print("WIDE-OSCALE PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
