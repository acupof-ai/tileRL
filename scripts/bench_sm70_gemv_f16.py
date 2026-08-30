"""sm70 fp16-twiddle GEMV micro-benchmark: latency + MBU at realistic shapes.

Fits in the free GPU memory alongside a running server (WQ for N=K=4864 is
~12 MB). Measures the decode GEMV (M=1) the fp16-twiddle targets.

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70f16 \
    TILERL_TARGET=cuda PYTHONPATH=packages/tilerl-kernels/src:src \
    CUDA_VISIBLE_DEVICES=0 python3 scripts/bench_sm70_gemv_f16.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src"))

import torch  # noqa: E402

from tilerl_kernels import reference  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

bk = get_backend()
assert bk.arch == "sm70", f"sm70 only, got {bk.arch}"
HBM = 900e9  # V100 SXM2 HBM2 bandwidth, bytes/s

torch.manual_seed(0)
g = torch.Generator().manual_seed(1)

print(f"{'N':>6} {'K':>6} {'lat(us)':>10} {'MBU%':>7} {'GB/s':>8}")
for (N, K) in [(4864, 4864), (4864, 13824), (13824, 4864), (4864, 32768)]:
    w = torch.randn(N, K, generator=g) * 0.1
    wq, sc = reference.pack_fp4(w, block=16)
    x = torch.randn(1, K, generator=g)
    wq_d, sc_d, x_d = wq.to(bk.device), sc.to(bk.device), x.to(bk.device)
    # warmup (first call twiddles + compiles)
    for _ in range(5):
        y = bk.linear_fp4(x_d, wq_d, sc_d)
    torch.cuda.synchronize()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        y = bk.linear_fp4(x_d, wq_d, sc_d)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    bytes_ = N * K * 0.75 + K * 4 + N * 4  # WQ + scale(f32/16) + X(f32, once) + Y
    gbs = bytes_ / us / 1e3
    mbu = bytes_ / us / 1e3 / (HBM / 1e9) * 100
    print(f"{N:>6} {K:>6} {us:>10.1f} {mbu:>6.1f}% {gbs:>8.1f}")
