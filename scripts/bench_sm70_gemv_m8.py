"""M=8 micro-benchmark: time the sm70 M-row GEMV directly (no engine)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src"))

import torch  # noqa: E402

from tilerl_kernels import reference  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

bk = get_backend()
g = torch.Generator().manual_seed(1)
for (N, K) in [(17408, 5120), (5120, 17408), (4864, 4864)]:
    w = torch.randn(N, K, generator=g) * 0.1
    wq, sc = reference.pack_fp4(w, block=16)
    wq = reference.twiddle_fp4_f16(wq)
    wq_d, sc_d = wq.to(bk.device), sc.to(bk.device)
    x = torch.randn(8, K, generator=g)
    x_d = x.to(bk.device)
    for _ in range(3):
        y = bk.linear_fp4(x_d, wq_d, sc_d)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        y = bk.linear_fp4(x_d, wq_d, sc_d)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / 20 * 1e6
    bytes_ = N * K * 0.75 + 8 * K * 4 + 8 * N * 4
    print(f"M=8 N={N} K={K}  {us:.1f} us  {bytes_/us/1e3:.1f} GB/s  {bytes_/us/1e3/900*100:.1f}% MBU")
