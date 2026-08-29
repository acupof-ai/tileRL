"""sm70 backend fp4 parity across M — run on the V100 (CUDA 12.4 nvcc).

M=1 hits make_linear_fp4_gemv_sm70; M>1 has no sm70 decode/prefill kernel, so
_plan returns None and Backend.linear_fp4 falls through to the generic f32
make_linear_fp4. This pins BOTH paths against reference.linear_fp4 — the peer's
point that the M>1 fallback, though correct by construction, was never measured.

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_c \
    TILERL_TARGET=cuda PYTHONPATH=packages/tilerl-kernels/src:src \
    python scripts/_sm70_backend_parity.py
"""
import sys

import torch

from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend


def run(M, N, K, block=32):
    torch.manual_seed(0)
    w = torch.randn(N, K, dtype=torch.float32) * 0.3
    wq, scale = reference.pack_fp4(w, block=block)
    scale, oscale = reference.renorm_fp4_scale(scale)
    x = torch.randn(M, K, dtype=torch.float32) * 0.5

    ref = reference.linear_fp4(x, wq, scale, oscale)  # [M,N] f32

    be = get_backend()
    y = be.linear_fp4(
        x.to(be.device), wq.to(be.device), scale.to(be.device), oscale=oscale.to(be.device)
    ).float().cpu()

    rel = (y - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    ok = torch.allclose(y, ref, rtol=1e-2, atol=0) or rel < 1e-2
    path = "gemv_sm70" if M == 1 else "generic f32 (fallback)"
    print(f"M={M:2d} N={N} K={K}: rel={rel:.3e} pass={ok}  [{path}]")
    return ok


if __name__ == "__main__":
    print("device", torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))
    ok = True
    for M in (1, 4, 16, 17):
        ok &= run(M, 5120, 5120)
    print("BACKEND PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
