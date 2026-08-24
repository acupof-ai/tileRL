"""Metal target gate for tilerl.

Mirrors ``test_gpu_targets`` in test_e2e.py: the Metal target compiles and
runs the same kernel source as CPU (with the metal-specific gemm schedule
the dispatch matrix registers for this arch). Auto-runs on Apple Silicon
with tilelang Metal JIT + torch MPS available; skips otherwise.

Run: TILERL_TARGET=metal uv run pytest tests/test_metal_target.py -v
"""

from __future__ import annotations

import os

import pytest
import torch

# Skip conditions, evaluated at import time like the CUDA check in
# test_e2e.test_gpu_targets.
_MPS_BUILT = torch.backends.mps.is_built()
_MPS_AVAILABLE = torch.backends.mps.is_available()
# is_available() is True on CI macos runners, but the VM has no GPU
# entitlement: even a 1-byte MPS allocation raises OOM. Probe allocation.
_MPS_USABLE = False
if _MPS_AVAILABLE:
    try:
        torch.empty(1, device="mps")
        _MPS_USABLE = True
    except RuntimeError:
        pass

# Metal JIT availability: tilelang 0.1.13 ships the metal pipeline on macOS
# arm64 wheels; probe tilelang's own check rather than the platform alone, so
# a wheel built without Metal still skips cleanly.
try:  # pragma: no cover - import probe
    from tilelang.metal.target import check_metal_availability

    _METAL_JIT = bool(check_metal_availability())
except Exception:  # noqa: BLE001 - any probe failure -> skip
    _METAL_JIT = False

pytestmark = pytest.mark.skipif(
    not (_MPS_USABLE and _METAL_JIT),
    reason=f"Metal target unavailable (mps_built={_MPS_BUILT}, "
    f"mps_available={_MPS_AVAILABLE}, mps_usable={_MPS_USABLE}, metal_jit={_METAL_JIT})",
)


def _reset_backend():
    """Force target re-resolution against TILERL_TARGET."""
    from tilerl.ops import backend as backend_mod

    backend_mod._BACKEND = None
    return backend_mod


def test_metal_target_resolves():
    """TILERL_TARGET=metal resolves to the metal target on the mps device."""
    prev = os.environ.get("TILERL_TARGET")
    os.environ["TILERL_TARGET"] = "metal"
    backend_mod = _reset_backend()
    try:
        backend = backend_mod.get_backend()
        assert backend.target == "metal"
        assert backend.arch == "metal"
        assert backend.device.type == "mps"
    finally:
        backend_mod._BACKEND = None
        if prev is None:
            os.environ.pop("TILERL_TARGET", None)
        else:
            os.environ["TILERL_TARGET"] = prev


def test_metal_rmsnorm():
    """A kernel compiles and runs on Metal with correct results."""
    prev = os.environ.get("TILERL_TARGET")
    os.environ["TILERL_TARGET"] = "metal"
    backend_mod = _reset_backend()
    try:
        backend = backend_mod.get_backend()
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float32, device=backend.device)
        w = torch.randn(64, dtype=torch.float32, device=backend.device)
        y = backend.rmsnorm(x, w, eps=1e-6)
        assert y.device.type == "mps"
        rstd = torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
        assert torch.allclose(y, x * rstd * w, atol=1e-4, rtol=1e-4)
    finally:
        backend_mod._BACKEND = None
        if prev is None:
            os.environ.pop("TILERL_TARGET", None)
        else:
            os.environ["TILERL_TARGET"] = prev


def test_metal_gemm():
    """The metal-specific gemm schedule (naive FMA) matches torch.matmul."""
    prev = os.environ.get("TILERL_TARGET")
    os.environ["TILERL_TARGET"] = "metal"
    backend_mod = _reset_backend()
    try:
        backend = backend_mod.get_backend()
        torch.manual_seed(0)
        a = torch.randn(8, 16, dtype=torch.float32, device=backend.device)
        w = torch.randn(24, 16, dtype=torch.float32, device=backend.device)
        y = backend.linear(a, w)
        assert y.device.type == "mps"
        assert torch.allclose(y, a @ w.T, atol=1e-3, rtol=1e-3)
    finally:
        backend_mod._BACKEND = None
        if prev is None:
            os.environ.pop("TILERL_TARGET", None)
        else:
            os.environ["TILERL_TARGET"] = prev
