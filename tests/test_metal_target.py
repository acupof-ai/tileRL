"""Metal target gate for tilerl.

Mirrors ``test_gpu_targets`` in test_e2e.py: the Metal target compiles and
runs the same kernel source as CPU (with the metal-specific gemm schedule
the dispatch matrix registers for this arch). Auto-runs on Apple Silicon
with tilelang Metal JIT + torch MPS available; skips otherwise.

Run: TILERL_TARGET=metal uv run pytest tests/test_metal_target.py -v
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest
import torch

from tilerl.config import tiny
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import build_random
from tilerl.train import _training_kv

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


@contextmanager
def _target(name: str):
    """Yield a backend resolved against TILERL_TARGET=name, then restore — a
    stale ``_BACKEND`` left behind by a failed assert poisons every test module
    imported later (no conftest owns the singleton)."""
    from tilerl.ops import backend as backend_mod

    prev = os.environ.get("TILERL_TARGET")
    os.environ["TILERL_TARGET"] = name
    backend_mod._BACKEND = None
    try:
        yield backend_mod.get_backend()
    finally:
        backend_mod._BACKEND = None
        if prev is None:
            os.environ.pop("TILERL_TARGET", None)
        else:
            os.environ["TILERL_TARGET"] = prev


def test_metal_target_resolves():
    """TILERL_TARGET=metal resolves to the metal target on the mps device."""
    with _target("metal") as backend:
        assert backend.target == "metal"
        assert backend.arch == "metal"
        assert backend.device.type == "mps"


def test_metal_rmsnorm():
    """A kernel compiles and runs on Metal with correct results."""
    with _target("metal") as backend:
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float32, device=backend.device)
        w = torch.randn(64, dtype=torch.float32, device=backend.device)
        y = backend.rmsnorm(x, w, eps=1e-6)
        assert y.device.type == "mps"
        rstd = torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
        assert torch.allclose(y, x * rstd * w, atol=1e-4, rtol=1e-4)


def test_metal_gemm():
    """The metal-specific gemm schedule (naive FMA) matches torch.matmul."""
    with _target("metal") as backend:
        torch.manual_seed(0)
        a = torch.randn(8, 16, dtype=torch.float32, device=backend.device)
        w = torch.randn(24, 16, dtype=torch.float32, device=backend.device)
        y = backend.linear(a, w)
        assert y.device.type == "mps"
        assert torch.allclose(y, a @ w.T, atol=1e-3, rtol=1e-3)


_HET_CFG = replace(tiny(), fp4=True)  # tiny() is fp4=False: nothing quantized to gate
_PROMPT = list(range(1, 17))


def _het_run(target: str):
    """Prefill logits + 8 greedy tokens for the fp4 tiny model on one target."""
    with _target(target) as backend:
        model = build_random(_HET_CFG, seed=7)
        ids = np.asarray(_PROMPT, dtype=np.int64)
        kv = _training_kv(model, 1, ids.size, device=backend.device)
        logits = model.forward(ids.reshape(1, -1), np.arange(ids.size), kv, backend)
        engine = build_engine(_HET_CFG, model, backend, num_blocks=8, num_slots=2)
        try:
            rid = engine.submit(_PROMPT, SamplingParams(temperature=0.0, max_new_tokens=8, seed=0))
            done: dict = {}
            for _ in range(128):
                done.update(engine.poll())
                if rid in done:
                    break
                engine.step()
        finally:
            engine.shutdown()
        return logits.float().cpu(), done[rid]


def test_cpu_metal_decode_parity():
    """One fp4 model, two targets, same answer — the heterogeneity gate.

    Greedy decode is the strict half (any per-op divergence past the argmax
    margin flips a token); the logits allclose is the numeric half.
    """
    cpu_logits, cpu_ids = _het_run("cpu")
    metal_logits, metal_ids = _het_run("metal")
    assert cpu_ids == metal_ids, f"cpu {cpu_ids} != metal {metal_ids}"
    assert torch.allclose(cpu_logits, metal_logits, rtol=1e-2, atol=1e-2), (
        (cpu_logits - metal_logits).abs().max()
    )
