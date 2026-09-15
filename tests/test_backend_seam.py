"""Backend-isolation gate for the dp/verify seam (finding 23).

The framework (src/tilerl, outside the kernels package) must reach torch
distributed and kernel tile constants only through a Backend method or
read-only property, never by importing torch.distributed or a backend private:

* ``_MAX_VERIFY_W``  -> ``Backend.max_verify_width``
* ``backend._dp_pg`` + bare ``torch.distributed.all_gather`` ->
  ``Backend.dp_all_gather``

RefBackend (src/tilerl/testing.py) is the one ALLOWED point: it is itself a
backend implementation, the CPU mirror of the tilelang Backend, so its private
process groups and torch.distributed calls are the seam's other half, not a
leak around it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src" / "tilerl"

#: The reference backend implements the same seam; it is allowed torch.
_ALLOWED = {"testing.py"}


def _framework_files():
    return [p for p in _SRC.rglob("*.py") if p.name not in _ALLOWED]


def test_framework_does_not_touch_backend_privates_or_distributed():
    bad: list[str] = []
    for p in _framework_files():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in ("_MAX_VERIFY_W", "_dp_pg"):
                bad.append(f"{p.name}:{node.lineno} references {node.id}")
            # import torch.distributed / from torch.distributed import ...
            if isinstance(node, ast.Import) and any(
                    a.name == "torch.distributed" for a in node.names):
                bad.append(f"{p.name}:{node.lineno} imports torch.distributed")
            if isinstance(node, ast.ImportFrom) and node.module == "torch.distributed":
                bad.append(f"{p.name}:{node.lineno} imports from torch.distributed")
            # an explicit torch.distributed.<collective> attribute path
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "distributed"
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "torch"):
                bad.append(f"{p.name}:{node.lineno} calls torch.distributed.{node.attr}")
    assert not bad, "framework must use the Backend seam, not:\n" + "\n".join(sorted(bad))


def test_both_backends_expose_the_seam():
    from tilerl_kernels.backend import MAX_VERIFY_W, Backend

    from tilerl.testing import RefBackend

    assert isinstance(MAX_VERIFY_W, int) and MAX_VERIFY_W == 8
    for cls in (Backend, RefBackend):
        assert isinstance(getattr(cls, "max_verify_width"), property)
        assert callable(getattr(cls, "dp_all_gather"))

    ref = RefBackend()
    assert ref.max_verify_width == MAX_VERIFY_W
    assert ref.dp_world == 1  # the CPU test default
    t = torch.arange(4.0)
    out = ref.dp_all_gather(t)
    assert len(out) == 1 and out[0] is t  # world-1 fast path: the same tensor


def test_engine_reads_verify_width_off_the_backend():
    """The width error message must source its bound from the backend, proving
    the private constant is gone from the engine call site."""
    engine_py = (_SRC / "engine.py").read_text()
    assert "_MAX_VERIFY_W" not in engine_py
    assert "backend.max_verify_width" in engine_py


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
