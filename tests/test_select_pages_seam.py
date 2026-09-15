"""Backend seam for sparse page selection (finding 20, CPU half).

sparse_engine used to import select_pages straight from
tilerl_kernels.reference; it now calls backend.select_pages. These gates pin:

* isolation: the framework (src/tilerl, RefBackend excepted -- it is the CPU
  backend implementation) no longer imports reference.select_pages in any
  binding shape (bare name / attribute / import-as);
* parity through the seam: a CPU-tiny sparse engine at full k reproduces the
  dense engine token-for-token (its selection now runs via RefBackend.select_pages);
* same source: Backend.select_pages and RefBackend.select_pages both return
  reference.select_pages's exact result, including the n_window forced set.

The CUDA Backend delegates to the torch reference pending a real select kernel.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
import torch
from tilerl_kernels import reference
from tilerl_kernels.backend import Backend

from tilerl.testing import RefBackend

_SRC = Path(__file__).resolve().parent.parent / "src" / "tilerl"
_ALLOWED = {"testing.py"}  # RefBackend is the CPU backend, the one import site


def _violations(src: str) -> list[str]:
    """Every binding shape of reference.select_pages is a violation; the one
    allowed call is `<...>.backend.select_pages` (the seam).

    * ``from ...reference import select_pages [as x]`` -- caught at the import,
      since the as-bound bare name is then untraceable;
    * ``<recv>.select_pages`` whose receiver is not a ``backend`` -- catches
      ``reference.select_pages``, an aliased ``_r.select_pages``, and the fully
      qualified ``tilerl_kernels.reference.select_pages`` alike.
    """
    bad: list[str] = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.endswith("reference"):
                for a in node.names:
                    if a.name == "select_pages":
                        bad.append(f"imports select_pages as {a.asname or a.name}")
        if isinstance(node, ast.Attribute) and node.attr == "select_pages":
            recv = ast.unparse(node.value)
            if recv != "backend" and not recv.endswith(".backend"):
                bad.append(f"{recv}.select_pages is not the backend seam")
    return bad


def test_framework_reaches_select_pages_only_via_the_backend():
    bad: list[str] = []
    for p in _SRC.rglob("*.py"):
        if p.name in _ALLOWED or "select_pages" not in p.read_text():
            continue
        for v in _violations(p.read_text()):
            bad.append(f"{p.name}: {v}")
    assert not bad, "framework must call backend.select_pages:\n" + "\n".join(sorted(bad))


@pytest.mark.parametrize("mutant", [
    "from tilerl_kernels.reference import select_pages\n",
    "from tilerl_kernels.reference import select_pages as sp\n",
    "import tilerl_kernels.reference as _r\nz = _r.select_pages\n",
    "from tilerl_kernels import reference\nz = reference.select_pages\n",
    "import tilerl_kernels\nz = tilerl_kernels.reference.select_pages\n",
])
def test_every_reference_binding_shape_is_red(mutant):
    """The four ways to reach the reference directly must all be violations; the
    last two differ only by the import binding name."""
    assert _violations(mutant), f"gate missed:\n{mutant}"


def test_backend_receiver_is_not_a_violation():
    assert _violations("sel = self.backend.select_pages(a, b, n_window=w)\n") == []
    assert _violations("sel = backend.select_pages(a, b)\n") == []


def _case(k_pages: int, n_window: int):
    # b=1, layers=1: the shape SparseForward.fill always calls select_pages in.
    torch.manual_seed(0)
    pages = 6
    block_table = torch.randint(1, 50, (1, pages))
    n_pages = torch.tensor([pages])
    scores = torch.randn(1, 1, pages)
    want = reference.select_pages(
        block_table, n_pages, scores, k_pages, n_window=n_window)
    return block_table, n_pages, scores, k_pages, n_window, want


@pytest.mark.parametrize("n_window", [0, 2])
def test_both_backends_return_the_reference_result(n_window):
    block_table, n_pages, scores, k, win, want = _case(3, n_window)
    ref = RefBackend()
    got_ref = ref.select_pages(block_table, n_pages, scores, k, n_window=win)
    assert torch.equal(got_ref, want)
    # the tilelang Backend's torch-reference delegate (its CPU twin until a
    # real select kernel lands); built without initializing a device backend.
    got_be = Backend.select_pages(
        Backend.__new__(Backend), block_table, n_pages, scores, k, n_window=win)
    assert torch.equal(got_be, want)
    if win:
        # n_window forces the last win valid pages into the selection even when
        # they are not top-k: the seam must propagate the kwarg, not drop it.
        assert not torch.equal(
            want, reference.select_pages(block_table, n_pages, scores, k, n_window=0))


def test_full_k_sparse_engine_is_dense_token_for_token_via_seam():
    """The existing full-k parity, named here as the seam's end-to-end gate:
    every page selected means sparse attention is dense attention, and the
    selection that decides that now runs through RefBackend.select_pages."""
    import numpy as np
    from test_sparse_engine import _drain, _engine

    from tilerl.engine import SamplingParams

    prompt = np.arange(7, 7 + 12 * 16, dtype=np.int64)
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)
    dense = _engine(False)
    td = _drain(dense, dense.submit(prompt, params), 8)
    dense.shutdown()
    sparse = _engine(True, 64)
    ts = _drain(sparse, sparse.submit(prompt, params), 8)
    sparse.shutdown()
    assert ts == td
