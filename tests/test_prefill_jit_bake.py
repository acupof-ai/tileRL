"""Gate logic for the prefill JIT bake script (awb impl#E / #840).

The bake's acceptance is that a verify sweep over prompt lengths NOT used in the
bake sweep adds ZERO cubins (they land in the same prefill buckets). The device
run does the real compile; this pins the exit-code decision and the cubin
counting so "missed a bucket" cannot silently exit 0.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bake():
    spec = importlib.util.spec_from_file_location(
        "probe_prefill_jit_bake", ROOT / "scripts" / "probe_prefill_jit_bake.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bake_gate_fails_when_fresh_lengths_add_cubins():
    mod = _bake()
    # no --cache-dir: nothing measured, cannot gate
    assert mod._bake_ok(None)
    # complete bake: zero new cubins on the fresh-length sweep
    assert mod._bake_ok(0)
    # negative control: one (or more) missed bucket -> not OK, caller exits 1
    assert not mod._bake_ok(1)
    assert not mod._bake_ok(7)


def test_cubin_count_only_counts_cubin_files(tmp_path):
    mod = _bake()
    (tmp_path / "a.cubin").write_bytes(b"x")
    (tmp_path / "b.cubin").write_bytes(b"x")
    (tmp_path / "index.json").write_text("{}")
    (tmp_path / ".hidden").write_text("x")
    assert mod._cubins(str(tmp_path)) == 2
    assert mod._cubins(str(tmp_path / "missing")) == 0
    assert mod._cubins("") is None
