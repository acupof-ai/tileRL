"""The narrow-rmsnorm invariant: a narrowed norm may only feed a quantized linear.

``backend.rmsnorm(..., narrow=True)`` makes the kernel write ``gemv_io`` (f16 on
sm70) instead of f32, which is free when the next op is the fp4 GEMV -- it would
narrow anyway. It is NOT free for any other consumer: rope, attention and the GDN
scan are f32, so a f32 -> f16 -> f32 round trip there drops 13 mantissa bits for
nothing.

This failure mode is SILENT and has already happened once on this branch: an
earlier version narrowed ``rmsnorm_apply`` for all of sm70, which quietly degraded
q_norm/k_norm, and EVERY gate passed because ``rope`` calls ``_f32()``
defensively. Nothing numeric catches it -- the defensive widening restores the
dtype, not the bits -- so the gate has to be structural.

Structural also means it runs here, on a GPU-less machine, against the source
rather than against a card.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("TILERL_TARGET", "cpu")

MODEL_PY = Path(__file__).resolve().parent.parent / "src" / "tilerl" / "model.py"
BACKEND_PY = (
    Path(__file__).resolve().parent.parent
    / "packages" / "tilerl-kernels" / "src" / "tilerl_kernels" / "backend.py"
)


def _narrow_targets(tree: ast.AST) -> list[tuple[str, int, ast.FunctionDef]]:
    """(assigned name, lineno, enclosing function) for each `v = rmsnorm(..., narrow=True)`."""
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            c = node.value
            if not (isinstance(c.func, ast.Attribute) and c.func.attr == "rmsnorm"):
                continue
            if not any(k.arg == "narrow" and getattr(k.value, "value", False) is True
                       for k in c.keywords):
                continue
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                out.append((node.targets[0].id, node.lineno, fn))
    return out


def _consumers(name: str, after: int, fn: ast.FunctionDef) -> list[str]:
    """Every call that reads `name` positionally, below `after`, as `recv.attr`."""
    seen = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or getattr(node, "lineno", 0) <= after:
            continue
        if not any(isinstance(a, ast.Name) and a.id == name for a in node.args):
            continue
        f = node.func
        seen.append(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "?"))
    return seen


#: The only consumers a narrowed norm may have. `_linear`/`_base_linear` dispatch to
#: the fp4 GEMV, which reads gemv_io natively; `_add_via` forwards to a linear too.
#: `_tp_fork` is dtype-transparent -- it returns `x.view_as(x)` and its whole job is
#: to put an entry on the tape, so the narrowed value reaches the linear unchanged.
ALLOWED = {"_linear", "_base_linear", "_add_via", "_tp_fork"}


def test_a_narrowed_rmsnorm_only_feeds_a_quantized_linear():
    """Every narrow=True site's output must reach nothing but a linear.

    Negative control: add `backend.rope(h, ...)` after any of these norms and this
    fails naming rope -- which is precisely the edit that silently cost 13 mantissa
    bits when the narrowing was arch-wide instead of per-call-site.
    """
    tree = ast.parse(MODEL_PY.read_text())
    sites = _narrow_targets(tree)
    assert len(sites) >= 4, f"expected the 4+ known narrow sites, found {len(sites)}"

    for name, line, fn in sites:
        got = _consumers(name, line, fn)
        assert got, f"model.py:{line}: `{name}` is narrowed and then never used"
        bad = sorted(set(got) - ALLOWED)
        assert not bad, (
            f"model.py:{line}: narrowed `{name}` in {fn.name}() flows into {bad}, "
            f"which read f32 -- a f32->f16->f32 round trip drops 13 mantissa bits "
            f"and no numeric gate catches it (rope/attention call _f32 defensively). "
            f"Either drop narrow=True here or give that consumer a narrow variant."
        )


def test_q_and_k_norm_are_never_narrowed():
    """q_norm/k_norm feed rope and attention, so they must stay f32.

    Guards the specific regression: narrowing them is invisible downstream and was
    shipped once. A norm whose weight key ends in q_norm/k_norm has no business
    being narrowed, whatever the call site looks like.
    """
    src = MODEL_PY.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "rmsnorm":
            continue
        if not any(k.arg == "narrow" and getattr(k.value, "value", False) is True
                   for k in node.keywords):
            continue
        seg = ast.get_source_segment(src, node) or ""
        assert "q_norm" not in seg and "k_norm" not in seg, (
            f"model.py:{node.lineno}: q_norm/k_norm narrowed -- they feed rope and "
            f"attention, which are f32"
        )


def test_narrow_is_a_no_op_where_the_gemv_takes_f32():
    """`narrow=True` must not change the kernel on an arch whose GEMV reads f32.

    The flag is a request, not an instruction: sm90 and cpu keep rmsnorm_apply. If
    this ever selected the narrow kernel by the flag alone, every non-sm70 target
    would silently lose precision at the four call sites.

    Asserts the condition's TERMS, not one formatting of it: the first cut pinned
    the whole line as a string and went red when a rebase reflowed it across three
    lines to admit `out_f32` -- a green-to-red on a change that preserved the
    invariant exactly. Parse the branch instead.
    """
    import ast

    src = BACKEND_PY.read_text()
    tree = ast.parse(src)
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_rmsnorm"),
        None,
    )
    assert fn is not None, "backend._rmsnorm is gone; the narrow gate has no subject"
    picks = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.IfExp)
        and "rmsnorm_apply_narrow" in ast.dump(n)
    ]
    assert len(picks) == 1, f"expected one narrow-kernel choice in _rmsnorm, found {len(picks)}"
    cond = ast.dump(picks[0].test)
    assert "narrow" in cond, "the narrow kernel must be gated on the caller's request"
    assert "gemv_io" in cond, (
        "backend._rmsnorm must gate the narrow kernel on gemv_io, not on `narrow` alone: "
        "an arch whose GEMV reads f32 would round-trip through f16 for nothing"
    )


def test_the_reference_rmsnorm_stays_f32():
    """The parity target must not narrow, or the gate compares f16 to f16.

    reference.rmsnorm accepts `narrow` and ignores it -- that is deliberate, and it
    is what makes the CPU twin a real check on the sm70 kernel's output.
    """
    import torch
    from tilerl_kernels import reference

    x = torch.randn(2, 3, 8, dtype=torch.float32)
    w = torch.ones(8)
    plain = reference.rmsnorm(x, w, 1e-6)
    asked = reference.rmsnorm(x, w, 1e-6, narrow=True)
    assert asked.dtype == torch.float32, "the reference must stay f32"
    assert torch.equal(plain, asked), "narrow must not change the reference's output"


if __name__ == "__main__":
    test_a_narrowed_rmsnorm_only_feeds_a_quantized_linear()
    test_q_and_k_norm_are_never_narrowed()
    test_narrow_is_a_no_op_where_the_gemv_takes_f32()
    test_the_reference_rmsnorm_stays_f32()
    print("narrow invariant holds")
