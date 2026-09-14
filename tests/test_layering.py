"""Layer direction gate for the wrap-up architecture (docs/design-architecture.md).

Imports point downward only. The layer table is literal data; the canonical copy
is the doc. The allowlist holds edges that violate today and may only shrink:
the assertion is exact, so removing an edge without removing it here also fails.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "tilerl"

# Lower rank = lower layer. Pre-placed future modules (bench) are
# absent from disk today and must not error while absent.
LAYERS: dict[str, frozenset[str]] = {
    "L0": frozenset({"precision", "config", "tokenizer", "testing"}),
    "L1": frozenset({"model", "tensor_parallel", "autograd"}),
    "L2": frozenset({"kv_cache", "kv_tiers", "sparse_index"}),
    "L3": frozenset({"engine", "decode_graph", "sparse_engine", "spec", "memory"}),
    "L4": frozenset({"build"}),
    "L5": frozenset(
        {
            "server",
            "messages",
            "responses",
            "prompt",
            "ui_assets",
            "generate",
            "eval",
            "judge",
            "math_answer",
            "bench",
            "train",
            "iso",
            "merge",
            "calibration",
            "recipes",
            "ledger",
            "kernel_cost",
        }
    ),
    "L6": frozenset({"cli"}),
}
_RANK = {m: i for i, mods in enumerate(LAYERS.values()) for m in mods}

# Package markers, not architecture modules.
_SKIP = frozenset({"__init__", "__main__"})

# Every upward edge observed at fa2ae898. May only shrink.
ALLOWLIST: frozenset[tuple[str, str]] = frozenset()


def read_sources(root: Path = SRC) -> dict[str, str]:
    return {p.stem: p.read_text() for p in root.glob("*.py")}


def violations(sources: dict[str, str]) -> tuple[set[tuple[str, str]], set[str]]:
    """Returns (upward import edges (importer, imported), unlayered module names).

    Both top-level and function-body (lazy) imports are parsed. Absolute
    ``tilerl.<m>`` and relative ``from .<m>`` forms are resolved; the kernels
    package and third-party modules are outside the table.
    """
    upward: set[tuple[str, str]] = set()
    unlayered = {name for name in sources if name not in _RANK and name not in _SKIP}
    for name, text in sources.items():
        if name not in _RANK:
            continue
        for node in ast.walk(ast.parse(text, filename=name)):
            imported: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    if node.module:
                        imported = [node.module.split(".")[0]]
                    else:  # `from . import x`
                        imported = [a.name.split(".")[0] for a in node.names]
                elif node.level == 0 and node.module == "tilerl":
                    # `from tilerl import x` — a level-0 sibling import.
                    imported = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.Import):
                imported = [
                    a.name.split(".")[1] for a in node.names if a.name.startswith("tilerl.")
                ]
            for mod in imported:
                if mod in _RANK and _RANK[mod] > _RANK[name]:
                    upward.add((name, mod))
    return upward, unlayered


def test_real_tree_has_only_allowlisted_upward_edges() -> None:
    upward, unlayered = violations(read_sources())
    assert unlayered == set(), f"module missing from LAYERS: {sorted(unlayered)}"
    assert upward == ALLOWLIST, (
        f"upward edges changed.\n new: {sorted(upward - ALLOWLIST)}\n"
        f" removed (delete from ALLOWLIST): {sorted(ALLOWLIST - upward)}"
    )


def _inject(sources: dict[str, str], name: str, text: str) -> dict[str, str]:
    patched = dict(sources)
    patched[name] = text
    return patched


def test_injected_upward_import_is_flagged_through_same_violations() -> None:
    sources = _inject(
        read_sources(),
        "memory",
        "def f():\n    from .cli import x  # negative control\n",
    )
    upward, unlayered = violations(sources)
    assert unlayered == set()
    assert ("memory", "cli") in upward


def test_level0_from_tilerl_import_of_a_higher_module_is_flagged() -> None:
    sources = _inject(read_sources(), "memory", "from tilerl import cli\n")
    upward, unlayered = violations(sources)
    assert unlayered == set()
    assert ("memory", "cli") in upward


def test_unlisted_module_is_flagged() -> None:
    _, unlayered = violations(_inject(read_sources(), "newmodule", "x = 1\n"))
    assert "newmodule" in unlayered


def test_preplaced_absent_future_modules_do_not_error() -> None:
    for future in ("bench",):
        assert future in _RANK and not (SRC / f"{future}.py").exists()
    # violations() over the real tree (which lacks them) is already green above.


def test_downward_edge_is_not_flagged() -> None:
    upward, _ = violations(_inject(read_sources(), "engine", "from .precision import Format\n"))
    assert not any(src == "engine" and dst == "precision" for src, dst in upward)
