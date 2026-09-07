"""A bench that spawns a server must not pin the child's TileLang cache to one pod's path.

Five scripts built the child env as ``dict(os.environ, TILELANG_CACHE_DIR="/work/...")``,
which OVERRIDES the caller. On a box with no ``/work`` the child either dies inside its own
redirected log -- the parent then reports only "server exited" after up to --boot-s 900 --
or starts from an empty cache and recompiles every shape, which is exactly what the arms
using it assert did not happen. Measured on a V100 with no /work: the jitwarm arm could not
open its log file and no server started.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _child_env_overrides(path: Path) -> list[int]:
    """Lines where a dict(os.environ, ...) call pins a cache dir the caller cannot change.

    Only the ``dict(...)`` keyword form, which is the one all five scripts used; an
    assignment or ``|=`` after the copy overrides just as hard and is not caught.
    """
    bad = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "dict"):
            continue
        if any(k.arg == "TILELANG_CACHE_DIR" for k in node.keywords):
            bad.append(node.lineno)
    return bad


def test_no_spawner_pins_the_childs_tilelang_cache():
    hits = {p.name: lines for p in sorted(_SCRIPTS.glob("*.py"))
            if (lines := _child_env_overrides(p))}
    assert not hits, (
        f"these pass TILELANG_CACHE_DIR as a dict(os.environ, ...) keyword, which discards "
        f"the caller's value: {hits}. Use env.setdefault so the default still holds and the "
        f"caller wins")


def test_the_guard_sees_the_shape_it_forbids():
    """Negative control: the checker must fire on the exact expression that was removed."""
    src = 'env = dict(os.environ, TILELANG_CACHE_DIR="/work/tilelang_cache")\n'
    tmp = _SCRIPTS.parent / "tests" / "_env_probe.py"
    tmp.write_text(src, encoding="utf-8")
    try:
        assert _child_env_overrides(tmp) == [1]
    finally:
        tmp.unlink()


if __name__ == "__main__":
    test_no_spawner_pins_the_childs_tilelang_cache()
    test_the_guard_sees_the_shape_it_forbids()
    print("ok")
