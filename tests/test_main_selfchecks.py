"""Every `__main__` self-check in src/tilerl/ runs, because CI never ran any of them.

AGENTS.md offers "an `assert` in `__main__` or one small test -- whichever is
lighter" as equal alternatives. Authors took the first, correctly, and the
pipeline ran the second: `ruff` + `pytest` + `tests/*_world[0-9].py` + `uv build`
invoked none of the 86 asserts in ten modules' blocks. They are not vacuous
checks -- they cannot even go green, they simply never execute. `spec.py`'s block
holds the #22 block-parallel reject as arithmetic over the ladder and the
staircase constants carrying the 12.9x noise amplification from
wins/2026-09-04-a-difference-amplifies-its-operands-noise.md; a verdict with no
run behind it decays silently. Found by tilerl-25's dead-code audit, 2026-09-08.

SUBPROCESS, NOT runpy. Both execute the block and both propagate a planted
failure -- verified. But `runpy.run_module(m, run_name="__main__")` warns
"found in sys.modules ... prior to execution ... may result in unpredictable
behaviour", because it re-executes module-level code in a second namespace beside
the live one. Nothing broke when measured, and a gate against unrun checks should
not itself rest on "nothing broke today". A subprocess also costs less (5.5 s for
all ten against ~6 s in-process) and is how the authors run these by hand.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "tilerl"

# A ratchet, not a constant. Ten modules carry an asserting block today; if one
# legitimately loses its block, lower this deliberately and let that edit be the
# review point. A gate whose coverage can shrink in silence is the hole this closes.
_MIN_MODULES = 10


def _asserting_main_modules() -> list[str]:
    """Module names whose `if __name__ == "__main__"` block contains an assert.

    DERIVED, NEVER HARDCODED. ci.yml:56 already records what a hand-maintained
    list costs -- "a hand-maintained list omitted three gates" -- so a new
    module's self-check is gated the day it lands rather than when someone
    remembers. cli.py is excluded by the >=1-assert rule, not by an exception:
    its block only calls main(), and pulling a CLI entry point into a self-check
    gate would run the real command line.
    """
    out = []
    for path in sorted(_SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not (isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)):
                continue
            if any(isinstance(n, ast.Assert) for n in ast.walk(node)):
                out.append(path.stem)
    return out


def test_the_module_list_is_derived_and_has_not_shrunk():
    mods = _asserting_main_modules()
    assert len(mods) >= _MIN_MODULES, (
        f"only {len(mods)} module(s) carry an asserting __main__ block, expected "
        f">= {_MIN_MODULES}: {mods}. A block was removed or renamed -- lower the "
        "floor deliberately if that was intended."
    )


@pytest.mark.parametrize("module", _asserting_main_modules())
def test_main_selfcheck_passes(module):
    """`python3 -m tilerl.<module>` exits 0.

    PYTHONPATH is set to src/ rather than trusting the invoking interpreter to have
    tilerl installed. `sys.executable` under `uv run pytest` is the venv's python and
    resolves the import; under a bare `python3 -m pytest` it is the system python and
    every module fails with ModuleNotFoundError -- a gate that only works under one
    launcher reports the launcher, not the checks. tilerl_kernels needs its own root:
    it is a separate package under packages/, and tensor_parallel imports it.

    The CPU target is pinned rather than inherited: these run on GPU-less CI hosts and
    a self-check that silently picks a different backend is not the check the author
    wrote.
    """
    repo = _SRC.parents[1]
    roots = [str(_SRC.parent), str(repo / "packages" / "tilerl-kernels" / "src")]
    env = {**os.environ, "TILERL_TARGET": "cpu"}
    env["PYTHONPATH"] = os.pathsep.join([*roots, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", f"tilerl.{module}"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=_SRC.parents[1],
    )
    assert proc.returncode == 0, (
        f"tilerl.{module} __main__ self-check failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


if __name__ == "__main__":  # runnable check
    mods = _asserting_main_modules()
    assert len(mods) >= _MIN_MODULES, mods
    print(f"selfcheck gate: {len(mods)} modules derived -> {', '.join(mods)}")
