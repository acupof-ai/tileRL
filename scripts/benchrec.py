"""Compatibility shim for the packaged bench store/writer.

The implementation moved to :mod:`tilerl.benchrec` so a wheel/sdist install
(which does not ship this ``scripts/`` directory) can validate and append
benchmark records. Scripts import ``benchrec`` with ``scripts/`` on
``sys.path``; this shim re-exports the package module.

It also registers itself under BOTH module names: scripts mutate module
globals (``benchrec.STORE = path``) and are imported under ``scripts``-style
cwd, while ``tilerl.ledger`` imports the canonical ``tilerl.benchrec`` —
aliasing ``sys.modules["benchrec"]`` to the same object makes the global
swap visible on both sides. Delete this only once every collector is a
packaged entry point and no test/script does ``sys.path.insert(..., scripts)``.
"""

from __future__ import annotations

import sys

import tilerl.benchrec as _benchrec

sys.modules["benchrec"] = _benchrec

# Re-export every public name for `from benchrec import ...` / star imports.
for _name in dir(_benchrec):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_benchrec, _name)

if __name__ == "__main__":
    # Historical: `python3 scripts/benchrec.py` ran the schema selftest. The
    # selftest body now lives in the packaged module; invoke it directly there.
    import runpy

    runpy.run_path(_benchrec.__file__, run_name="__main__")
