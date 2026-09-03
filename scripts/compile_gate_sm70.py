"""Compile every kernel the sm70 fp4 cell dispatches. Catches a removed extern.

A grep for call sites cannot close this gate: the CUDA externs live in Python
string constants and only reach nvcc at tilelang JIT time, on the GPU host. A
template deleted from _FP4_TWIDDLE_SRC_F16 or a sibling is valid Python and
fails as CUDA, so the CPU test suite and a call-site audit both pass.

Cases come from the registry rather than a hand-written list, so a kernel added
to the cell is covered without editing this file. The GEMV ladder's factory args
(M, xh, sh) are enumerated explicitly because they are compile-time template
parameters -- one entry in the registry is five compiled variants on the shipped
path.

Compiles only: no correctness claim, no timing. Exit 0 means every variant the
sm70 fp4 path can dispatch produced device code.

  scripts/v100.sh run cg '/usr/bin/python3 -u scripts/compile_gate_sm70.py'
"""

from __future__ import annotations

import inspect
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("TILERL_TARGET", "cuda")

from tilerl_kernels.backend import get_backend  # noqa: E402
from tilerl_kernels.registry import _resolve  # noqa: E402

#: Extra factory kwargs to enumerate, by kernel name. These are compile-time
#: template args, so each combination is a separate nvcc invocation.
LADDER = [
    dict(M=m, xh=xh, sh=sh)
    for m in (1, 2, 4, 8, 32)
    for xh, sh in ((True, True), (True, False), (False, False))
]
VARIANTS = {"linear_fp4_gemv_sm70_m": LADDER}


def main() -> None:
    be = get_backend()
    if be.arch != "sm70":
        raise SystemExit(f"arch is {be.arch}, not sm70 — nothing this gate covers")
    cell = _resolve("fp4", be.arch)
    cases = []
    for name, factory in sorted(cell.items()):
        for kw in VARIANTS.get(name, [{}]):
            # A factory whose signature does not accept these kwargs would raise
            # TypeError below and read as a compile failure, which is the wrong
            # diagnosis; skip what it cannot take.
            params = inspect.signature(factory).parameters
            if all(k in params for k in kw):
                cases.append((name, factory, kw))

    print(f"# arch={be.arch} target={be.target}: {len(cases)} variants from the fp4 cell")
    bad = []
    for name, factory, kw in cases:
        label = name + (" " + " ".join(f"{k}={v}" for k, v in kw.items()) if kw else "")
        try:
            factory(be.target, **kw)
            print(f"  ok    {label}")
        except Exception as exc:  # noqa: BLE001 - report every failure, do not stop
            bad.append((label, exc))
            print(f"  FAIL  {label}: {type(exc).__name__}: {str(exc)[:160]}")

    if bad:
        print(f"\n{len(bad)} of {len(cases)} FAILED TO COMPILE")
        for label, exc in bad:
            print(f"\n=== {label} ===")
            traceback.print_exception(type(exc), exc, exc.__traceback__, limit=3)
        raise SystemExit(1)
    print(f"\nall {len(cases)} compiled: no sm70 fp4 extern is missing")


if __name__ == "__main__":
    main()
