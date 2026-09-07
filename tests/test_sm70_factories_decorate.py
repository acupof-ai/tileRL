"""Can every sm70-registered kernel factory be CONSTRUCTED, off Volta?

`_register("bf16", "sm70", _SM70_KERNELS)` stores factories; nothing on the cpu
path calls them. So a cell that cannot be built is invisible to the entire cpu
suite -- 452 tests passed against one whose `@tilelang.jit` decoration raised
`TypeError: _pass_configs() takes 0 positional arguments but 1 was given` on the
first call the V100 made.

Its own module because `test_sm70_prefill_cell.py` carries a module-level skip
for non-Volta machines, which would skip this too -- and a gate that never runs
is what this file exists to prevent.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
from tilerl_kernels.registry import _SM70_KERNELS


@pytest.mark.parametrize("target", ["c", "cuda"])
def test_every_sm70_factory_decorates(target):
    """No card and no CUDA toolchain needed: tilelang builds the JIT wrapper
    without one, verified on this Mac for target="cuda". Compilation is not
    checked and this does not claim it; what it catches is the signature and
    import errors in a factory body, the class that reached the card.
    """
    broken = []
    for name, factory in sorted(_SM70_KERNELS.items()):
        try:
            factory(target)
        except Exception as exc:  # noqa: BLE001 -- any failure to construct is the finding
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not broken, (
        f"{len(broken)} of {len(_SM70_KERNELS)} sm70 factories do not decorate for "
        f"target={target!r}:\n  " + "\n  ".join(broken)
    )
