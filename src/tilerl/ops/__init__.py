"""Operator layer: the only tilerl modules that talk to tilelang directly.

* :mod:`tilerl.ops.backend`  — :class:`Backend` dispatch, ``get_backend()``
* :mod:`tilerl.ops.registry` — precision×arch dispatch matrix, target resolution
* :mod:`tilerl.ops.kernels`  — TileLang JIT kernels (target-neutral, CPU + GPU)
* :mod:`tilerl.ops.reference` — torch-eager parity oracle and day-1 backward fallback
"""
