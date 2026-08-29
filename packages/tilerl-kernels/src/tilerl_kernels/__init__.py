"""Operator layer: the only tilerl modules that talk to tilelang directly.

* :mod:`tilerl_kernels.backend`  — :class:`Backend` dispatch, ``get_backend()``
* :mod:`tilerl_kernels.registry` — precision×arch dispatch matrix, target resolution
* :mod:`tilerl_kernels.kernels`  — TileLang JIT kernels (target-neutral, CPU + GPU)
* :mod:`tilerl_kernels.reference` — torch-eager parity oracle and day-1 backward fallback
"""
