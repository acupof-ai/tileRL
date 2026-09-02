# tilerl-kernels

The kernel layer of [tileRL](https://github.com/cklxx/tileRL): TileLang kernel
sources plus the `Backend` seam that dispatches them.

One kernel file tree, three targets that have executed it: `cpu` (the CI and
parity path — every kernel has a CPU-executable twin), `metal`, and CUDA sm90.
`Backend` resolves the target once; a per-arch registry cell swaps the kernels a
target needs its own schedule for (sm90 holds the fp4/fp8 tensor-core set).

```python
from tilerl_kernels.backend import get_backend

backend = get_backend()          # TILERL_TARGET=cpu|cuda|metal|auto
y = backend.linear(x, w)
```

**This package is the only place that imports TileLang or calls torch beyond
the tensor container type.** Everything in `tilerl` above it is
backend-neutral, which is why the two ship separately: a kernel change rebuilds
this, a serving change does not.

`tilerl_kernels.reference` holds a torch-eager implementation of every op. It is
the parity oracle the TileLang kernels are gated against (`allclose(rtol=1e-2)`
on the tiny model) and the day-1 backward for ops without a TileLang backward.

Install `tilerl` instead if you want the engine, server or training loop.
