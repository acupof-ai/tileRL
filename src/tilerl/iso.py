"""ISO: fixed-spectrum optimization (arXiv 2607.19331).

A 2D weight is held as ``W = U S V^T`` with ``S`` frozen at its initial
singular values. The base optimizer moves the frames ``U`` and ``V`` only;
a polar retraction puts them back on the Stiefel manifold after each step,
so every 2D weight keeps its spectrum for the whole run. Same contract as
:class:`tilerl.autograd.Adafactor` — ``streams`` / ``begin`` / ``step_one`` —
so the streamed full-parameter path in ``train._step`` is unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from . import precision
from .autograd import Adafactor


def polar(x: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Nearest matrix with orthonormal columns, by Newton-Schulz. Converges
    quadratically from ``||X^T X - I|| < 1``, which a small step never leaves."""
    # ponytail: torch matmul in the retraction; tilelang kernel when perf demands
    for _ in range(iters):
        x = 1.5 * x - 0.5 * x @ (x.T @ x)
    return x


def frame_grads(g: torch.Tensor, u: torch.Tensor, s: torch.Tensor, v: torch.Tensor):
    """Chain rule through ``W = U S V^T``: ``G_U = G V S``, ``G_V = G^T U S``."""
    return g @ v * s, g.T @ u * s


class ISO:
    """Wrap a base optimizer (default :class:`Adafactor`, :class:`AdamW` also
    works) so 2D params train in fixed-spectrum coordinates. Non-2D params go
    straight to the base optimizer.

    Cost: ``U``, ``S``, ``V`` in the policy's frame dtype per 2D weight — two
    fp32 copies of the weight today (``precision.dtype("frame")``), plus the
    base optimizer's state on each frame instead of on the weight.
    """

    streams = True

    def __init__(
        self,
        base: Any | None = None,
        polar_iters: int = 5,
        offload: bool | None = None,
    ) -> None:
        self.base = Adafactor() if base is None else base
        self.frame_dtype = precision.dtype("frame")
        self.polar_iters = polar_iters
        # fp32 frames of the 27B are 200 GiB: they live on the host and one
        # matrix is staged to the device per update — the streamed step already
        # goes one parameter at a time. None = offload when the param is on cuda.
        self.offload = offload
        self._frames: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def lr(self) -> float:
        return self.base.lr

    @lr.setter
    def lr(self, value: float) -> None:
        self.base.lr = value

    def begin(self) -> None:
        self.base.begin()

    def step(self, params: Any, grads: dict[int, torch.Tensor]) -> None:
        self.begin()
        for p in params:
            g = grads.get(id(p))
            if g is not None:
                self.step_one(p, g)

    def frames(self, p: torch.Tensor):
        fr = self._frames.get(id(p))
        if fr is None:
            # ponytail: torch.linalg.svd at init, Newton-Schulz when it matters
            u, s, vh = torch.linalg.svd(p.to(self.frame_dtype), full_matrices=False)
            # Guard at the point of use: a frame dtype that cannot hold
            # orthonormality would train silently on a drifting spectrum.
            err = float((u.T @ u - torch.eye(u.shape[1], dtype=u.dtype)).abs().max())
            if err > 1e-3:
                raise ValueError(f"ISO: frames in {self.frame_dtype} are not orthonormal "
                                 f"(max|UᵀU−I| = {err:.1e}); change precision.py, not this call")
            fr = (u.contiguous(), s, vh.T.contiguous())
            if self._offloaded(p):
                fr = tuple(t.cpu() for t in fr)
            self._frames[id(p)] = fr
        return fr

    def _offloaded(self, p: torch.Tensor) -> bool:
        return p.device.type == "cuda" if self.offload is None else self.offload

    def step_one(self, p: torch.Tensor, g: torch.Tensor) -> None:
        if p.dim() != 2:
            self.base.step_one(p, g)
            return
        u, s, v = self.frames(p)
        staged = self._offloaded(p)
        uu, ss, vv = (t.to(p.device, copy=True) for t in (u, s, v)) if staged else (u, s, v)
        gu, gv = frame_grads(g.to(uu.dtype), uu, ss, vv)
        # State is keyed per matrix, not per staging buffer.
        self.base.step_one(uu, gu, key=(id(p), "u"))
        self.base.step_one(vv, gv, key=(id(p), "v"))
        uu.copy_(polar(uu, self.polar_iters))
        vv.copy_(polar(vv, self.polar_iters))
        p.copy_(((uu * ss) @ vv.T).to(p.dtype))
        if staged:
            u.copy_(uu)
            v.copy_(vv)
