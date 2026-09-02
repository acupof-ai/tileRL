"""ISO: the frame gradient, the retraction, and the invariant they buy.

The optimizer only moves U and V, so the gates are the chain rule into the
frames, orthonormality after a step, and singular values that do not move
over a run — then the loss on the tiny model has to actually go down.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from tilerl.autograd import Adafactor, AdamW
from tilerl.cli import _build_model
from tilerl.iso import ISO, frame_grads, polar
from tilerl.testing import RefBackend
from tilerl.train import train_step


def _orth_err(x):
    q = x.shape[1]
    return (x.T @ x - torch.eye(q, dtype=x.dtype)).abs().max().item()


def test_frame_gradient_matches_finite_difference():
    """<G_U, xi> must equal the directional derivative of L(W(U)) along a
    Stiefel tangent xi through the polar retraction, for a smooth non-linear L."""
    torch.manual_seed(0)
    w = torch.randn(12, 7, dtype=torch.float64)
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    v = vh.T
    loss = lambda w: torch.sin(w).sum()  # noqa: E731
    gu, gv = frame_grads(torch.cos(w), u, s, v)
    for x, other, mk in ((u, v, lambda x: (x * s) @ v.T), (v, u, lambda x: (u * s) @ x.T)):
        a = torch.randn(x.shape[1], x.shape[1], dtype=torch.float64)
        b = torch.randn(x.shape, dtype=torch.float64)
        xi = x @ (a - a.T) / 2 + b - x @ (x.T @ b)  # tangent at x: x^T xi skew
        g = gu if x is u else gv
        h = 1e-5
        fd = (loss(mk(polar(x + h * xi))) - loss(mk(polar(x - h * xi)))) / (2 * h)
        an = (g * xi).sum()
        assert abs(fd - an) < 1e-3 * abs(an), f"frame grad {an:.6f} vs fd {fd:.6f}"


def test_step_keeps_frames_orthonormal():
    torch.manual_seed(0)
    for base in (Adafactor(lr=1e-2), AdamW(lr=1e-3)):
        opt = ISO(base)
        p = torch.randn(64, 48).to(torch.bfloat16)
        for _ in range(3):
            opt.begin()
            opt.step_one(p, torch.randn(64, 48))
        u, s, v = opt.frames(p)
        assert max(_orth_err(u), _orth_err(v)) < 1e-4


def test_spectrum_preserved_on_tiny_model():
    torch.manual_seed(0)
    _, model = _build_model("tiny", seed=0, keep_master=True)
    opt = ISO(Adafactor(lr=1e-2))
    s0 = {k: torch.linalg.svdvals(p.float()) for k, p in model.params.items() if p.dim() == 2}
    backend, ids = RefBackend(), np.arange(1, 2 * 16 + 1, dtype=np.int64).reshape(2, 16)
    for _ in range(5):
        train_step(model, ids, backend, opt)
    for k, s in s0.items():
        p = model.params[k]
        u, s_fr, v = opt.frames(p)
        # the invariant lives in the fp32 frames; the bf16 param is their rounding
        rebuilt = torch.linalg.svdvals((u * s_fr) @ v.T)
        assert torch.allclose(rebuilt, s, rtol=1e-3), f"{k}: spectrum moved"
        assert torch.allclose(torch.linalg.svdvals(p.float()), s, rtol=2e-2, atol=1e-2), k


def test_iso_lowers_loss_on_tiny_model():
    torch.manual_seed(0)
    _, model = _build_model("tiny", seed=0, keep_master=True)
    opt = ISO(Adafactor(lr=1e-2))
    ids = np.random.default_rng(0).integers(1, 300, size=(2, 32))
    backend = RefBackend()
    t0 = time.time()
    losses = [train_step(model, ids, backend, opt) for _ in range(8)]
    dt = (time.time() - t0) / 8
    assert losses[-1] < losses[0] - 0.1, f"ISO did not learn: {losses}"
    return losses, dt


if __name__ == "__main__":  # runnable check
    test_frame_gradient_matches_finite_difference()
    test_step_keeps_frames_orthonormal()
    test_spectrum_preserved_on_tiny_model()
    losses, dt = test_iso_lowers_loss_on_tiny_model()
    print(f"iso: loss {losses[0]:.4f} -> {losses[-1]:.4f}, {dt:.2f}s/step")
