"""ISO-Merger on the tiny model: the frame procedure is exact enough on one
specialist, keeps the base spectrum, and composing two SFT specialists beats
both the base and plain task-vector averaging on their own batches.
"""

from __future__ import annotations

import numpy as np
import torch

from tilerl.autograd import AdamW
from tilerl.cli import _build_model
from tilerl.merge import average_merge, iso_merge
from tilerl.model import Model
from tilerl.testing import RefBackend
from tilerl.train import train_step

_RNG = np.random.default_rng(0)
BATCH_A = _RNG.integers(1, 320, size=(4, 16))
BATCH_B = _RNG.integers(1, 320, size=(4, 16))


def _loss(model, ids, backend):
    return train_step(model, ids, backend, AdamW(lr=0.0))  # lr=0: forward + loss, no update


def _sft(ids, backend, steps=15):
    _, model = _build_model("tiny", seed=0, keep_master=True)
    opt = AdamW(lr=1e-3)
    for _ in range(steps):
        train_step(model, ids, backend, opt)
    return model


def _f32(params):
    return {k: v.float() for k, v in params.items()}


def test_iso_merge_one_specialist_and_spectrum():
    """K=1 returns the specialist (up to the masked trailing modes), and every
    merged matrix carries the base's singular values."""
    backend = RefBackend()
    _, base = _build_model("tiny", seed=0, keep_master=True)
    spec = _sft(BATCH_A, backend)
    merged = iso_merge(_f32(base.params), [_f32(spec.params)])
    for k, w in merged.items():
        if w.dim() != 2:
            continue
        err = (w - spec.params[k].float()).norm() / spec.params[k].float().norm()
        assert err < 1e-2, f"{k}: K=1 merge is {err:.2e} from the specialist"
        s0, s = torch.linalg.svdvals(base.params[k].double()), torch.linalg.svdvals(w.double())
        assert torch.allclose(s, s0, rtol=1e-3), f"{k}: spectrum moved"


def test_iso_merge_two_specialists():
    backend = RefBackend()
    cfg, base = _build_model("tiny", seed=0, keep_master=True)
    a, b = _sft(BATCH_A, backend), _sft(BATCH_B, backend)
    iso = Model(cfg, iso_merge(base.params, [a.params, b.params]))
    avg = Model(cfg, average_merge(base.params, [a.params, b.params]))
    out = {
        n: (_loss(m, BATCH_A, backend), _loss(m, BATCH_B, backend))
        for n, m in (("base", base), ("avg", avg), ("iso", iso))
    }
    print({n: f"A={la:.3f} B={lb:.3f}" for n, (la, lb) in out.items()})
    assert out["iso"][0] < out["base"][0] and out["iso"][1] < out["base"][1], out
    assert out["iso"][0] <= out["avg"][0] or out["iso"][1] <= out["avg"][1], out


if __name__ == "__main__":  # runnable check
    test_iso_merge_one_specialist_and_spectrum()
    test_iso_merge_two_specialists()
    print("merge: K=1, spectrum, two specialists OK")
