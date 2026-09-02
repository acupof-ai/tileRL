"""Same tape gradient, two targets: is a failing CUDA gradcheck a wrong
gradient or a finite-difference probe that bf16 cannot support?

    TILERL_TARGET=cpu  uv run python scripts/tape_grad_targets.py
    TILERL_TARGET=cuda python3 scripts/tape_grad_targets.py
"""

from __future__ import annotations

import numpy as np
import torch

from tilerl.autograd import RecordingBackend, Tape
from tilerl.config import tiny
from tilerl.model import build_random
from tilerl.train import _training_kv
from tilerl_kernels.backend import get_backend

KEYS = ("embed_tokens", "layers.0.q_proj", "layers.1.in_proj_a", "final_norm")


def main() -> None:
    cfg = tiny()
    model = build_random(cfg, seed=42)
    backend = get_backend()
    batch = np.random.default_rng(3).integers(3, cfg.vocab_size, size=(2, 16)).astype(np.int64)
    positions = np.arange(16, dtype=np.int64)

    def loss_and_grads():
        kv = _training_kv(model, 2, 16, device=backend.device)
        tape = Tape()
        with torch.no_grad(), tape:
            logits = model.forward(batch, positions, kv, RecordingBackend(backend))
        loss, dlogits = backend.cross_entropy_loss_grad(logits, batch)
        return loss, tape.backward(dlogits)

    loss, grads = loss_and_grads()
    print(f"target {backend.target}  loss {float(loss):.6f}")
    for key in KEYS:
        p = model.params[key]
        idx = (0, 0) if p.ndim == 2 else (0,)
        analytic = grads[id(p)][idx].item()
        # a sound central-difference probe reports the same slope at all three steps.
        nums = []
        for step in (0.1, 0.05, 0.025):
            orig = p[idx].item()
            p[idx] = orig + step
            lp, _ = loss_and_grads()
            p[idx] = orig - step
            lm, _ = loss_and_grads()
            p[idx] = orig
            nums.append((float(lp) - float(lm)) / (2 * step))
        spread = (max(nums) - min(nums)) / max(abs(np.mean(nums)), 1e-12)
        print(f"  {key:22} tape {analytic:+.4e}  fd {[f'{n:+.4e}' for n in nums]}"
              f"  fd-spread {100 * spread:.1f}%  dtype {p.dtype}")


if __name__ == "__main__":
    main()
