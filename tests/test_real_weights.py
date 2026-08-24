"""Real-weight smoke test on the committed fixture.

``tests/fixtures/qwen35-2layer-mlx4`` is cropped from the local Qwen3.5-0.8B
MLX-4bit (scripts/crop_fixture.py): 2 layers (one GDN, one full-attn), vocab
trimmed to 1024, MLX affine-4bit format kept on disk. This exercises the real
GatedDeltaNet tensor names and the MLX dequant path hermetically — no external
model, runs on CI. The full 24-layer generation check stays manual (the 0.8B
is too large for the suite); see docs/experience/wins/.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TILERL_TARGET", "cpu")

import numpy as np
import torch

from tilerl.autograd import AdamW, Tape
from tilerl.config import qwen35_08b
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import load_hf
from tilerl.ops.backend import get_backend
from tilerl.train import train_step

_FIXTURE = Path(__file__).parent / "fixtures" / "qwen35-2layer-mlx4"


def test_real_08b_fixture_forward_and_train():
    """Load the cropped real checkpoint, generate through the engine, then one
    train_step: finite outputs, real GDN math + MLX dequant end to end."""
    cfg = replace(qwen35_08b(), vocab_size=1024, num_layers=2, full_attn_layers=(1,))
    model = load_hf(cfg, str(_FIXTURE))
    # MLX dequant ran: the packed uint32 embedding is now a bf16 [V, H] matrix.
    assert model.params["embed_tokens"].dtype != torch.uint32
    assert tuple(model.params["embed_tokens"].shape) == (1024, 1024)
    backend = get_backend()
    ids = np.random.default_rng(0).integers(3, 1024, size=(1, 8)).astype(np.int64)

    engine = build_engine(
        cfg, model, backend, num_blocks=8, num_slots=4, max_batch=4, max_total_tokens=512
    )
    rid = engine.submit(ids[0], SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    out = {}
    for _ in range(128):
        engine.step()
        out.update(engine.poll())
        if rid in out:
            break
    assert rid in out, "engine did not finish the real-weight request"
    assert len(out[rid]) == 2 and all(0 <= t < 1024 for t in out[rid])
    engine.shutdown()

    loss = train_step(model, ids, backend, AdamW(lr=1e-4), Tape())
    assert np.isfinite(loss), f"non-finite loss: {loss}"
