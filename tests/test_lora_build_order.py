"""add_lora must run after the model is materialized.

build_engine calls backend.materialize, which replaces params moving device/dtype
with NEW tensors (new id). An adapter attached before that binds to tensors the
forward never reads, so the tape records no gradient for it and a captured graph
reads stale adapters — a silent no-op. add_lora refuses on an unmaterialized
model; build_random is born materialized (RefBackend.materialize is identity).
"""

from __future__ import annotations

import numpy as np
import torch

from tilerl.config import tiny
from tilerl.model import Model, add_lora, build_random
from tilerl.testing import RefBackend


def test_add_lora_raises_before_materialize_then_binds_after():
    """Pre-build add_lora is the stale-adapter bug and must raise; once materialized,
    the attached adapters are the tensors the forward reads and every one gets a grad."""
    import pytest

    cfg = tiny()
    # A raw Model (what load_hf returns before build_engine): default unmaterialized.
    model = Model(cfg, build_random(cfg, seed=0).params)
    assert model.materialized is False
    with pytest.raises(RuntimeError, match="before the model was materialized"):
        add_lora(model, rank=4)

    # Stand-in for a GPU materialize: cross-device .to returns new-id tensors; clone()
    # reproduces that identity break on CPU.
    model.params = {k: v.clone() if torch.is_tensor(v) else v
                    for k, v in model.params.items()}
    model.materialized = True
    trainable = add_lora(model, rank=4)

    from tilerl.autograd import RecordingBackend, Tape
    from tilerl.train import _training_kv

    backend = RefBackend()
    ids = np.array([[1, 2, 3, 4]])
    tape = Tape()
    with torch.no_grad(), tape:
        logits = model.forward(
            ids, np.arange(4, dtype=np.int64),
            _training_kv(model, 1, 4, device=backend.device),
            RecordingBackend(backend))
    grads = tape.backward(torch.ones_like(logits),
                          needs={id(v) for v in trainable.values()})
    assert {id(v) for v in trainable.values()} <= set(grads), \
        "a materialized adapter the forward reads must receive a gradient"


def test_build_random_is_born_materialized_for_cpu_forward_tests():
    """The many CPU tests that add_lora a build_random model and drive forward/train_step
    directly (no engine) stay valid: RefBackend.materialize is identity."""
    cfg = tiny()
    model = build_random(cfg, seed=0, keep_master=False)
    assert model.materialized is True
    adapters = add_lora(model, rank=4)
    assert adapters and all(k in model.params for k in adapters)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
