"""The precision table is total over its roles, and the call sites read it."""

import torch

from tilerl import precision


def test_policy_is_total_and_device_aware():
    for r in precision.roles():
        assert isinstance(precision.dtype(r, "cpu"), torch.dtype)
    assert precision.dtype("recurrent_state", "cuda") == torch.float32
    assert precision.dtype("recurrent_state", "cpu") == torch.bfloat16
    assert precision.dtype("optimizer_state") == torch.float32


def test_iso_frames_follow_the_policy():
    from tilerl.iso import ISO

    u, _, _ = ISO().frames(torch.randn(6, 4, dtype=torch.bfloat16))
    assert u.dtype == precision.dtype("frame")


def test_on_policy_guard_refuses_cached_engines():
    import pytest

    from tilerl.cli import _build_model
    from tilerl.engine import build_engine
    from tilerl.testing import RefBackend
    from tilerl.train import grpo_loop

    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    engine = build_engine(cfg, model, RefBackend(), num_blocks=32, num_slots=4)  # prefix cache on
    with pytest.raises(ValueError, match="on-policy"):
        grpo_loop(engine, model, [[1, 2, 3]], lambda p, c: 0.0, 1, RefBackend())


if __name__ == "__main__":  # runnable check
    test_policy_is_total_and_device_aware()
    print("precision: policy OK")
