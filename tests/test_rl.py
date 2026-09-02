"""GRPO: the policy-gradient step and the group baseline.

The step's gradient is the causal-CE gradient scaled per row by the advantage
and masked to completion tokens, so the sharp gate is algebraic — advantage 1
everywhere must reproduce an SFT step exactly, and advantage 0 must be a no-op.
The loop then has to actually raise reward on a task the tiny model can learn.
"""

from __future__ import annotations

import numpy as np
import torch

from tilerl.autograd import AdamW
from tilerl.cli import _build_model
from tilerl.eval import last_number
from tilerl.testing import RefBackend
from tilerl.train import group_advantages, rl_step, train_step


def _snapshot(model):
    return {k: v.clone() for k, v in model.params.items()}


def _max_delta(model, snap):
    return max((model.params[k] - v).abs().max().item() for k, v in snap.items())


def test_group_advantages():
    # Group-normalized: mean 0, unit std, and no signal when the group ties.
    adv = group_advantages([1.0, 2.0, 3.0, 4.0, 5.0, 5.0], group=3)
    assert abs(adv[:3].mean()) < 1e-12 and abs(adv[:3].std() - 1.0) < 1e-6
    assert np.allclose(group_advantages([2.0, 2.0], group=2), 0.0)


def test_rl_step_matches_sft_at_unit_advantage():
    """A=1 on every row with no prompt is exactly the SFT gradient: the RL step
    must not be a second training path, only a reweighting of this one."""
    ids = np.arange(1, 2 * 12 + 1, dtype=np.int64).reshape(2, 12)
    backend = RefBackend()
    out = {}
    for name, fn in (
        ("sft", lambda m, o: train_step(m, ids, backend, o)),
        ("rl", lambda m, o: rl_step(m, ids, np.ones(2), np.ones(2, dtype=np.int64),
                                    backend, o)),
    ):
        _, model = _build_model("tiny", seed=0, keep_master=True)
        snap = _snapshot(model)
        fn(model, AdamW(lr=1e-3))
        out[name] = {k: model.params[k] - v for k, v in snap.items()}
    worst = max((out["sft"][k] - out["rl"][k]).abs().max().item() for k in out["sft"])
    assert worst < 1e-6, f"rl_step diverges from train_step at A=1: {worst:.2e}"


def test_rl_step_zero_advantage_is_a_noop():
    ids = np.arange(1, 2 * 12 + 1, dtype=np.int64).reshape(2, 12)
    _, model = _build_model("tiny", seed=0, keep_master=True)
    snap = _snapshot(model)
    rl_step(model, ids, np.zeros(2), np.full(2, 4, dtype=np.int64), RefBackend(),
            AdamW(lr=1e-3))
    assert _max_delta(model, snap) == 0.0


def test_rl_step_ignores_padding():
    """Right-padding past seq_len must not reach the gradient. Only padding:
    prompt tokens legitimately move it, through the forward pass that conditions
    every scored position on them."""
    base = np.arange(1, 12 + 1, dtype=np.int64).reshape(1, 12)
    other = base.copy()
    other[0, 9:] = 77  # padding past seq_len; causal, so no scored logit sees it
    deltas = []
    for ids in (base, other):
        _, model = _build_model("tiny", seed=0, keep_master=True)
        snap = _snapshot(model)
        rl_step(model, ids, np.array([1.0]), np.array([3]), RefBackend(),
                AdamW(lr=1e-3), seq_lens=np.array([9]))
        deltas.append({k: model.params[k] - v for k, v in snap.items()})
    worst = max((deltas[0][k] - deltas[1][k]).abs().max().item() for k in deltas[0])
    assert worst < 1e-6, f"prompt/padding leaked into the RL gradient: {worst:.2e}"


def test_grpo_loop_raises_reward():
    """End to end on the tiny model: rollouts through the engine, a reward the
    policy can move, and reward must go up. The engine that samples is the model
    that trains — no second copy of the weights."""
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.train import grpo_loop

    torch.manual_seed(0)
    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    backend = RefBackend()
    engine = build_engine(cfg, model, backend, num_blocks=256, num_slots=8,
                          decode_graph=False, prefix_store=NoPrefixStore())
    half = cfg.vocab_size // 2

    # Dense reward: an untrained policy's group needs variance for a gradient at step 0.
    def reward(prompt, completion):
        return sum(1 for t in completion if t < half) / max(len(completion), 1)

    prompts = [[1, 2, 3, 4]]
    hist = grpo_loop(engine, model, prompts, reward, 12, backend,
                     AdamW(lr=0.05), group=6, sampling=SamplingParams(max_new_tokens=6), seed=0)
    first = np.mean([r for r, *_ in hist[:3]])
    last = np.mean([r for r, *_ in hist[-3:]])
    assert last > first, f"GRPO did not raise reward: {first:.3f} -> {last:.3f}"


def test_last_number():
    assert last_number("so the answer is 1,234.5 dollars") == 1234.5
    assert last_number("#### -7") == -7.0
    assert last_number("no digits") is None and last_number(None) is None


def test_opd_keeps_adapter_tensor_identity():
    """A captured decode graph holds the adapter tensor objects. The
    teacher/student swap in opd_loop must copy into them, never rebind — a
    rebind samples from the captured (stale) tensors on CUDA and never raises."""
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.model import add_lora
    from tilerl.train import opd_loop

    cfg, model = _build_model("tiny", seed=0)
    backend = RefBackend()
    engine = build_engine(cfg, model, backend, num_blocks=64, num_slots=4,
                          decode_graph=False, prefix_store=NoPrefixStore())
    trainable = add_lora(model, rank=4)
    ids = {k: id(model.params[k]) for k in trainable}
    opd_loop(engine, model, [[1, 2, 3, 4]], 2, backend, AdamW(lr=1e-3),
             trainable=trainable, sampling=SamplingParams(max_new_tokens=4))
    assert {k: id(model.params[k]) for k in trainable} == ids


if __name__ == "__main__":  # runnable check
    test_group_advantages()
    test_rl_step_matches_sft_at_unit_advantage()
    print("rl: advantage + step OK")
