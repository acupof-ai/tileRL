"""Per-step re-quantization: after a full-SFT optimizer step the served fp4
faces (``.wq/.scale/.oscale``) are re-packed from the trained bf16 masters in
place, so a shared serving engine decodes the updated weights without a reload.

Two claims, each with the negative control that makes it real:

1. Re-packed served logits equal the bf16 master's logits within the fp4
   parity budget (tiny fixture at the 27B's real ~0.02 weight magnitude — at
   N(0,1) the 4-bit grid is meaningless) and pick the same argmax tokens.
2. After a step the served slot BYTES change in place (addresses preserved for
   a captured graph) and equal a fresh repack; a stale path that skips the
   re-pack leaves them byte-identical to step 0 and must go red. After two
   steps a fixed-seed decode off the served faces picks exactly the trained
   bf16 master's tokens.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from tilerl_kernels.reference import pack_fp4

from tilerl.autograd import Adafactor
from tilerl.config import tiny
from tilerl.model import build_random, drop_quantized, fp4_param_keys, requantize_fp4
from tilerl.testing import RefBackend
from tilerl.train import _training_kv, train_step

#: 27B linears land around this (test_ops_parity packs w * 0.02); N(0,1) saturates
#: the e2m1 grid to the point fp4 parity is meaningless, so the gate scales down.
_W_SCALE = 0.02


def _fp4_tiny():
    cfg = replace(tiny(), fp4=True)
    model = build_random(cfg, seed=0, keep_master=True)
    with torch.no_grad():
        for k, p in model.params.items():
            if p.dim() == 2 and f"{k}.wq" in model.params:
                p.mul_(_W_SCALE)
    requantize_fp4(model)  # served slots start at the scaled master
    return cfg, model


def _served_logits(model, ids):
    backend = RefBackend()
    kv = _training_kv(model, ids.shape[0], ids.shape[1], device=backend.device)
    with torch.no_grad():
        return model.forward(ids, np.arange(ids.shape[1], dtype=np.int64), kv, backend)


def _train(model, steps, *, requant, lr=0.05):
    backend = RefBackend()
    opt = Adafactor(lr=lr)
    post = (lambda: requantize_fp4(model)) if requant else None
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randint(0, model.cfg.vocab_size, (2, 64), generator=gen)
        train_step(model, x, backend, opt, post_step=post)


def test_repacked_served_logits_match_the_bf16_master():
    cfg, served = _fp4_tiny()
    _, master = _fp4_tiny()
    drop_quantized(master)
    ids = (np.arange(64) % (cfg.vocab_size - 1)).astype(np.int64).reshape(1, 64)
    ls, lm = _served_logits(served, ids), _served_logits(master, ids)
    assert torch.allclose(ls, lm, rtol=5e-2, atol=1e-1)
    assert (ls[0].argmax(-1) == lm[0].argmax(-1)).all()


def test_a_step_refreshes_the_served_slot_bytes_in_place():
    cfg, model = _fp4_tiny()
    key = sorted(k for k in fp4_param_keys(cfg) if f"{k}.wq" in model.params)[0]
    before = {s: model.params[key + s].clone()
              for s in (".wq", ".scale", ".oscale")}
    addrs = {s: model.params[key + s].data_ptr() for s in before}

    _train(model, 2, requant=True)

    # changed...
    assert all(not torch.equal(model.params[key + s], before[s]) for s in before)
    # ...at the same addresses (a captured decode graph keeps baking these)...
    assert all(model.params[key + s].data_ptr() == addrs[s] for s in before)
    # ...and equal a fresh repack of the trained master exactly.
    master = model.params[key]
    wq, _ = pack_fp4(master, block=master.shape[1] // model.params[key + ".scale"].shape[1])
    assert torch.equal(model.params[key + ".wq"], wq)


def test_stale_fp4_path_leaves_the_served_bytes_untouched():
    """Negative control: skip the re-pack and the served slots stay byte-identical
    to step 0 even though the masters trained — exactly the off-policy trap."""
    cfg, stale = _fp4_tiny()
    key = sorted(k for k in fp4_param_keys(cfg) if f"{k}.wq" in stale.params)[0]
    before = {s: stale.params[key + s].clone() for s in (".wq", ".scale", ".oscale")}

    _train(stale, 2, requant=False)

    assert all(torch.equal(stale.params[key + s], before[s]) for s in before)


def test_two_steps_served_tokens_equal_the_trained_master_fixed_seed():
    cfg, served = _fp4_tiny()
    _, master = _fp4_tiny()
    _train(served, 2, requant=True, lr=5e-3)
    _train(master, 2, requant=False, lr=5e-3)
    drop_quantized(master)
    ids = (np.arange(40) % (cfg.vocab_size - 1)).astype(np.int64).reshape(1, 40)
    ts = _served_logits(served, ids)[0].argmax(-1)
    tm = _served_logits(master, ids)[0].argmax(-1)
    assert (ts == tm).all()  # the served face samples the step-2 weights


def test_stale_and_repacked_streams_diverge_token_level():
    """Fixed-seed decode after training: a served face that was never re-packed
    samples a DIFFERENT token sequence than one repacked every step (and the
    trained master). This is the off-policy failure at the token level, with an
    arm strong enough to move argmax (short/low-lr steps do not)."""
    cfg, served = _fp4_tiny()
    _, stale = _fp4_tiny()
    steps, lr = 10, 0.2
    _train(served, steps, requant=True, lr=lr)
    _train(stale, steps, requant=False, lr=lr)
    ids = (np.arange(40) % (cfg.vocab_size - 1)).astype(np.int64).reshape(1, 40)
    ts = _served_logits(served, ids)[0].argmax(-1)
    tx = _served_logits(stale, ids)[0].argmax(-1)
    assert (ts != tx).any()  # stale served bytes sample the old policy


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
