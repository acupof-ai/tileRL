"""Per-step re-quantization: after a full-SFT optimizer step the served fp4
faces (``.wq/.scale/.oscale``) are re-packed from the trained bf16 masters in
place, so a shared serving engine decodes the updated weights without a reload.

Claims, each with the negative control that makes it real:

1. Re-packed served logits pick the bf16 master's exact argmax tokens and stay
   inside a DISTRIBUTION bound (p99 effective relative error, max abs) — not a
   single allclose band, which a plausible quantization error can fail and a
   wrong-block repack can pass. The bounds sit in the measured gap between a
   correct repack and the intended mutant.
2. After a step the served slot BYTES change in place (addresses preserved for
   a captured graph) and equal a fresh repack; a stale path that skips the
   re-pack leaves them byte-identical to step 0.
3. A slot a card's ``materialize`` rewrote into its arch layout
   (``_tl_layout=tw-bf16``) is re-packed THROUGH that same twiddle, so a
   twiddled-layout decode kernel never receives natural nibbles.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from tilerl_kernels.reference import pack_fp4, twiddle_fp4

from tilerl.autograd import Adafactor
from tilerl.config import tiny
from tilerl.model import build_random, drop_quantized, fp4_param_keys, requantize_fp4
from tilerl.testing import RefBackend
from tilerl.train import _training_kv, train_step

#: 27B linears land around this (test_ops_parity packs w * 0.02); N(0,1) saturates
#: the e2m1 grid to the point fp4 parity is meaningless, so the gate scales down.
_W_SCALE = 0.02

#: Distribution bounds for the correct repack, placed in the MEASURED gap between a
#: correct repack (p99 eff-rel 0.150, max abs 0.152) and a wrong-block mutant
#: (p99 0.444, max abs 0.314): correct must pass, the block-16/32 mixup must fail.
_P99_EFF_REL = 0.25
_MAX_ABS = 0.22


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


def _eff_rel(got, ref):
    """Effective relative error per logit: |got-ref| / |ref|, no floor (a near-zero
    logit keeps its own denominator — its error stays in the distribution)."""
    return ((got - ref).abs() / ref.abs()).flatten()


def test_repacked_served_logits_match_the_bf16_master():
    cfg, served = _fp4_tiny()
    _, master = _fp4_tiny()
    drop_quantized(master)
    ids = (np.arange(64) % (cfg.vocab_size - 1)).astype(np.int64).reshape(1, 64)
    ls, lm = _served_logits(served, ids), _served_logits(master, ids)
    # Exact argmax (64/64 on this fixture), plus a distribution bound placed in the
    # gap to a wrong-block repack — not a hand-picked allclose band.
    assert (ls[0].argmax(-1) == lm[0].argmax(-1)).all()
    eff = _eff_rel(ls, lm)
    assert torch.quantile(eff, 0.99).item() < _P99_EFF_REL
    assert (ls - lm).abs().max().item() < _MAX_ABS


def test_the_distribution_bound_discriminates_a_wrong_block_repack():
    """The intended mutant: re-pack at block 16 while the served slot/scale are the
    block-32 face load_hf built. It keeps argmax EXACT (argmax alone is blind to it)
    but blows both distribution bounds — the reason the bound exists instead of a
    token-equality or a single allclose."""
    cfg, bad = _fp4_tiny()
    _, ref = _fp4_tiny()
    _train(bad, 2, requant=False, lr=5e-3)
    _train(ref, 2, requant=False, lr=5e-3)
    # Corrupt every served face the way a block-mismatched repack would: block-16
    # nibbles and a block-16 scale widened into the block-32 slot.
    from tilerl_kernels.reference import renorm_fp4_scale

    with torch.no_grad():
        for k in [x for x in bad.params if x.endswith(".wq")]:
            base = k[:-3]
            m = bad.params[base]
            wq16, sc16 = pack_fp4(m, block=16)
            sc16, os16 = renorm_fp4_scale(sc16)
            bad.params[k].copy_(wq16[:, :bad.params[k].shape[1]])
            bad.params[base + ".scale"].copy_(
                sc16.repeat(1, 2)[:, :bad.params[base + ".scale"].shape[1]])
            bad.params[base + ".oscale"].copy_(os16)
    drop_quantized(ref)
    ids = (np.arange(64) % (cfg.vocab_size - 1)).astype(np.int64).reshape(1, 64)
    lb, lr_ = _served_logits(bad, ids), _served_logits(ref, ids)
    assert (lb[0].argmax(-1) == lr_[0].argmax(-1)).all(), "premise: argmax does not catch it"
    eff = _eff_rel(lb, lr_)
    assert torch.quantile(eff, 0.99).item() > _P99_EFF_REL
    assert (lb - lr_).abs().max().item() > _MAX_ABS


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


def test_a_twiddled_slot_is_repacked_through_the_same_twiddle():
    """A card's materialize rewrites .wq once and tags it (sm90 = tw-bf16). The
    re-pack must re-apply that SAME twiddle: natural pack_fp4 nibbles copied into a
    twiddled slot feed the sm90 decode kernel the wrong byte layout silently
    (RefBackend.materialize is identity, so without this gate all paths are blind).

    Simulated on CPU exactly as materialize would leave the slot: tag tw-bf16 and
    bytes == twiddle_fp4(natural). After a step the slot must equal
    twiddle_fp4(pack_fp4(master)), NOT pack_fp4(master).
    """
    cfg, model = _fp4_tiny()
    key = sorted(k for k in fp4_param_keys(cfg) if f"{k}.wq" in model.params)[0]
    with torch.no_grad():
        nat = model.params[key + ".wq"]
        model.params[key + ".wq"] = twiddle_fp4(nat).contiguous()
        model.params[key + ".wq"]._tl_layout = "tw-bf16"

    _train(model, 2, requant=True)

    master = model.params[key]
    natural, _ = pack_fp4(master, block=master.shape[1] // model.params[key + ".scale"].shape[1])
    slot = model.params[key + ".wq"]
    assert getattr(slot, "_tl_layout", None) == "tw-bf16"
    assert torch.equal(slot, twiddle_fp4(natural))
    assert not torch.equal(slot, natural)  # the red assertion on the pre-fix code


def test_requant_with_no_served_slots_raises():
    """n==0 is a wiring error (full SFT dropped the faces, or a non-fp4 config),
    not a silent success."""
    cfg, model = _fp4_tiny()
    drop_quantized(model)
    with pytest.raises(RuntimeError, match="re-packed nothing"):
        requantize_fp4(model)


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
