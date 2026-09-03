"""End-to-end gates: engine, prefix cache, training, tape, spec decode (hermetic, CPU)."""

from __future__ import annotations

import math
import os
from dataclasses import replace

os.environ.setdefault("TILERL_TARGET", "cpu")

import numpy as np
import pytest
import torch
from tilerl_kernels.backend import get_backend
from tilerl_kernels.reference import dequant_fp4, pack_fp4, top_p_probs, unpack_fp4

from tilerl.autograd import Adafactor, AdamW, RecordingBackend, Tape, clip_grad_norm, cosine_warmup
from tilerl.cli import _build_model
from tilerl.config import tiny
from tilerl.engine import (
    _PHASE_DECODE,
    BLOCK_TOKENS,
    BatchKv,
    Engine,
    SamplingParams,
    _restrict,
    _step_seed,
    build_engine,
)
from tilerl.kv_cache import NoPrefixStore, PagedKvPool
from tilerl.model import add_lora, build_random, fp4_param_keys, param_specs
from tilerl.spec import DraftHead
from tilerl.testing import RefBackend
from tilerl.train import _training_kv, opd_loop, train_step


def _build_engine(seed: int) -> Engine:
    cfg = tiny()
    model = build_random(cfg, seed=seed)
    backend = get_backend()
    return build_engine(
        cfg, model, backend, num_blocks=8, num_slots=4, max_batch=4, max_total_tokens=512
    )


def _drain(engine, request_ids, max_new_tokens: int, max_ticks: int = 512):
    """Step until every request has produced its full length (poll drains, so accumulate)."""
    done: dict = {}
    for _ in range(max_ticks):
        done.update(engine.poll())
        if all(rid in done and len(done[rid]) >= max_new_tokens for rid in request_ids):
            return done
        engine.step()
    raise TimeoutError(f"engine did not finish requests {request_ids} in {max_ticks} ticks")


def test_step_seed_uses_all_seed_bits():
    """Regression: a shift-mask kept only the low 11 bits, so OPD replayed rollouts past step 2048."""
    assert _step_seed(1, 0) != _step_seed(2049, 0)
    assert len({_step_seed(s, 7) for s in range(10000)}) > 9990


def test_generate():
    """Same seed -> identical tokens, different seed -> different tokens."""
    engine = _build_engine(seed=1234)
    try:
        prompt = np.random.default_rng(0).integers(3, 320, size=16).astype(np.int64)
        params_a = SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=16, seed=7)
        params_b = SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=16, seed=7)
        params_c = SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=16, seed=8)
        id_a = engine.submit(prompt, params_a)
        id_b = engine.submit(prompt, params_b)
        id_c = engine.submit(prompt, params_c)
        out = _drain(engine, [id_a, id_b, id_c], max_new_tokens=16)
    finally:
        engine.shutdown()

    toks_a, toks_b, toks_c = out[id_a], out[id_b], out[id_c]
    # a random model samples eos with p ~1/320 per step: no exact-length assert
    assert 1 <= len(toks_a) <= 16
    assert toks_a == toks_b, "same seed must produce identical tokens"
    assert toks_a != toks_c, "different seed must produce different tokens"


def test_prefix_cache():
    """A second prompt sharing a block-aligned prefix adopts the cached blocks."""
    engine = _build_engine(seed=99)
    try:
        rng = np.random.default_rng(1)
        head = rng.integers(3, 320, size=16).astype(np.int64)  # one full block
        tail = rng.integers(3, 320, size=8).astype(np.int64)
        params = SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=8, seed=5)
        id_1 = engine.submit(head, params)
        out_1 = _drain(engine, [id_1], max_new_tokens=8)[id_1]
        id_2 = engine.submit(np.concatenate([head, tail]), params)
        out_2 = _drain(engine, [id_2], max_new_tokens=8)[id_2]
    finally:
        engine.shutdown()

    assert 1 <= len(out_1) <= 8 and 1 <= len(out_2) <= 8
    assert engine.stats()["prefix_hits"] == 1 and engine.stats()["prefix_misses"] == 1


def test_generated_prefix_matches_cold_path():
    cfg = tiny()
    backend = get_backend()
    prompt = np.random.default_rng(4).integers(3, 320, size=14).astype(np.int64)
    params = SamplingParams(temperature=0.0, max_new_tokens=3, seed=0)
    cached = build_engine(
        cfg, build_random(cfg, seed=12), backend, num_blocks=8, max_total_tokens=512
    )
    cold = build_engine(
        cfg, build_random(cfg, seed=12), backend, num_blocks=8, max_total_tokens=512
    )
    first = _drain(cached, [cached.submit(prompt, params)], 3)
    generated = next(iter(first.values()))
    followup = np.concatenate([prompt, generated[:2], np.array([7, 8], dtype=np.int64)])
    next_params = SamplingParams(temperature=0.0, max_new_tokens=2, seed=1)
    cached_id = cached.submit(followup, next_params)
    cold_id = cold.submit(followup, next_params)
    assert _drain(cached, [cached_id], 2)[cached_id] == _drain(cold, [cold_id], 2)[cold_id]
    assert cached.stats()["prefix_hits"] == 1


def test_submit_rollback_and_terminal_failure():
    cfg = tiny()
    engine = build_engine(
        cfg,
        build_random(cfg, seed=3),
        get_backend(),
        num_blocks=2,
        num_slots=1,
        max_total_tokens=32,
    )
    engine.submit([1], SamplingParams(max_new_tokens=1))
    free_blocks = engine._kv.free_blocks
    with pytest.raises(RuntimeError, match="LinearStatePool exhausted"):
        engine.submit([2], SamplingParams(max_new_tokens=1))
    assert engine._kv.free_blocks == free_blocks

    engine._model.forward = lambda *_, **__: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        engine.step()
    with pytest.raises(RuntimeError, match="boom"):
        engine.take(1)
    assert engine.stats()["blocks_used"] == 0
    assert engine.stats()["slots_used"] == 0


def test_decode_growth_evicts_finished_prefix():
    cfg = tiny()
    engine = build_engine(
        cfg,
        build_random(cfg, seed=9),
        get_backend(),
        num_blocks=2,
        num_slots=4,
        max_batch=4,
        max_total_tokens=512,
    )
    # A decodes past a block boundary so its prefix stays pinned after finish.
    rid_a = engine.submit([1, 2, 3], SamplingParams(max_new_tokens=20, seed=1))
    assert len(_drain(engine, [rid_a], 20)[rid_a]) == 20
    assert engine._kv.free_blocks == 1  # A's published block stays pinned
    # B's prompt fits the last free block; its decode growth must evict A.
    rid_b = engine.submit([4, 5, 6], SamplingParams(max_new_tokens=20, seed=2))
    assert len(_drain(engine, [rid_b], 20)[rid_b]) == 20
    assert engine._prefix.stats()["evictions"] >= 1


def test_prefix_snapshots_die_with_their_store_entry():
    """Boundary snapshots are 74.81 MiB each at 27B: eviction must free them."""
    cfg = tiny()
    engine = build_engine(
        cfg,
        build_random(cfg, seed=9),
        get_backend(),
        num_blocks=2,
        num_slots=4,
        max_batch=4,
        max_total_tokens=512,
    )
    for i in range(4):
        rid = engine.submit([i + 1, i + 2, i + 3], SamplingParams(max_new_tokens=20, seed=i))
        assert len(_drain(engine, [rid], 20)[rid]) == 20
    assert engine._prefix.stats()["evictions"] >= 1


def test_prefix_hit_survives_evicting_its_own_entry():
    """submit()'s own evict_until_free can evict the entry it just matched;
    the snapshot must be read before that, not after."""
    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=9), get_backend(), num_blocks=8, num_slots=4)

    def publish(toks, n, state=None):
        blocks = [engine._kv.alloc_block() for _ in range(n)]
        engine._prefix.insert(list(toks), blocks, state)
        for b in blocks:
            engine._kv.free_block(b)  # the store keeps its own retain

    tokens = list(range(1, 4 * BLOCK_TOKENS + 1))
    key = tuple(tokens[: 2 * BLOCK_TOKENS])
    publish(key, 2, (engine._states.states[0].clone(), None))  # matched entry, FIFO head
    publish(range(9000, 9000 + 2 * BLOCK_TOKENS), 2)  # younger, unretained
    [engine._kv.alloc_block() for _ in range(engine._kv.free_blocks - 1)]

    rid = engine.submit(tokens, SamplingParams(max_new_tokens=4))
    assert engine._prefix.stats()["evictions"] >= 1 and rid > 0


def test_stop_token_is_not_returned():
    engine = _build_engine(seed=6)
    engine._sample_batch = lambda rows: [7] * len(rows)
    rid = engine.submit([1, 2], SamplingParams(max_new_tokens=4, stop_token_ids=(7,)))
    engine.step()
    assert engine.take(rid) == []


def test_adafactor_streaming_matches_collecting():
    """Updating each parameter inside backward equals collecting first. At the
    tape+optimizer level: train_step's collecting path also clips the global norm."""
    backend = get_backend()
    ids = np.random.default_rng(7).integers(3, tiny().vocab_size, size=(2, 16)).astype(np.int64)

    def run(streaming: bool) -> dict[str, torch.Tensor]:
        model = build_random(tiny(), seed=2026)
        opt = Adafactor(lr=1e-2)
        for _ in range(3):
            model.params = backend.materialize(model.params)
            by_id = {id(p): p for p in model.params.values()}
            kv = _training_kv(model, 2, 16, device=backend.device)
            tape = Tape()
            with torch.no_grad(), tape:
                logits = model.forward(ids, np.arange(16), kv, RecordingBackend(backend))
            g = torch.ones_like(logits) / logits.numel()
            opt.begin()
            if streaming:
                tape.backward(g, needs=set(by_id),
                              on_grad=lambda t, gr: (t in by_id
                                                     and opt.step_one(by_id[t], gr)) or True)
            else:
                for tid, gr in tape.backward(g, needs=set(by_id)).items():
                    if tid in by_id:
                        opt.step_one(by_id[tid], gr)
        return {k: v.clone() for k, v in model.params.items()}

    streamed, collected = run(True), run(False)
    for k, v in collected.items():
        assert torch.equal(streamed[k], v), f"{k} diverged"


def test_train_step_does_not_sync_per_parameter():
    """One host sync per step (the loss), not two per parameter inside Adafactor.step_one."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class CountSyncs(TorchDispatchMode):
        n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func) == "aten._local_scalar_dense.default":
                CountSyncs.n += 1
            return func(*args, **(kwargs or {}))

    backend = get_backend()
    model = build_random(tiny(), seed=11)
    opt = Adafactor(lr=1e-3)
    ids = [[1, 2, 3, 4, 5, 6, 7, 8]]
    train_step(model, ids, backend, opt)  # warm: first step allocates state
    with CountSyncs():
        train_step(model, ids, backend, opt)
    assert CountSyncs.n <= 2, f"{CountSyncs.n} host syncs per step (expected the loss only)"


def test_adafactor_trains_with_factored_state():
    """Loss falls over 20 steps and a 2D param's second moment is O(rows+cols)."""
    cfg = tiny()
    model = build_random(cfg, seed=2026)
    backend = get_backend()
    optimizer = Adafactor(lr=1e-2)
    batch = np.random.default_rng(7).integers(3, cfg.vocab_size, size=(2, 32)).astype(np.int64)

    losses = [float(train_step(model, batch, backend, optimizer)) for _ in range(20)]
    first5, last5 = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last5 < first5, f"loss did not decrease: {first5:.4f} -> {last5:.4f}"

    for p, state in ((p, optimizer._state[id(p)]) for p in model.params.values()
                     if id(p) in optimizer._state):
        held = sum(t.numel() for t in state)
        assert held == (sum(p.shape) if p.dim() == 2 else p.numel()), \
            f"{tuple(p.shape)}: optimizer holds {held} elements"


def test_train_loss_decreases():
    """20 train steps on a fixed batch: last-5 mean loss < first-5 mean."""
    cfg = tiny()
    model = build_random(cfg, seed=2026)
    backend = get_backend()
    optimizer = AdamW(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    batch = np.random.default_rng(7).integers(3, cfg.vocab_size, size=(2, 32)).astype(np.int64)

    losses = [float(train_step(model, batch, backend, optimizer)) for _ in range(20)]
    first5, last5 = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last5 < first5, f"loss did not decrease: {first5:.4f} -> {last5:.4f}"


def test_recompute_matches_stored_activations():
    """A checkpointed MLP block is replayed in backward instead of stored, so
    its gradients must equal the ones the stored forward gives. The MLP is ~60%
    of a layer's activations and the 27B's group of 8 does not fit without this.
    """
    backend = RefBackend()
    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    ids = np.arange(1, 33, dtype=np.int64).reshape(2, 16) % cfg.vocab_size
    pos = np.arange(16, dtype=np.int64)

    def run(recompute):
        kv = _training_kv(model, 2, 16, device=backend.device)
        tape = Tape(recompute=recompute)
        with torch.no_grad(), tape:
            logits = model.forward(ids, pos, kv, RecordingBackend(backend))
        held = len(tape._entries)
        _, gl = backend.cross_entropy_loss_grad(logits, ids)
        return held, tape.backward(gl)

    stored, ref = run(False)
    replayed, got = run(True)
    assert replayed < stored, f"recompute recorded {replayed} entries, stored {stored}"
    by_id = {id(v): k for k, v in model.params.items()}
    ref = {by_id[k]: v for k, v in ref.items() if k in by_id}
    got = {by_id[k]: v for k, v in got.items() if k in by_id}
    assert set(ref) == set(got), f"recompute lost {sorted(set(ref) - set(got))[:4]}"
    worst = max((ref[k] - got[k]).abs().max().item() for k in ref)
    assert worst < 1e-6, f"recomputed gradients differ by {worst:.2e}"


def test_backward_streaming_matches_collecting():
    """Streaming gradients out of backward equals collecting them, bit for bit
    (the 27B cannot hold every weight gradient at once)."""
    backend = RefBackend()
    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    ids = np.arange(1, 33, dtype=np.int64).reshape(2, 16) % cfg.vocab_size
    b, t = ids.shape
    pos = np.arange(t, dtype=np.int64)

    def run(stream):
        kv = _training_kv(model, b, t, device=backend.device)
        tape = Tape()
        with torch.no_grad(), tape:
            logits = model.forward(ids, pos, kv, RecordingBackend(backend))
        _, gl = backend.cross_entropy_loss_grad(logits, ids)
        if not stream:
            return dict(tape.backward(gl))
        out = {}

        def take(tid, g):
            out[tid] = g.clone()
            return True

        left = tape.backward(gl, on_grad=take)
        assert not left, f"streaming left {len(left)} gradients behind"
        return out

    # by name, not id(): streaming frees tensors earlier and ids get reused
    by_id = {id(v): k for k, v in model.params.items()}
    ref = {by_id[k]: v for k, v in run(False).items() if k in by_id}
    got = {by_id[k]: v for k, v in run(True).items() if k in by_id}
    assert set(ref) == set(got), (
        f"streamed {len(got)} parameter gradients, collected {len(ref)}: "
        f"{sorted(set(ref) ^ set(got))[:4]}"
    )
    assert len(ref) >= 20, f"expected the tiny model's params, got {len(ref)}"
    worst = max((ref[k] - got[k]).abs().max().item() for k in ref)
    assert worst == 0.0, f"streamed gradients differ by {worst:.3e}"


def test_tape_gradcheck():
    """Tape backward vs central finite differences on rmsnorm+linear+CE, in f32
    (bf16 swamps a 1e-3 step)."""
    backend = get_backend()
    if backend.target.startswith("cuda"):
        # Backend._rows casts every activation to bf16 there, whose eps is 7.8e-3,
        # so the 1e-3 step is a tenth of one ulp and rounds away. Measured on H20:
        # a f32-in/f32-out linear differs from the reference by 1.9e-2, 38x this
        # test's atol. The tape's cuda path is covered by tests/test_ops_parity.py.
        pytest.skip("finite differences need an f32 forward; cuda casts to bf16")
    gen = torch.Generator().manual_seed(0)
    batch, dim, vocab = 4, 8, 16
    x = torch.randn(batch, dim, generator=gen, dtype=torch.float32)
    w_norm = torch.randn(dim, generator=gen, dtype=torch.float32)
    w_proj = torch.randn(vocab, dim, generator=gen, dtype=torch.float32) * 0.5
    targets = torch.randint(0, vocab, (batch,), generator=gen)
    eps = 1e-6

    def forward(x_, wn_, wp_):
        # the tape covers rmsnorm+linear; CE and dL/dlogits are torch-eager
        rec = RecordingBackend(backend)
        with Tape() as tape:
            hidden = rec.rmsnorm(x_, wn_, eps)
            logits = rec.linear(hidden, wp_)
        return tape, logits

    def ce_loss(logits):
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return -log_probs[torch.arange(batch), targets].mean()

    _, logits = forward(x, w_norm, w_proj)
    dlogits = torch.softmax(logits.float(), dim=-1)
    dlogits[torch.arange(batch), targets] -= 1.0
    dlogits = (dlogits / batch).to(logits.dtype)

    tape, _ = forward(x, w_norm, w_proj)
    grads = tape.backward(dlogits)
    analytic = {name: grads[id(t)].float() for name, t in (("w_norm", w_norm), ("w_proj", w_proj))}
    step = 1e-3

    def numeric_grad(tensor):
        result = torch.zeros_like(tensor)
        flat = tensor.view(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            flat[i] = orig + step
            loss_plus = ce_loss(forward(x, w_norm, w_proj)[1]).item()
            flat[i] = orig - step
            loss_minus = ce_loss(forward(x, w_norm, w_proj)[1]).item()
            flat[i] = orig
            result.view(-1)[i] = (loss_plus - loss_minus) / (2 * step)
        return result

    for name, tensor in (("w_norm", w_norm), ("w_proj", w_proj)):
        numeric = numeric_grad(tensor)
        expected = analytic[name].cpu()
        assert torch.allclose(expected, numeric, rtol=5e-2, atol=5e-4), (
            f"{name}: tape grad mismatch, max abs diff "
            f"{(expected - numeric).abs().max().item():.2e}"
        )


def test_recording_uses_master_weight_and_consumes_tape():
    backend = RefBackend()
    recording = RecordingBackend(backend)
    x = torch.randn(2, 32)
    master = torch.randn(8, 32)
    wq, scale = pack_fp4(master)
    tape = Tape()
    with tape:
        y = recording.linear_fp4(x, wq, scale, master=master)
    assert torch.allclose(y, backend.linear(x, master))
    grads = tape.backward(torch.ones_like(y))
    assert id(master) in grads and not tape._entries
    with pytest.raises(RuntimeError, match="reused"), tape:
        pass


def test_fp4_train_step():
    """Every fp4 linear keeps its bf16 master under the tape (the STE grad lands on it)."""
    cfg = replace(tiny(), fp4=True)
    model = build_random(cfg, seed=4, keep_master=True)
    assert fp4_param_keys(cfg) <= set(model.params)
    ids = np.arange(3, 11, dtype=np.int64)[None, :]
    assert math.isfinite(train_step(model, ids, RefBackend(), AdamW(lr=1e-3)))


@pytest.mark.parametrize("block", [32, 16])
def test_fp4_roundtrip(block):
    """Values on the e2m1 grid survive pack/unpack at both checkpoint block sizes."""
    gen = torch.Generator().manual_seed(0)
    n_rows, k_cols = 16, 32
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    scale = torch.rand(n_rows, k_cols // block, generator=gen, dtype=torch.float32) * 0.05 + 0.01
    signs = torch.randint(0, 2, (n_rows, k_cols), generator=gen) * 2 - 1
    indices = torch.randint(0, 8, (n_rows, k_cols), generator=gen)
    indices[:, ::block] = 7  # every block holds a 6 so block_max/6 reproduces the scale
    weights = (signs.float() * grid[indices] * scale.repeat_interleave(block, dim=1)).to(
        torch.bfloat16
    )
    dequant = unpack_fp4(*pack_fp4(weights, block))
    assert dequant.shape == weights.shape, f"shape drift: {dequant.shape} vs {weights.shape}"
    max_err = (dequant.float() - weights).abs().max().item()
    assert max_err < 1e-2, f"fp4 roundtrip max error {max_err:.2e} >= 1e-2"


def test_cosine_warmup():
    """Warmup is linear from 0; the peak is lr; the tail is a half-cosine to 0."""
    assert cosine_warmup(0, 100, 10, 1e-3) == 0.0
    assert abs(cosine_warmup(10, 100, 10, 1e-3) - 1e-3) < 1e-12  # peak
    assert abs(cosine_warmup(55, 100, 10, 1e-3) - 0.5e-3) < 1e-12  # cos(pi/2)
    assert cosine_warmup(100, 100, 10, 1e-3) == 0.0  # end


def test_clip_grad_norm():
    g = torch.ones(4)
    grads = {0: g.clone()}
    pre = clip_grad_norm(grads, 1.0)
    assert abs(pre - 2.0) < 1e-6  # sqrt(4)
    assert abs(grads[0].norm().item() - 1.0) < 1e-6
    grads = {0: torch.full((4,), 0.1)}
    pre = clip_grad_norm(grads, 1.0)
    assert abs(pre - 0.2) < 1e-6 and torch.allclose(grads[0], torch.full((4,), 0.1))
    grads = {0: torch.tensor([float("nan"), 1.0])}
    assert not math.isfinite(clip_grad_norm(grads, 1.0))


def test_production_model_gradcheck():
    """The full tiny model under a tape: a finite grad for every param, and
    central finite differences on params from different layers.
    # A CUDA gradcheck read 8.1e-2 at step 0.1 and 2.6e-1 at 0.025 while the
    # tape held 0.4%: the probe, not the tape, was wrong (errors/, 2026-08-28).
    """
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
        # the production CE: a local re-derivation once hid 9/9 injected corruptions
        loss, dlogits = backend.cross_entropy_loss_grad(logits, batch)
        return loss, tape.backward(dlogits)

    loss, grads = loss_and_grads()
    assert math.isfinite(loss)
    specs = param_specs(cfg)
    for k in specs:
        g = grads.get(id(model.params[k]))
        assert g is not None, f"no grad for param {k}"
        assert torch.isfinite(g).all(), f"non-finite grad for {k}"

    # one element each from embed, a full-attn weight, a GDN weight, final_norm;
    # worst clean rel error is 3.6%, rtol=0.1 catches every injected corruption.
    # A bf16 central difference whose slope moves with the step size cannot
    # judge a gradient, so an inconsistent probe is skipped, not blamed on the tape.
    step = 0.1
    checked = 0
    # down_proj is inside the checkpointed MLP block: its gradient comes from a
    # replayed forward, so it needs the finite difference as much as the rest.
    for key in ("embed_tokens", "layers.0.q_proj", "layers.1.in_proj_a", "final_norm",
                "layers.0.down_proj"):
        p = model.params[key]
        idx = (0, 0) if p.ndim == 2 else (0,)
        analytic = grads[id(p)][idx].item()
        nums = []
        for h in (step, step / 2, step / 4):
            orig = p[idx].item()
            p[idx] = orig + h
            lp, _ = loss_and_grads()
            p[idx] = orig - h
            lm, _ = loss_and_grads()
            p[idx] = orig
            nums.append((float(lp) - float(lm)) / (2 * h))
        mean = sum(nums) / len(nums)
        spread = (max(nums) - min(nums)) / max(abs(mean), 1e-12)
        if spread > 0.25:
            continue
        checked += 1
        numeric = min(nums, key=lambda n: abs(n - analytic))
        assert abs(analytic - numeric) < 0.1 * abs(numeric), (
            f"{key}: tape {analytic:.4e} vs numeric {numeric:.4e} (probe spread {spread:.1%})"
        )
    assert checked, "no finite-difference probe was numerically sound"


def test_logprobs_are_returned_and_deterministic():
    """Every sampled token carries log p under the distribution it was drawn
    from: one per token, never positive, reproduced by the same seed."""
    backend = get_backend()
    cfg, model = _build_model("tiny", seed=0)
    engine = build_engine(cfg, model, backend, num_blocks=64, num_slots=8)
    sp = SamplingParams(temperature=0.7, max_new_tokens=4, seed=3, logprobs=True)

    def run():
        rid = engine.submit(list(range(8)), sp)
        for _ in range(200):
            engine.step()
            done = engine.poll()  # poll drains, so read the tokens from it
            if rid in done:
                return done[rid], engine.logprobs(rid)
        raise AssertionError("request never finished")

    out, lps = run()
    assert lps is not None and len(lps) == len(out) == 4, (out, lps)
    assert all(x <= 1e-5 for x in lps), lps
    # The value, not just the shape: log q under the sampler's own nucleus. A
    # log_softmax over the un-renormalized logits reads low by log(kept mass),
    # which every shape assertion above accepts.
    sp_tp = replace(sp, top_p=0.8, top_k=20, max_new_tokens=3, seed=11)
    fresh = build_engine(cfg, model, backend, num_blocks=64, num_slots=8,
                         decode_graph=False, prefix_store=NoPrefixStore())
    rid = fresh.submit(list(range(8)), sp_tp)
    for _ in range(200):
        fresh.step()
        done = fresh.poll()
        if rid in done:
            toks = done[rid]
            break
    scored = fresh.logprobs(rid)
    seq = np.asarray(list(range(8)) + list(toks), dtype=np.int64)[None, :]
    kv = _training_kv(model, 1, seq.shape[1], device=backend.device)
    with torch.no_grad():
        dense = model.forward(seq, np.arange(seq.shape[1], dtype=np.int64), kv, backend)
    for i, tok in enumerate(toks):
        row = _restrict(dense[0, 7 + i].unsqueeze(0), sp_tp) / sp_tp.temperature
        probs, order = top_p_probs(row, sp_tp.top_p)
        want = float(probs[0, (order[0] == tok).nonzero().item()].log())
        assert abs(scored[i] - want) < 2e-3, f"token {i}: reported {scored[i]} vs log q {want}"
    # greedy must not report the point mass (every logprob exactly 0)
    g = SamplingParams(temperature=0.0, max_new_tokens=3, seed=1, logprobs=True)
    rid = engine.submit(list(range(8)), g)
    for _ in range(200):
        engine.step()
        if rid in engine.poll():
            break
    greedy_lps = engine.logprobs(rid)
    assert greedy_lps and all(x < -1e-6 for x in greedy_lps), greedy_lps
    assert (out, lps) == run(), "same seed, different scores"
    # a second read raises: None would be indistinguishable from "never asked"
    with pytest.raises(KeyError, match="already taken"):
        engine.logprobs(rid)
    rid = engine.submit(list(range(8)), SamplingParams(max_new_tokens=2, seed=3))
    for _ in range(200):
        engine.step()
        if rid in engine.poll():
            break
    assert engine.logprobs(rid) is None


def test_opd_lora_self_teacher():
    """OPD with LoRA: the frozen base stays bit-identical and some adapter moves."""
    backend = get_backend()
    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    teacher = build_engine(cfg, model, backend, num_blocks=64, num_slots=4, decode_graph=False, prefix_store=NoPrefixStore())
    trainable = add_lora(model, rank=4)
    assert trainable, "add_lora attached nothing: nothing for the tape to train"
    base = {k: v.clone() for k, v in model.params.items() if k not in trainable}
    before = {k: v.clone() for k, v in trainable.items()}
    prompts = [list(range(8)), list(range(4, 12))]
    losses = opd_loop(teacher, model, prompts, steps=2, backend=backend,
                      optimizer=AdamW(lr=1e-2), seed=0, trainable=trainable)
    assert len(losses) == 2 and all(l == l for l in losses), losses
    for k, v in base.items():
        assert torch.equal(model.params[k], v), f"frozen base moved: {k}"
    moved = sum(not torch.equal(trainable[k], v) for k, v in before.items())
    assert moved, "no adapter moved"


def test_prefix_snapshot_includes_conv_window():
    cfg = tiny()
    engine = _build_engine(seed=5)
    try:
        prompt = np.random.default_rng(0).integers(3, 320, size=16).astype(np.int64)
        engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
        for _ in range(32):
            engine.step()
            if engine.stats()["prefix_published"]:
                break
        hit = engine._prefix.lookup(list(prompt))
        assert hit is not None, "no prefix snapshot published"
        states, windows = hit.state
        assert windows is not None
        assert windows.shape[-2] == cfg.linear_conv_kernel_dim - 1
    finally:
        engine.shutdown()


def test_concurrent_prefills_not_starved():
    """Three concurrent requests reach decode together (same-width prefills pack into one forward)."""
    engine = _build_engine(seed=77)
    try:
        prompt = np.random.default_rng(0).integers(3, 320, size=8).astype(np.int64)
        params = SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=8, seed=3)
        ids = [engine.submit(prompt, params) for _ in range(3)]
        ticks = 0
        for _ in range(16):
            engine.step()
            ticks += 1
            if sum(1 for r in engine._running if r.phase == _PHASE_DECODE) == 3:
                break
        assert sum(1 for r in engine._running if r.phase == _PHASE_DECODE) == 3
        assert ticks <= 2, f"three same-length prompts took {ticks} ticks to prefill"
        out = _drain(engine, ids, max_new_tokens=8)
    finally:
        engine.shutdown()
    assert all(1 <= len(out[i]) <= 8 for i in ids)


def test_chunked_prefill_matches_one_shot():
    """Chunked and one-shot prefill produce identical tokens."""
    prompt = np.random.default_rng(0).integers(3, 320, size=40).astype(np.int64)
    params = SamplingParams(temperature=0.0, max_new_tokens=4, seed=11)
    outs = []
    for budget in (512, 16):
        cfg = tiny()
        engine = build_engine(
            cfg,
            build_random(cfg, seed=5),
            get_backend(),
            num_blocks=8,
            num_slots=4,
            max_batch=4,
            max_total_tokens=512,
            max_num_batched_tokens=budget,
        )
        try:
            rid = engine.submit(prompt, params)
            out = _drain(engine, [rid], max_new_tokens=4)[rid]
        finally:
            engine.shutdown()
        outs.append(out)
    assert outs[0] == outs[1], f"chunked prefill diverged: {outs[0]} vs {outs[1]}"
    assert 1 <= len(outs[1]) <= 4


def test_gpu_targets():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available on this host")
    target = "cuda"
    prev = os.environ.get("TILERL_TARGET")
    os.environ["TILERL_TARGET"] = target
    from tilerl_kernels import backend as backend_mod

    backend_mod._BACKEND = None
    try:
        backend = backend_mod.get_backend()
        assert backend.target == target, f"resolved {backend.target!r}, want {target!r}"
        x = torch.randn(4, 4, dtype=torch.float32, device=backend.device)
        w = torch.randn(4, dtype=torch.float32, device=backend.device)
        y = backend.rmsnorm(x, w, eps=1e-6)
        assert y.device == backend.device
    finally:
        backend_mod._BACKEND = None
        if prev is None:
            os.environ.pop("TILERL_TARGET", None)
        else:
            os.environ["TILERL_TARGET"] = prev


def test_frozen_fp4_base_gives_dx_only():
    """No master = frozen base: dX flows, the quantized weight gets no gradient."""
    backend = RefBackend()
    recording = RecordingBackend(backend)
    x = torch.randn(2, 32)
    w = torch.randn(8, 32)
    wq, scale = pack_fp4(w)
    with Tape() as tape:
        y = recording.linear_fp4(x, wq, scale)
    g = torch.randn_like(y)
    grads = tape.backward(g)
    assert set(grads) == {id(x)}
    assert torch.allclose(grads[id(x)], g @ dequant_fp4(wq, scale), atol=1e-4)


def test_lora_train_step_on_frozen_fp4_base():
    """Frozen fp4 base + LoRA: B=0 at step 0, then only the adapters move."""
    cfg = replace(tiny(), fp4=True)
    model = build_random(cfg, seed=4)  # no keep_master: the base is frozen
    backend = RefBackend()
    ids = np.arange(3, 11, dtype=np.int64)[None, :]
    base_logits = model.forward(ids, np.arange(ids.shape[1]), _training_kv(model, 1, ids.shape[1]), backend)
    new = add_lora(model, rank=4, seed=1)
    assert new and all(k.endswith((".lora_a", ".lora_b")) for k in new)
    after = model.forward(ids, np.arange(ids.shape[1]), _training_kv(model, 1, ids.shape[1]), backend)
    assert torch.allclose(base_logits, after)  # B = 0

    before = {k: v.clone() for k, v in model.params.items()}
    assert math.isfinite(train_step(model, ids, backend, AdamW(lr=1e-2), trainable=new))
    moved = {k for k in before if not torch.equal(model.params[k], before[k])}
    assert moved and moved <= set(new)  # adapters move, the quantized base does not


def test_opd_ema_self_teacher_shares_the_model():
    """Self-teacher OPD: one model, one engine, the teacher on an EMA of the adapters."""
    cfg = replace(tiny(), fp4=True)
    model = build_random(cfg, seed=7)
    backend = get_backend()
    # build_engine first: it materializes the params the adapters must point at.
    teacher = build_engine(cfg, model, backend, num_blocks=8, num_slots=4, max_batch=4,
                           max_total_tokens=512, decode_graph=False, prefix_store=NoPrefixStore())
    trainable = add_lora(model, rank=4, seed=2)
    before = {k: v.clone() for k, v in model.params.items()}
    try:
        prompts = [np.random.default_rng(0).integers(3, cfg.vocab_size, size=8).astype(np.int64)]
        losses = opd_loop(teacher, model, prompts, steps=2, backend=backend, seed=0,
                          trainable=trainable, ema_decay=0.5)
    finally:
        teacher.shutdown()
    assert len(losses) == 2 and all(math.isfinite(x) for x in losses)
    moved = {k for k in before if not torch.equal(model.params[k], before[k])}
    # sm90's first served forward rewrites .wq into the twiddled layout in place
    # (Backend._served_fp4); that is a layout change, not a weight update.
    twiddled = {k for k in before if getattr(model.params[k], "_tl_twiddled", False)}
    assert moved and moved <= set(trainable) | twiddled


def test_max_think_tokens_forces_the_block_closed():
    """After the budget the engine emits end_think_ids, then sampling resumes."""
    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=11), get_backend(), num_blocks=8,
                          num_slots=4, max_batch=4, max_total_tokens=512)
    end = (5, 6)
    try:
        wid = engine.submit(
            [3, 4, 5],
            SamplingParams(temperature=0.0, max_new_tokens=8, seed=0,
                           max_think_tokens=2, end_think_ids=end),
        )
        out = None
        for _ in range(200):
            engine.step()
            if wid in (done := engine.poll()):
                out = done[wid]
                break
    finally:
        engine.shutdown()
    assert out is not None and len(out) == 8
    assert tuple(out[2:4]) == end  # forced at the budget, then sampling resumes


class _OracleDraft(DraftHead):
    """A draft head proposing the trunk's own continuation: full acceptance
    every tick, so the verify path (chain KV, GDN state selection, multi-token
    commit) is exercised; a random head is rejected at position 0. Inherits the
    drafter contract and replaces only the two calls ``step`` makes into it."""

    def __init__(self, cfg, expected: dict[int, int]):
        self.cfg = replace(cfg, num_layers=1, full_attn_layers=(0,))
        self.params: dict = {}
        self.expected = expected  # absolute position -> token
        self.width = 3
        self.has_confidence = False

    def forward(self, hidden, ids, positions, kv, backend, hidden_out=None):
        pos = np.atleast_2d(np.asarray(positions))
        logits = torch.zeros(*pos.shape, self.cfg.vocab_size, device=backend.device)
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                logits[i, j, self.expected.get(int(pos[i, j]) + 1, 0)] = 10.0
        if hidden_out is not None:
            hidden_out.append(torch.as_tensor(hidden))
        return logits

    def confidence(self, hidden, probs, backend):
        return probs


def _random_draft(cfg, seed: int, trunk):
    dcfg = replace(cfg, num_layers=1, full_attn_layers=(0,), fp4=False)
    params = {k: v for k, v in build_random(dcfg, seed=seed).params.items()
              if k.startswith("layers.")}
    h = cfg.hidden_size
    gen = torch.Generator().manual_seed(seed)
    params["fc"] = (torch.randn(h, 2 * h, generator=gen) * 0.02).to(torch.bfloat16)
    params["norm"] = torch.ones(h, dtype=torch.bfloat16)
    params["pre_fc_norm_hidden"] = torch.ones(h, dtype=torch.bfloat16)
    return DraftHead(trunk, params, num_layers=1)


def _spec_run(prompt, n, draft=None, depth=3):
    cfg = tiny()
    model = build_random(cfg, seed=7)
    engine = build_engine(
        cfg, model, get_backend(), num_blocks=16, num_slots=4, max_batch=4,
        max_total_tokens=512, draft=None if draft is None else draft(cfg, model),
        spec_depth=depth,
    )
    rid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=n, seed=0))
    out = _drain(engine, [rid], n)[rid]
    return out, engine.stats()


def _full_context_draft(cfg, model, draft, backend, toks: list[int]) -> torch.Tensor:
    """draft_check.py's shape: the head teacher-forced over the whole sequence;
    returns its logits at position len(toks) - 1."""
    n = len(toks) - 1
    hid: list = []
    model.forward(np.array([toks]), np.arange(len(toks)),
                  _training_kv(model, 1, len(toks), device=backend.device),
                  backend, hidden_out=hid, last_only=False)
    nblk = -(-n // BLOCK_TOKENS) + 1
    kv = BatchKv(
        block_table=torch.arange(nblk, dtype=torch.long).reshape(1, nblk),
        seq_len=torch.tensor([n]), state_slot=torch.zeros(1, dtype=torch.long),
        kv_pool=PagedKvPool(nblk, cfg.num_kv_heads, cfg.head_dim, num_layers=1,
                            device=backend.device, layer_map=(0,)),
        state_pool=None, seq_q_lens=torch.tensor([n]),
    )
    return draft.forward(hid[-1][:, :n], np.array([toks[1:]]),
                         np.arange(1, n + 1), kv, backend)[0, -1].float()


@pytest.mark.parametrize(
    "rows,plen,batched_tokens,depth",
    [
        (1, 6, 512, 1),    # one row, prompt in one chunk, one draft
        (3, 6, 512, 1),    # ragged widths: rows commit different counts after a reject
        (1, 24, 8, 1),     # chunked prefill: the prompt spans several forwards
        (1, 6, 512, 2),    # a chain, so a rejected step leaves stale KV behind it
    ],
    ids=["single", "multirow", "chunked", "depth2"],
)
def test_engine_draft_matches_full_context_draft(rows, plen, batched_tokens, depth):
    """The draft the engine runs equals the draft draft_check.py measures. A
    context-starved draft is still CORRECT (rejected drafts cost throughput,
    never output), so no other spec test sees a broken KV fill; the
    parametrization covers ragged widths, chunked prefill and stale chain KV."""
    cfg = tiny()
    backend = get_backend()
    model = build_random(cfg, seed=7)
    draft = _random_draft(cfg, 21, model)
    engine = build_engine(
        cfg, model, backend, num_blocks=64, num_slots=8, max_batch=8,
        max_total_tokens=512, max_num_batched_tokens=batched_tokens,
        draft=draft, spec_depth=depth,
    )
    seen: dict[int, tuple] = {}
    step = {"n": 0}
    inner = draft.forward

    def spy(hidden, ids, positions, kv, be, hidden_out=None):
        out = inner(hidden, ids, positions, kv, be, hidden_out=hidden_out)
        # chain step 0 on full-batch ticks only: later steps consume the draft's
        # own hidden, and a partial batch would shift row -> request
        if step["n"] % max(depth, 1) == 0 and out.shape[0] == rows:
            pos = np.asarray(positions)
            for i in range(rows):
                seen[i] = (int(pos[i][-1]), out[i, -1].detach().float().clone())
        step["n"] += 1
        return out

    draft.forward = spy
    prompts = [[3 + (i + r) % 40 for i in range(plen + r)] for r in range(rows)]
    ids = [engine.submit(p, SamplingParams(temperature=0.0, max_new_tokens=8, seed=r))
           for r, p in enumerate(prompts)]
    outs = _drain(engine, ids, 8)
    draft.forward = inner
    assert len(seen) == rows, f"the draft never ran on a full {rows}-row tick"
    if batched_tokens < plen:
        assert engine.stats()["prefill_forwards"] > rows, "the prompt did not chunk"

    for i, (pos, got) in sorted(seen.items()):
        # greedy is deterministic: the final sequence truncated is the sequence at that draft
        row = prompts[i] + outs[ids[i]]
        full = _full_context_draft(cfg, model, draft, backend, row[: pos + 1])
        assert int(full.argmax()) == int(got.argmax()), (
            f"row {i}: engine drafted {int(got.argmax())} at position {pos}, "
            f"full context drafts {int(full.argmax())}"
        )
        # dense vs paged hidden differ ~4e-3, x10 through the head; a chain-local KV reads ~1.4
        rel = ((full - got).norm() / full.norm()).item()
        assert rel < 0.1, f"row {i} at position {pos}: norm-relative {rel:.2e}"


def test_speculation_reproduces_greedy_decode():
    """With a draft attached the engine emits exactly what it emits without
    one: rejected (random head) and fully accepted (oracle head, GDN state rewind)."""
    prompt, n = [3, 4, 5, 6], 24
    base, _ = _spec_run(prompt, n)
    assert len(base) == n

    rand, rstats = _spec_run(prompt, n, draft=lambda cfg, m: _random_draft(cfg, 21, m))
    assert rand == base, f"random draft changed the output: {rand} != {base}"
    assert rstats["spec_drafted"] > 0

    expected = {i: t for i, t in enumerate(prompt + base)}
    spec, sstats = _spec_run(prompt, n, draft=lambda cfg, m: _OracleDraft(cfg, expected))
    assert spec == base, f"oracle draft changed the output: {spec} != {base}"
    assert sstats["spec_accepted"] > sstats["spec_drafted"] * 0.9, sstats

    # chains trimmed below spec_depth write narrower step planes than the pool's.
    # Patch tilerl.spec: DraftHead.step resolves verify_lens in that namespace, so
    # patching any other module's copy leaves this arm untrimmed and identical to
    # the one above it.
    import tilerl.spec as spec_mod

    orig, spec_mod.verify_lens = spec_mod.verify_lens, lambda surv: [2] * len(surv)
    try:
        trimmed, tstats = _spec_run(prompt, n, draft=lambda cfg, m: _OracleDraft(cfg, expected))
    finally:
        spec_mod.verify_lens = orig
    assert tstats["spec_drafted"] < sstats["spec_drafted"], (
        f"the trim never took effect: {tstats['spec_drafted']} drafted, same as untrimmed"
    )
    assert trimmed == base, f"trimmed chain changed the output: {trimmed} != {base}"


def test_verify_commits_the_trunks_own_draw():
    """The two properties speculation guarantees on EVERY backend, which string
    equality above only implies on the CPU reference: a committed token is this
    verify tick's own draw from the trunk at that chain position, and the state
    adopted with it is a step plane the same tick wrote. The second is not free
    -- ``alloc_slot`` does not zero the step planes, so a plane written by an
    earlier tick is a previous owner's state. Slots are reused across waves."""
    prompt, n, depth = [3, 4, 5, 6], 24, 7
    base, _ = _spec_run(prompt, n)
    expected = {i: t for i, t in enumerate(prompt + base)}
    cfg = tiny()
    backend = get_backend()
    model = build_random(cfg, seed=7)
    # the oracle head is accepted whole, so ``n_ok`` reaches the top step plane
    engine = build_engine(cfg, model, backend, num_blocks=64, num_slots=1, max_batch=4,
                          max_total_tokens=512, draft=_OracleDraft(cfg, expected),
                          spec_depth=depth)
    written: set[tuple[int, int]] = set()
    undrawn: list = []
    stale: list = []
    deep: list = []
    step, verify, scatter = engine.step, engine._verify, backend.state_scatter
    decode, select = backend.gdn_decode, engine._states.select_step

    def note(slots, planes):
        written.update((int(s), p) for s in torch.as_tensor(slots).reshape(-1).tolist()
                       for p in range(planes))

    def w_step():
        written.clear()
        return step()

    def w_scatter(states, windows, slots, layer, new_state, new_window, parity=None, steps=False):
        if steps:
            note(slots, new_state.shape[1])
        return scatter(states, windows, slots, layer, new_state, new_window, parity, steps)

    def w_decode(q, k, v, g, beta, pool, slots, layer, keep_steps=0, **kw):
        out = decode(q, k, v, g, beta, pool, slots, layer, keep_steps=keep_steps, **kw)
        if out is not None and keep_steps:  # sm90 writes the planes inside the kernel
            note(slots, keep_steps)
        return out

    def w_select(slot, plane):
        if (int(slot), int(plane)) not in written:
            stale.append((int(slot), int(plane), sorted(written)))
        deep.append(int(plane))
        return select(slot, plane)

    def w_verify(rows, chains, logits, hidden):
        before = [len(r.output) for r in rows]
        out = verify(rows, chains, logits, hidden)
        for i, (r, n0) in enumerate(zip(rows, before)):
            for j in range(len(r.output) - n0):  # temperature 0: the tile's own argmax
                if r.output[n0 + j] != int(logits[i, j].argmax()):
                    undrawn.append((r.req_id, n0 + j))
        return out

    engine.step, engine._verify, engine._states.select_step = w_step, w_verify, w_select
    backend.state_scatter, backend.gdn_decode = w_scatter, w_decode
    try:
        outs = []
        for _ in range(3):  # submit() takes the slot, so reuse needs a drain between waves
            rid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=n, seed=0))
            outs.append(_drain(engine, [rid], n)[rid])
    finally:
        backend.state_scatter, backend.gdn_decode = scatter, decode
    assert not undrawn, f"committed a token the verify tick did not draw: {undrawn[:5]}"
    assert not stale, f"adopted step planes this tick never wrote: {stale[:2]}"
    # coverage: the whole chain was accepted at least once, so the top plane was adopted
    assert max(deep) == depth, f"deepest plane adopted was {max(deep)}, not {depth}"
    assert outs == [base] * 3, "a reused slot changed the output"


def test_generate_fans_a_corpus_across_workers(tmp_path):
    """Offline batch generation through the real subprocess path: every prompt back exactly once."""
    import json

    from tilerl.generate import generate

    src = tmp_path / "prompts.jsonl"
    with open(src, "w") as f:
        for i in range(5):
            f.write(json.dumps({"token_ids": [3 + i, 7, 11, 13]}) + "\n")
    out = tmp_path / "out.jsonl"

    stats = generate(str(src), str(out), devices=[0], source=None, max_new_tokens=3,
                     max_batch=4)

    assert stats["prompts"] == 5 and stats["rows"] == 5, stats
    with open(out) as f:
        rows = [json.loads(x) for x in f]
    assert sorted(r["index"] for r in rows) == list(range(5)), "a prompt was lost or doubled"
    assert all(r["finished"] and r["output_ids"] for r in rows), rows
    assert not list(tmp_path.glob("*.part*")), "per-worker parts must be cleaned up"


def test_noprefix_store_retains_no_snapshot():
    """Regression: training engines (NoPrefixStore) leaked one state clone per block boundary."""
    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=3), get_backend(), num_blocks=8,
                          max_total_tokens=512, decode_graph=False, prefix_store=NoPrefixStore())
    prompt = np.random.default_rng(5).integers(3, 320, size=40).astype(np.int64)
    _drain(engine, [engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=40, seed=0))], 40)
    assert engine.stats()["prefix_published"] == 0
