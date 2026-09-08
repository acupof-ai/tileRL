"""Eager vs captured decode parity (CUDA only).

The captured decode tick (_DecodeGraph in engine.py) must produce the same
token stream as the eager path on the same inputs: same weights, same prompt,
greedy sampling => identical tokens. The ``verify`` arm runs the same check on
a width-2 tick, where the fused GDN and paged-attention decode kernels
carry the whole chain in one graph. Runs on the pod CUDA target; skips on
CPU/metal (no CUDA graphs there).

Run: TILERL_TARGET=cuda uv run pytest tests/test_decode_graph.py -v
"""

from __future__ import annotations

import os
import warnings

# Hermetic default: auto maps to cpu on this Mac; the test skips off-CUDA.
os.environ.setdefault("TILERL_TARGET", "cpu")

from dataclasses import replace

import pytest
import torch
from tilerl_kernels.backend import get_backend

from tilerl.config import tiny
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import build_random
from tilerl.spec import DraftHead


def _draft(cfg, trunk):
    seed = 21
    dcfg = replace(cfg, num_layers=1, full_attn_layers=(0,), fp4=False)
    params = {k: v for k, v in build_random(dcfg, seed=seed).params.items()
              if k.startswith("layers.")}
    gen = torch.Generator().manual_seed(seed)
    h = cfg.hidden_size
    params["fc"] = (torch.randn(h, 2 * h, generator=gen) * 0.02).to(torch.bfloat16)
    params["norm"] = torch.ones(h, dtype=torch.bfloat16)
    params["pre_fc_norm_hidden"] = torch.ones(h, dtype=torch.bfloat16)
    return DraftHead(trunk, params, num_layers=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph decode is CUDA-only")
@pytest.mark.parametrize("spec", [False, True], ids=["decode", "verify"])
def test_decode_graph_matches_eager(spec):
    backend = get_backend()
    cfg = tiny()
    prompt = torch.randint(
        0, cfg.vocab_size, (16,), generator=torch.Generator().manual_seed(11)
    ).tolist()
    params = SamplingParams(temperature=0.0, max_new_tokens=6, seed=3)

    def engine(decode_graph):
        model = build_random(cfg, seed=7)
        return build_engine(
            cfg, model, backend, num_blocks=8, num_slots=2, decode_graph=decode_graph,
            draft=_draft(cfg, model) if spec else None, spec_depth=1,
        )

    eager, captured = engine(False), engine(True)
    we = eager.submit(prompt, params)
    wc = captured.submit(prompt, params)
    for _ in range(64):
        eager.step()
        captured.step()
        pe, pc = eager.poll(), captured.poll()
        if we in pe or wc in pc:
            assert pe.get(we) == pc.get(wc), f"eager {pe.get(we)} vs captured {pc.get(wc)}"
            # A capture failure degrades to eager with a warning — that would
            # make the parity check vacuous. Require the graph to exist, and at
            # the verify width require the wide one, not just the W=1 fallback.
            widths = {w for _, w in captured._decode_graphs}
            assert captured._decode_graph_on and widths, "decode graph capture fell back to eager"
            assert not spec or max(widths) > 1, f"no verify-width graph captured: {widths}"
            return
    raise AssertionError("requests did not finish")


def test_the_graphs_padding_row_is_not_taken_from_the_callers_capacity():
    """``num_slots=N`` must serve N concurrent requests with the graph on.

    A replay's padding rows write to the state and KV pools, so they need a slot
    and a block. Taking those from the pools the caller sized left N slots
    serving N-1, and the N-th ``submit`` raised from ``alloc_slot`` with no
    fallback. Runs on any target: what is gated is the reservation and the
    accounting, not the capture, which only CUDA does.
    """
    cfg, backend = tiny(), get_backend()
    n = 3

    def engine(decode_graph):
        return build_engine(cfg, build_random(cfg, seed=7), backend, num_blocks=16,
                            num_slots=n, max_batch=n, decode_graph=decode_graph)

    on, off = engine(True), engine(False)
    # The pad row is engine overhead: the pool grows by it, the reported capacity does not.
    assert on._states.num_slots == n + 1 and on._kv.num_blocks == 16 + 1
    assert on._pad_slot is not None and on._pad_block is not None
    assert on.stats()["slots_total"] == n and on.stats()["blocks_total"] == 16
    # Negative control: with the graph off nothing is reserved and nothing is added.
    assert off._states.num_slots == n and off._kv.num_blocks == 16
    assert off._pad_slot is None and off.stats()["slots_total"] == n

    prompt = torch.randint(0, cfg.vocab_size, (8,),
                           generator=torch.Generator().manual_seed(5)).tolist()
    params = SamplingParams(temperature=0.0, max_new_tokens=2, seed=0)
    ids = [on.submit(prompt, params) for _ in range(n)]  # the N-th used to raise
    on.step()  # slots are taken at admission now, not in submit
    assert len(set(ids)) == n and on.stats()["slots_used"] == n


@pytest.mark.parametrize("decode_graph", [False, True], ids=["eager", "graph"])
def test_submitting_past_usable_slots_queues_rather_than_raising(decode_graph):
    """The warning at engine.py:402 must describe what over-subscription does.

    Two clauses of it were false, both stale from the pad-row fix, so both are
    asserted here:

    * ``submit raises beyond it`` -- it does not. ``submit`` checks only the
      prompt, the stop texts, ``max_total_tokens`` and the KV pool; the slot is
      taken in ``_admit``, which returns False on ``free_slots < 1``. The excess
      queues. That is the worst of the three for a benchmark arm: a raise kills
      the run and a drop shows in the counts, while queuing produces a table
      that looks finished at half the intended concurrency.
    * ``Pass num_slots >= max_batch + 1 for the pad row`` -- ``build_engine``
      already adds it, so ``usable_slots == num_slots`` on both arms and a
      caller who followed that advice over-allocated.

    ``decode_graph`` is parametrized rather than left to default because
    ``_graph_on`` resolves None to ``device.type == "cuda"``: on CPU the pad
    branch would never run, and the second clause -- the one about the pad --
    would have no coverage on the machine CI uses. The reservation is pure
    Python in ``build_engine``, so both arms run anywhere; only the capture is
    CUDA-only.
    """
    cfg, backend = tiny(), get_backend()
    usable, over = 4, 8
    prompt = torch.randint(0, cfg.vocab_size, (8,),
                           generator=torch.Generator().manual_seed(5)).tolist()
    params = SamplingParams(temperature=0.0, max_new_tokens=2, seed=0)

    def run(num_slots):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            e = build_engine(cfg, build_random(cfg, seed=7), backend, num_blocks=64,
                             num_slots=num_slots, max_batch=over,
                             max_total_tokens=1024, decode_graph=decode_graph)
        # The pad row is the engine's, so the pool grows by it and usable does not:
        # num_slots >= max_batch is exact through build_engine on either arm.
        assert e._states.num_slots == num_slots + decode_graph
        assert e.usable_slots == num_slots
        ids = [e.submit(prompt, params) for _ in range(over)]  # no raise past the slots
        assert len(set(ids)) == over
        widths, done = [], set()
        for _ in range(64):
            e.step()
            widths.append(e.stats()["slots_used"])
            done |= set(e.poll())
            if len(done) == over:
                break
        texts = [str(w.message) for w in caught if "usable state slots" in str(w.message)]
        return set(ids), done, max(widths), texts

    ids, done, peak, texts = run(usable)
    # Every row ran -- queued, not dropped -- and never more than usable at once.
    assert done == ids, f"{len(done)} of {over} finished"
    assert peak == usable, f"peak concurrency {peak} != {usable}"
    # The message is the artifact that misled two sessions, so assert its text: it
    # must say queues, and its remedy must name `num_slots` at max_batch exactly --
    # the old "+ 1" sent build_engine callers one slot over.
    assert len(texts) == 1, texts
    assert "queues" in texts[0] and "raise" not in texts[0].replace("raising", "")
    assert f"num_slots >= {over}" in texts[0], texts[0]
    # The pad row is named only where one exists, and the direct-pool number carries
    # it: that half had no coverage while decode_graph defaulted to False on CPU.
    assert ("pad row" in texts[0]) == decode_graph, texts[0]
    assert f"LinearStatePool for {over + decode_graph}" in texts[0], texts[0]

    # Negative control: the slot count is what bound it, not the planner or the
    # prompt. With room for all 8 the same submits run at width 8, and no warning.
    _, wide_done, wide_peak, wide_texts = run(over)
    assert len(wide_done) == over and wide_peak == over, f"control peaked at {wide_peak}"
    assert wide_texts == [], wide_texts


def test_the_kv_guard_measures_usable_capacity_not_the_pool():
    """``submit``'s KV guard must compare against capacity net of the pad row.

    The pools are sized one larger when the captured tick is on, so a guard
    reading ``self._kv.num_blocks`` admits the one request sized to the whole
    pool and then fails on the allocation behind it — the same shape as the pad
    row itself, one level down.
    """
    cfg, backend = tiny(), get_backend()
    nb = 16  # 16 blocks x BLOCK_TOKENS 16 = 256 tokens usable, 272 gross

    def engine(decode_graph):
        return build_engine(cfg, build_random(cfg, seed=7), backend, num_blocks=nb,
                            num_slots=2, max_batch=2, max_total_tokens=512,
                            decode_graph=decode_graph)

    on, off = engine(True), engine(False)
    assert on.usable_blocks == nb and on._kv.num_blocks == nb + 1
    assert off.usable_blocks == nb and off._kv.num_blocks == nb

    # 260 + spec_depth needs 17 blocks: over the 16 usable, inside the 17 gross.
    big = torch.randint(0, cfg.vocab_size, (260,),
                        generator=torch.Generator().manual_seed(2)).tolist()
    params = SamplingParams(temperature=0.0, max_new_tokens=1, seed=0)
    for eng in (on, off):  # graph off is the control: same rejection, no pad row
        with pytest.raises(ValueError, match="exceeds KV pool capacity"):
            eng.submit(big, params)


def test_the_block_fit_prices_a_block_at_what_the_pools_actually_allocate(monkeypatch):
    """``_fit_blocks`` divides free memory by its own bytes-per-block, and nothing
    checked that figure against the pools it is sizing.

    The failure is silent in both directions and neither raises here: over-ask and
    the OOM lands later, in whatever allocates next (measured on the V100 --
    ``num_blocks=2048`` left the draft's prefill readout short 1.88 GiB with 892 MiB
    free, and the traceback named ``linear_fp4``); under-ask and serve quietly loses
    context it could have had. So this inverts the real function -- blocks it returns,
    times the bytes the pools really take -- and checks the product lands on the 2/3
    of free memory the function set out to spend.

    Two errors it catches, both live before it was written: ``per_block`` already
    carries the K+V factor, so a draft layer is ``per_block / len(full_attn_layers)``
    and dividing by ``2 * len`` charged half a layer (3.03% over-ask on the 27B); and
    the draft term was added unconditionally, so every dense engine -- all of
    training -- was charged for a pool it never builds.
    """
    from types import SimpleNamespace

    import torch

    import tilerl.engine as eng_mod
    from tilerl.kv_cache import PagedKvPool

    cfg = tiny()
    io = torch.float32
    free = 8 << 30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (free, free))
    backend = SimpleNamespace(device=torch.device("cuda"), io=io)

    def pool_bytes_per_block(layers, layer_map):
        p = PagedKvPool(8, cfg.num_kv_heads, cfg.head_dim, num_layers=layers,
                        device="cpu", dtype=io, layer_map=layer_map)
        return (p.k_pool.numel() + p.v_pool.numel()) * p.k_pool.element_size() / 8

    trunk = pool_bytes_per_block(len(cfg.full_attn_layers), cfg.full_attn_layers)
    # DraftHead.attach mirrors num_blocks with layer_map=range(draft cfg.num_layers).
    draft1 = pool_bytes_per_block(1, (0,))

    for draft_layers, real in ((0, trunk), (1, trunk + draft1)):
        blocks = eng_mod._fit_blocks(cfg, backend, io, 0, draft_layers=draft_layers)
        spent = blocks * real
        budget = free * 2 / 3
        assert 0.999 <= spent / budget <= 1.0, (
            f"draft_layers={draft_layers}: {blocks} blocks x {real:.0f} real B "
            f"= {spent / 2**30:.4f} GiB against a {budget / 2**30:.4f} GiB budget "
            f"({spent / budget:.4f}x) -- the fit is pricing a block wrong")


def test_fitting_the_kv_pool_happens_after_the_state_pool(monkeypatch):
    """``num_blocks=0`` must fit against memory the GDN pools have already taken.

    Order is the whole content of this: the state pool is 2.94 GiB at slots=3 depth=3
    on the 27B, so a fit measured before it OVER-asks by that much. Sizing it from
    cli.py, one call earlier, asked for 10.21 GiB with 4.96 free and OOMed inside
    PagedKvPool -- and the fit is arch- and card-specific, so no CPU gate can catch
    that numerically. Assert the sequence instead.
    """
    import tilerl.engine as eng_mod

    seen = []
    real_state, real_fit = eng_mod.LinearStatePool, eng_mod._fit_blocks
    real_quant = eng_mod._quantize_draft

    def spy_state(*a, **k):
        seen.append("state")
        return real_state(*a, **k)

    def spy_fit(*a, **k):
        seen.append("fit")
        return real_fit(*a, **k)

    def spy_quant(*a, **k):
        seen.append("draft")
        return real_quant(*a, **k)

    monkeypatch.setattr(eng_mod, "LinearStatePool", spy_state)
    monkeypatch.setattr(eng_mod, "_fit_blocks", spy_fit)
    monkeypatch.setattr(eng_mod, "_quantize_draft", spy_quant)
    cfg = tiny()
    e = build_engine(cfg, build_random(cfg, seed=7), get_backend(), num_blocks=0,
                     num_slots=2, max_batch=2, max_total_tokens=512, max_blocks=16)
    assert seen == ["state", "fit"], f"the KV fit ran before the state pool: {seen}"
    assert e.usable_blocks == 16, f"max_blocks must cap the fit, got {e.usable_blocks}"

    # With a draft, its WEIGHTS must be served before the fit reads free memory. They
    # used to be quantized inside Engine.__init__, i.e. after the fit had spent 2/3 of
    # free memory and PrefixStore a quarter of the rest -- so on the 27B at serve's
    # default --slots 16 the draft's own fp4 weights were charged to nothing and
    # `serve --blocks 0 --draft` died in materialize's twiddle with 104 MiB free,
    # before the fit's print. `draft` must come FIRST, not just before `fit`: the
    # reclaim between them is what turns the fit into a measurement.
    seen.clear()
    trunk = build_random(cfg, seed=7)
    build_engine(cfg, trunk, get_backend(), num_blocks=0, num_slots=3, max_batch=2,
                 max_total_tokens=512, max_blocks=16,
                 draft=_draft(cfg, trunk), spec_depth=3)
    assert seen[:3] == ["draft", "state", "fit"], (
        f"the draft's weights must be served before the state pool and the fit: {seen}")
    # Engine.__init__ still calls it, for a caller who builds an Engine directly with an
    # unquantized draft. That call is a no-op on already-packed params (_quantize_draft
    # returns its input when it sees a .wq/.w8 key), which is why it may follow the fit.
    assert seen[3:] in ([], ["draft"]), f"one re-serve at most, got {seen}"


def test_a_tick_with_no_pad_row_runs_eager_instead_of_capturing_mid_request():
    """Without the pad row, an under-full tick must NOT capture an exact-size graph.

    `precapture` builds `graph_keys()`, which enumerates buckets only: at
    max_batch=4 that is {1, 2, 4}. The old exact-size fallback set ``B = n``, so a
    3-row tick with the pad row gone asked `_graph_for(3, 1)` — a key precapture
    never builds — and captured it inside a live request (~14 s on the 27B), under
    exactly the pool pressure that removed the pad row. One eager tick is cheaper.

    Target-independent: what is gated is which key the tick asks for, not the
    capture. `_graph_for` is spied so the assertion is the call itself.
    """
    cfg, backend = tiny(), get_backend()
    e = build_engine(cfg, build_random(cfg, seed=7), backend, num_blocks=16,
                     num_slots=4, max_batch=4, decode_graph=True)
    asked: list[tuple[int, int]] = []
    e._graph_for = lambda B, W, keep: asked.append((B, W))  # None => caller runs eager

    # Only len(reqs) is read before the branch under test.
    reqs = [object(), object(), object()]
    assert e._graph_bucket(3) == 4, "this test needs a row count that pads"

    # Control: the pad row is there, so the tick keys on the bucket precapture built.
    assert e._pad_slot is not None
    assert e._run_decode_graph(reqs) is False  # the spy returns None
    assert asked == [(4, 1)], f"a padded tick must key on its bucket, got {asked}"

    # The case: no pad row and no capacity to take one.
    asked.clear()
    e._pad_slot = e._pad_block = None
    e._states._free.clear()
    e._kv._free.clear()
    assert e._run_decode_graph(reqs) is False
    assert asked == [], f"an unpadded tick asked for an off-grid graph: {asked}"
    assert (3, 1) not in e.graph_keys(), "3 is a bucket here; pick another row count"


def test_graph_keys_covers_what_a_decode_tick_keys_on():
    """`graph_keys` is what `precapture` builds, so it must contain every key
    `_run_decode_graph` would look up — otherwise warming succeeds, reports N
    graphs, and a real request captures anyway (~14 s on the 27B).

    That is exactly what a generate-and-hope warmup did: chain width depends on
    the draft's confidence, so no number of generated tokens guarantees a width
    appears, and two were left uncaptured. Runs off CUDA because it checks keys,
    not captures; capture parity is the CUDA test above.
    """
    backend = get_backend()
    cfg = tiny()
    for max_batch in (1, 2, 4, 8):
        e = build_engine(cfg, build_random(cfg, seed=21), backend, num_blocks=16,
                         num_slots=max_batch + 1, max_batch=max_batch,
                         max_total_tokens=256)
        keys = e.graph_keys()
        for rows in range(1, max_batch + 1):
            assert (e._graph_bucket(rows), 1) in keys, (
                f"max_batch={max_batch}: a {rows}-row tick keys on "
                f"{(e._graph_bucket(rows), 1)}, which precapture would not build"
            )
        assert e._graph_bucket(max_batch) <= max_batch, "a bucket may not exceed max_batch"

    # With a draft, every width a trimmed chain can present. Untested until it broke:
    # the widths branch read a `_spec_depth` attribute the Engine never sets, so
    # `precapture` died with AttributeError on the FIRST drafted run — every case
    # above builds a dense engine and takes the `(1,)` path.
    trunk = build_random(cfg, seed=21)
    e = build_engine(cfg, trunk, backend, num_blocks=16, num_slots=3, max_batch=2,
                     max_total_tokens=256, draft=_draft(cfg, trunk), spec_depth=3)
    keys = e.graph_keys()
    for w in range(1, e._width + 1):
        assert (e._graph_bucket(1), w) in keys, (
            f"a width-{w} verify tick keys on {(e._graph_bucket(1), w)}, which "
            f"precapture would not build; widths present: {sorted({k[1] for k in keys})}"
        )
    assert max(k[1] for k in keys) == e._width, (
        f"graph_keys goes past the drafter's settled width {e._width}: "
        f"{sorted({k[1] for k in keys})} — those captures are ~14 s each and dead"
    )


def test_the_engines_verify_width_is_reachable_from_outside():
    """A harness must be able to move a live engine's depth, and see that it moved.

    `scripts/ab_draft_depth.py` prices one draft forward as the difference between
    two depths, so its whole output is a difference. It set `e._spec_depth`, which
    was live until 7069a1f moved the chain loop onto the head — after that the
    assignment hit nothing, and the sweep would have measured ONE config four
    times and reported drafting as free. A no-op knob does not produce a wrong
    number, it produces a plausible one, which is why this is a test and not a
    comment: `_width` on the engine and `width` on the head are the two places a
    tick reads, and both have to answer.
    """
    backend, cfg = get_backend(), tiny()
    trunk = build_random(cfg, seed=21)
    head = _draft(cfg, trunk)
    e = build_engine(cfg, trunk, backend, num_blocks=16, num_slots=3, max_batch=2,
                     max_total_tokens=256, draft=head, spec_depth=3)
    assert e._width == 4, f"spec_depth=3 must build width 4, got {e._width}"
    for depth in (2, 1, 3):
        head.set_depth(depth)
        e._width = head.width
        assert head.width == depth + 1
        assert e._width == depth + 1
        # The width has to reach what a tick keys on, or the move is cosmetic.
        assert max(k[1] for k in e.graph_keys()) == depth + 1, (
            f"depth {depth} did not reach graph_keys: "
            f"{sorted({k[1] for k in e.graph_keys()})}"
        )
    # And the attribute the broken script wrote is still not one the engine reads,
    # so a harness that regresses to it fails here rather than on the pod.
    assert not hasattr(e, "_spec_depth"), (
        "the Engine grew a _spec_depth attribute; if it is now the depth knob, say "
        "so here — ab_draft_depth.py was broken for a whole refactor by assuming it"
    )


def test_the_sweeps_launch_buckets_match_each_arch():
    """`ab_draft_depth.py` groups ticks by launch shape, and the shape is per-arch.

    Its bucket function used to be sm70's ladder returning a bare int. On sm90 that
    mislabelled every tick while still partitioning them correctly, so the fit
    produced a residual and the only wrong thing was what the label meant: sm70's
    "64" is two 32-row launches, sm90's is one 64-row WGMMA tile. A wrong label in
    a log is not visible the way a crash is, so the table is pinned here.
    """
    import runpy
    import sys as _sys

    _sys.argv = ["ab_draft_depth.py", "--check"]
    try:
        runpy.run_path("scripts/ab_draft_depth.py", run_name="__main__")
    finally:
        _sys.argv = ["pytest"]




def test_invalidate_refills_the_cached_casts_a_replay_would_read_stale():
    """The refill walk, on cpu, where no card is needed to run it.

    `_const_f32` caches a parameter's cast and refills it in place when called
    with new values (#190). A graph replay calls nothing, so a run that keeps its
    captured graphs across an optimizer step reads the cast taken BEFORE the step
    unless something drives the refill -- which is what `invalidate_weights` now
    does. Measured before this existed: the address survived `p.copy_()` and the
    values did not.

    The count is asserted non-zero because a walk over an empty cache returns 0
    and would pass every assertion below it.
    """
    from tilerl.autograd import AdamW

    backend = get_backend()
    w = torch.randn(8, 8, dtype=torch.bfloat16, device=backend.device)
    cached = backend._const_f32(w)
    baked, before = cached.data_ptr(), cached.clone()

    opt = AdamW(lr=1.0)
    opt._step = 1
    opt.step_one(w, torch.randn(8, 8, device=backend.device) * 5.0)
    # The premise the whole scheme rests on: an in-place update, so the address a
    # capture baked is still the address the new values land behind.
    assert torch.equal(cached, before), "something refilled the cast without being asked"

    assert backend.refill_const_f32() >= 1, "the walk refilled nothing; every arm below is vacuous"
    # Ask the CACHE what it holds, not the handle: a refill that reallocates leaves
    # this handle -- the buffer a capture baked -- correct-looking and orphaned, and
    # the value assertion below could not tell that from never refilling at all.
    assert backend._const_f32_cache[(w.data_ptr(), None, torch.float32)][2] is cached, (
        "the refill rebound the cache to a new buffer; a captured graph still reads the old one"
    )
    assert cached.data_ptr() == baked, "the refill moved the buffer a graph would have baked"
    assert not torch.equal(cached, before), "the cached cast still holds the pre-step values"
    assert torch.equal(cached, w.to(torch.float32)), "the refilled cast is not the new weights"


def test_a_dead_parameter_leaves_no_entry_behind():
    """A freed parameter's address can be reused, so its entry must not outlive it.

    Without this the walk would re-cast through a dangling weakref, or keep an
    entry a later parameter at the same address would collide with.
    """
    backend = get_backend()
    w = torch.randn(4, 4, dtype=torch.bfloat16, device=backend.device)
    backend._const_f32(w)
    n = len(backend._const_f32_cache)
    assert n >= 1
    del w
    backend.refill_const_f32()
    assert len(backend._const_f32_cache) < n, "the entry outlived its parameter"


def test_a_live_drafted_tick_keys_on_a_width_precapture_built():
    """The W a real tick asks for comes from `len(chains[0])`, not from `_graph_bucket`.

    Every other test here derives the key the way `graph_keys` does, so all of them agree
    with `graph_keys` by construction. This one runs the drafter, lets it leave whatever
    chain its confidences produce, and checks the key `_run_decode_graph` actually looks
    up. That is the one path from the draft's own output to a graph key, and a width off
    the grid captures inside a live request (~14 s on the 27B) after warming reported
    success.
    """
    cfg, backend = tiny(), get_backend()
    trunk = build_random(cfg, seed=21)
    e = build_engine(cfg, trunk, backend, num_blocks=32, num_slots=3, max_batch=2,
                     max_total_tokens=256, draft=_draft(cfg, trunk), spec_depth=3,
                     decode_graph=True)
    keys = e.graph_keys()
    asked: list[tuple[int, int]] = []
    real = e._graph_for
    e._graph_for = lambda B, W, keep: (asked.append((B, W)), real(B, W, keep))[1]

    e.submit(list(range(1, 24)), SamplingParams(max_new_tokens=6, seed=0))
    for _ in range(40):
        e.step()
        if e.poll():
            break
    # Off CUDA the first capture fails and `_decode_graph_on` goes False, so exactly one
    # key is ever asked for. Assert the count: a bare `assert asked` would pass on that one
    # ask and could never see a width the drafter chose on a later tick.
    assert len(asked) >= 1, "no tick reached the graph path; the test proves nothing"
    off = [k for k in asked if k not in keys]
    assert not off, (
        f"a live tick keyed on {off}, which precapture never builds — widths in "
        f"graph_keys: {sorted({k[1] for k in keys})}, asked: {sorted(set(asked))}"
    )
    # The width came from the drafter's chain, not from `_graph_bucket`: a plain (B, 1)
    # would mean the draft left nothing and the verify path never ran.
    assert max(w for _, w in asked) > 1, (
        f"every ask was width 1, so no chain reached the graph key: {asked}"
    )
