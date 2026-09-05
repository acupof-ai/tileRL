"""End-to-end gates: engine, prefix cache, training, tape, spec decode (hermetic, CPU)."""

from __future__ import annotations

import contextlib
import math
import os
import threading
import time
import unittest.mock
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
    _PREFILL_BUCKET,
    BLOCK_TOKENS,
    BatchKv,
    Engine,
    SamplingParams,
    _restrict,
    _step_seed,
    build_engine,
)
from tilerl.kv_cache import (
    DramSnapshots,
    KvTier,
    NoPrefixStore,
    PagedKvPool,
    PrefixStore,
)
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


def test_restriction_is_the_same_batched_as_per_row():
    """When every row cuts the same way, the sampler restricts [N,V] once instead of
    N times -- one topk and one allowed_ids upload, not N of each. The rows must come
    out identical; a batched topk that took the kth value across the batch instead of
    per row would widen some rows' support and narrow others, and still sample."""
    torch.manual_seed(4)
    logits = torch.randn(4, 64)
    for p in (
        SamplingParams(top_k=5),
        SamplingParams(allowed_ids=(1, 3, 7, 11, 13, 20, 31)),
        SamplingParams(top_k=3, allowed_ids=(2, 5, 9, 14, 22, 40, 55, 60)),
    ):
        batched = _restrict(logits, p)
        for i in range(4):
            assert torch.equal(batched[i], _restrict(logits[i], p)), (p, i)


def test_rows_that_cut_differently_take_the_per_row_path():
    """Rows whose (allowed_ids, top_k) differ fall back to per-row restriction.
    Nothing else in the suite builds such a batch, so the fallback shipped
    unexercised, and it fails silently: cutting every row to row 0's rule still
    samples. The two rules disagree on purpose -- one row's allowed_ids exclude
    that row's own argmax -- so row 0's rule applied to the batch moves a token."""
    eng = _build_engine(seed=3)
    v = tiny().vocab_size
    torch.manual_seed(11)
    logits = torch.randn(2, v, device=eng._backend.device)

    class _R:
        def __init__(self, params):
            self.params = params

    top = logits.argmax(-1).tolist()
    wide = SamplingParams(temperature=0.0, seed=5)
    narrow = SamplingParams(temperature=0.0, seed=5,
                            allowed_ids=tuple(i for i in range(v) if i != top[1]))
    for a, b in ((wide, narrow), (narrow, wide)):
        rows = [(_R(a), logits[0], 0), (_R(b), logits[1], 0)]
        assert eng._sample_batch(rows) == [eng._sample_batch([r])[0] for r in rows], (a, b)

    # the assertions above only bite because the rules disagree on row 1
    assert eng._sample_batch([(_R(narrow), logits[1], 0)])[0] != top[1]


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


def test_tokens_generated_equals_the_tokens_poll_returned():
    """`stats()["tokens_generated"]` is the numerator of every tok/s figure in scripts/
    (24 references there, and until this test, zero in tests/). If it drifts from what
    the engine actually returned, every throughput number moves with it and nothing
    fails.

    The counter is incremented per committed token in `_commit`, and two branches there
    return BEFORE the increment: a stop token, and a forced end-think token. Both also
    skip `req.output.append`, so the two sides stay equal -- that is the invariant, not
    the increment's position. Asserted across three requests at once so a per-request
    reset or a double count on a batched tick shows up too.
    """
    engine = _build_engine(seed=99)
    try:
        prompt = np.random.default_rng(3).integers(3, 320, size=8).astype(np.int64)
        ids = [
            engine.submit(prompt, SamplingParams(max_new_tokens=6, temperature=0.0, seed=1)),
            engine.submit(prompt, SamplingParams(max_new_tokens=6, temperature=0.0, seed=2)),
        ]
        out = _drain(engine, ids, max_new_tokens=6)
        returned = sum(len(out[i]) for i in ids)
        counted = engine.stats()["tokens_generated"]
        assert counted == returned, (
            f"tokens_generated {counted} != {returned} tokens poll() returned. Every tok/s "
            f"in scripts/ divides by this counter, so a drift here silently rescales them."
        )

        # A stop token ends a reply through the branch that returns early. It is neither
        # appended nor counted, so the two must still agree.
        stop = int(out[ids[0]][2])
        rid = engine.submit(
            prompt,
            SamplingParams(max_new_tokens=6, temperature=0.0, seed=1, stop_token_ids=(stop,)),
        )
        for _ in range(64):
            got = engine.poll()
            if rid in got:
                break
            engine.step()
        else:
            raise TimeoutError("the stop-token request never finished")
        assert len(got[rid]) < 6, "the stop token did not end the reply early"
        assert engine.stats()["tokens_generated"] == returned + len(got[rid]), (
            "the stop-token path counted a token it did not return, or vice versa"
        )
    finally:
        engine.shutdown()


def test_blocks_used_is_what_the_engine_owns_not_what_the_pool_holds():
    """`blocks_used` gates admission (`usable_blocks` is compared against it), and it is
    maintained by hand: three `+= ` sites and one `-=`. Nothing asserted its value
    mid-run, only that it returns to 0 after a failure rollback (:240).

    The invariant is `blocks_used == sum(r.own_blocks)`, NOT `== pool.used_blocks`. The
    two diverge on purpose: `_finish` frees every block in `req.blocks` but subtracts
    only `own_blocks`, because a prefix hit adopts blocks the engine did not allocate.
    So on a hit the pool holds more than the engine owns, and that gap is the prefix
    store's retention -- which is why both counters exist.

    Checked while a request is live, since a counter that is only inspected at rest
    cannot show a leak that cancels on teardown.
    """
    engine = _build_engine(seed=99)
    try:
        rng = np.random.default_rng(1)
        head = rng.integers(3, 320, size=16).astype(np.int64)  # exactly one block
        params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=5)

        engine.submit(head, params)
        engine.step()  # prefill, so blocks are allocated and the request is still live
        live = list(engine._running) + list(engine._waiting)
        assert live, "no request is live, so this measures the idle state instead"
        owned = sum(r.own_blocks for r in live)
        assert engine.stats()["blocks_used"] == owned, (
            f"blocks_used {engine.stats()['blocks_used']} != {owned} owned by live requests; "
            f"admission compares usable_blocks against this number"
        )

        # Step through decode and re-check every tick. A request growing past a block
        # boundary allocates mid-run (engine.py:725 and :788), and both sites charge the
        # counter. Checking only after prefill and after finish misses that: the charge
        # and the release cancel, so a growth site that never charges still lands on 0.
        # Measured: without this loop, deleting the `_blocks_used += 1` at :725 was MISSED.
        grew = False
        for _ in range(64):
            engine.step()
            live = list(engine._running) + list(engine._waiting)
            if not live:
                break
            owned = sum(r.own_blocks for r in live)
            assert engine.stats()["blocks_used"] == owned, (
                f"mid-decode: blocks_used {engine.stats()['blocks_used']} != {owned} owned. "
                f"A growth site allocated a block without charging it, or charged twice."
            )
            grew = grew or owned > 1
        assert grew, (
            "no request ever owned more than one block, so the mid-run growth path at "
            "engine.py:725/:788 was never exercised and this loop asserts nothing about it"
        )
        engine.poll()

        # Second prompt extends the first by a whole block, so it adopts one.
        engine.submit(np.concatenate([head, rng.integers(3, 320, size=16).astype(np.int64)]),
                      params)
        engine.step()
        st = engine.stats()
        live = list(engine._running) + list(engine._waiting)
        # Asserted unconditionally, not under `if st["prefix_hits"]`: a conditional guard
        # here would skip the only two checks that matter whenever the adoption stopped
        # happening, which is exactly the regression to catch. Measured: hits 1,
        # blocks_used 1, pool_used_blocks 2.
        assert st["prefix_hits"] == 1, (
            f"the second prompt did not adopt the first's block (hits={st['prefix_hits']}), "
            f"so the two counters cannot be compared below"
        )
        assert st["blocks_used"] == sum(r.own_blocks for r in live), (
            "after a prefix hit the engine must count only the blocks it allocated"
        )
        assert st["blocks_used"] < st["pool_used_blocks"], (
            f"a prefix hit adopted blocks, so the pool ({st['pool_used_blocks']}) must "
            f"hold more than the engine owns ({st['blocks_used']}); equal means either "
            f"the adopted blocks were charged to the engine or nothing was adopted"
        )
    finally:
        engine.shutdown()
    # NOT asserted: blocks_used == 0 here. `shutdown` only stops the daemon thread; it
    # does not finish or free live requests, so a request still running legitimately keeps
    # its blocks charged. Asserting 0 fails at 1, and the defect would be in the test.
    assert engine.stats()["blocks_used"] == sum(
        r.own_blocks for r in list(engine._running) + list(engine._waiting)
    ), "a request outlived the accounting: charged blocks with no live owner"


def test_the_reread_prefix_survives_capacity_pressure():
    """The shape both real workloads have: one conversation's prefix re-read every
    turn while other traffic pushes the store past capacity.

    FIFO evicts by insertion order, so the conversation's own entry -- the oldest
    and the only one anybody asks for again -- goes first and every turn misses.
    LRU keeps it because each turn touches it. Asserted as a hit COUNT rather than
    a policy read, so it fails if `lookup` stops recording recency or `_evict_one`
    stops taking the LRU end.
    """
    pool = PagedKvPool(64, num_kv_heads=2, head_dim=8, num_layers=1)
    store = PrefixStore(pool, capacity=4)

    def blocks(n):
        return [pool.alloc_block() for _ in range(n)]

    convo = list(range(1, BLOCK_TOKENS + 1))
    store.insert(convo, blocks(1))
    for turn in range(6):
        # Each turn re-reads the conversation, then unrelated traffic arrives.
        assert store.lookup(convo) is not None, f"turn {turn}: the re-read prefix was evicted"
        store.insert(list(range(1000 + turn * 100, 1000 + turn * 100 + BLOCK_TOKENS)), blocks(1))

    # 4-entry capacity, 7 inserts, so eviction ran; the re-read entry outlived it.
    assert store.stats()["evictions"] >= 3
    assert store.hits == 6 and store.misses == 0


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


def test_a_ragged_prompt_publishes_and_its_state_matches_no_store():
    """A prompt whose length is not block-aligned must still become shareable, and the
    state a hit restores must equal the state a store-less engine computes.

    Two failures at once, both measured on the live V100 before the fix:
    `prefix_published` 4 with `prefix_hits` 0 over a 6-turn chat, every publish from
    decode and none from prefill. `_finish_prefills` only published when the WHOLE
    prompt length was block-aligned, and 15 of every 16 lengths are not -- so a chat
    client resending a growing conversation shared nothing. `_pick` now cuts the first
    chunk at a _PREFILL_BUCKET boundary and the chunk end publishes.

    The state assert is the half that cannot be dropped. A truncated entry pairs KV
    for N tokens with a state that absorbed more, and argmax over a vocabulary hides
    it: an earlier attempt at this was byte-identical in output while the snapshot's
    norm was 74.0 against a correct 39.75. Comparison happens with both arms at the
    SAME prefill_from -- reading after one tick puts the hit arm at 164 and the
    control at 128, which reports 66.5 vs 43.75 and is a measurement error, not a bug.

    `hits >= 1` is asserted first: without a hit nothing was restored and the state
    comparison is two identical fresh computations. On origin/main this test fails
    there (hits=0), which is what makes the rest of it mean anything.
    """
    cfg = tiny()
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=5)
    rng = np.random.default_rng(7)
    conv = rng.integers(3, 320, size=300).astype(np.int64)
    short, long = 100, 164  # neither is block-aligned; both cross a 64 boundary

    def run(no_store: bool):
        kw = {"prefix_store": NoPrefixStore()} if no_store else {}
        engine = build_engine(
            cfg, build_random(cfg, seed=99), get_backend(), num_blocks=64, num_slots=4,
            max_batch=4, max_total_tokens=2048, max_num_batched_tokens=512, **kw,
        )
        if not no_store:  # warm the store with the shorter prompt
            engine.submit(conv[:short], params)
            for _ in range(200):
                engine.step()
                if not (list(engine._running) + list(engine._waiting)):
                    break
            engine.poll()
        rid = engine.submit(conv[:long], params)
        for _ in range(50):
            engine.step()
            live = [r for r in list(engine._running) + list(engine._waiting) if r.req_id == rid]
            assert live, "the row finished before its prompt was prefilled"
            row = live[0]
            if row.prefill_from >= long:
                break
        return engine, engine._states.states[row.state_slot].clone()

    hit_engine, hit_state = run(False)
    _, ref_state = run(True)
    st = hit_engine.stats()
    assert st["prefix_published"] >= 1, (
        f"a {short}-token prompt published nothing ({short} % {BLOCK_TOKENS} = "
        f"{short % BLOCK_TOKENS}); the chunk was not cut at a bucket boundary"
    )
    assert st["prefix_hits"] >= 1, (
        f"no prefix hit (hits={st['prefix_hits']}), so the state comparison below is two "
        "identical fresh computations and this test is inert"
    )
    assert torch.allclose(hit_state, ref_state, rtol=1e-2, atol=1e-5), (
        f"the restored GDN state differs from a store-less engine's: max|delta| "
        f"{(hit_state - ref_state).abs().max():.3e}, norms {hit_state.norm():.4f} vs "
        f"{ref_state.norm():.4f}. The output can be byte-identical while this is wrong."
    )


def test_a_chunk_end_off_the_bucket_still_restores_an_exact_state():
    """A publish needs block alignment, NOT `_PREFILL_BUCKET` alignment — and the state it
    restores is exact either way.

    `_pick` aligns the first chunk to `_PREFILL_BUCKET` (64) while the publish gate tests
    `% BLOCK_TOKENS` (16), which a reviewer read as an inconsistency. It is not: a publish
    needs whole blocks to slice (`% 16`) and a state that is exact, which holds at ANY
    chunk end. 64 only guarantees that a single-chunk prompt HAS an aligned boundary.

    Driven with `max_num_batched_tokens=48` so mid-prompt chunk ends land at 48/96/144 —
    multiples of 16, none of them multiples of 64. The warm-up length is 150, which is
    NOT block-aligned, so the full-prompt publish branch cannot fire and every entry in
    the store comes from the mid-chunk `elif`. That isolation is what makes the negative
    control bite: with the gate switched to `_PREFILL_BUCKET` this configuration publishes
    **0** entries against 3, so the hit assert fails. A warm-up at 144 does NOT isolate it
    — the full-prompt branch publishes there and the test passes either way, which is how
    the first version of this test came out inert.
    """
    cfg = tiny()
    params = SamplingParams(temperature=0.0, max_new_tokens=6, seed=5)
    rng = np.random.default_rng(7)
    conv = rng.integers(3, 320, size=600).astype(np.int64)
    budget, warm_to, target = 48, 150, 272
    assert budget % BLOCK_TOKENS == 0 and budget % _PREFILL_BUCKET != 0, (
        "the chunk ends must be block-aligned but off the bucket, or this proves nothing"
    )
    assert warm_to % BLOCK_TOKENS != 0, (
        "the warm-up length must be ragged, or the full-prompt branch publishes and the "
        "mid-chunk branch this test targets is never exercised"
    )

    def run(no_store: bool):
        kw = {"prefix_store": NoPrefixStore()} if no_store else {}
        engine = build_engine(
            cfg, build_random(cfg, seed=99), get_backend(), num_blocks=64, num_slots=4,
            max_batch=4, max_total_tokens=2048, max_num_batched_tokens=budget, **kw,
        )
        if not no_store:
            engine.submit(conv[:warm_to], params)
            for _ in range(300):
                engine.step()
                if not (list(engine._running) + list(engine._waiting)):
                    break
            engine.poll()
        rid = engine.submit(conv[:target], params)
        for _ in range(300):
            engine.step()
            live = [r for r in list(engine._running) + list(engine._waiting) if r.req_id == rid]
            assert live, "the row finished before its prompt was prefilled"
            row = live[0]
            if row.prefill_from >= target:
                break
        return engine, engine._states.states[row.state_slot].clone()

    hit_engine, hit_state = run(False)
    _, ref_state = run(True)
    st = hit_engine.stats()
    assert st["prefix_hits"] >= 1, (
        f"no hit (hits={st['prefix_hits']}), so the state comparison is two identical fresh "
        "computations and this test is inert"
    )
    assert torch.allclose(hit_state, ref_state, rtol=1e-2, atol=1e-5), (
        f"a chunk end off the {_PREFILL_BUCKET} bucket restored an inexact state: max|delta| "
        f"{(hit_state - ref_state).abs().max():.3e} — then the publish gate must test the "
        f"bucket, not {BLOCK_TOKENS}"
    )


def test_an_intermediate_chunk_publish_stays_out_of_the_disk_tier(tmp_path):
    """Only the LAST publish of a prompt is offered to disk, and that is the whole 8.96%.

    Every chunk boundary publishes, and each publish offered to the tier costs a D2H inside
    the prefill that made it. The intermediate ones are pure waste: the prompt-complete
    publish covers the same tokens and more, and a GDN snapshot is a CONSTANT ~157 MB at
    every prefix length, so on H20 card 6 a 2729-token prompt's six publishes spilled
    1624 MB to serve one 325 MB entry — 0.925 s of a 2.041 s request against 0.180 s for the
    last publish alone.

    `spill=False` on the mid-chunk branch is that fix and nothing asserted it: dropping the
    kwarg leaves all 328 tests passing (measured), and the regression is invisible except as
    45.3% instead of 8.96% on a machine this suite does not run on.

    The count is what carries it, so both operands are asserted: `prefix_published` proves the
    intermediate publishes HAPPENED (otherwise `offered == 1` passes because there was only
    ever one publish), and `ssd_offered == 1` proves only one reached the tier. The warm-up is
    block-ALIGNED here, unlike the state test above: a ragged length never fires the
    prompt-complete branch, so the expected offer count would be 0 and the assertion could
    not tell "only the last publish spilled" from "nothing spilled at all". Measured at this
    configuration: 5 publishes, 1 offer.
    """
    cfg = tiny()
    params = SamplingParams(temperature=0.0, max_new_tokens=4, seed=5)
    rng = np.random.default_rng(7)
    conv = rng.integers(3, 320, size=600).astype(np.int64)
    budget, warm_to = 48, 240
    assert warm_to % BLOCK_TOKENS == 0 and warm_to > budget, (
        "the warm-up must be block-aligned so the prompt-complete publish fires (that is the "
        "one offer being counted) and longer than one chunk so intermediate publishes exist"
    )
    engine = build_engine(
        cfg, build_random(cfg, seed=99), get_backend(), num_blocks=64, num_slots=4,
        max_batch=4, max_total_tokens=2048, max_num_batched_tokens=budget,
        ssd_path=str(tmp_path), ssd_min_tokens=BLOCK_TOKENS,
    )
    engine.submit(conv[:warm_to], params)
    for _ in range(300):
        engine.step()
        if not (list(engine._running) + list(engine._waiting)):
            break
    engine.poll()

    st = engine.stats()
    assert st["prefix_published"] >= 3, (
        f"only {st['prefix_published']} publishes at a {budget}-token budget over {warm_to} "
        "tokens; with no intermediate publish the offer count below is trivially 1"
    )
    assert st["ssd_offered"] == 1, (
        f"{st['ssd_offered']} publishes reached the disk tier against "
        f"{st['prefix_published']} in memory; every intermediate one pays a device-to-host "
        "copy inside the prefill that made it, and the prompt-complete publish already "
        "covers those tokens"
    )


def test_the_dram_tier_pays_only_above_its_session_count():
    """The tier's condition is `concurrent sessions > HBM snapshot budget`, both operands.

    A demotion takes the LRU snapshot and a lookup wants the MRU one, so within ONE
    conversation they never meet — measured on the V100, 43 demotions and 0 promotions,
    and 1.51x worse wall clock for it. Across conversations the LRU end IS another
    session's newest entry, so the tier starts paying as soon as sessions outnumber the
    budget. Rotating N conversations against a 9-snapshot budget: 2 -> 0 promotions,
    4 -> 0, 9 -> 17, 12 -> 24, 20 -> 40.

    Both halves are asserted here because either alone reads as a law: I published
    "structurally dead" off a sweep that held sessions at 2, which is the same relation
    with the wrong operand pinned. The below-threshold arm is what stops that from
    happening again.
    """
    torch.manual_seed(0)

    def snap(i: int):
        return (torch.randn(3, 4, 8, 8) + i, torch.randn(3, 2, 16))

    one = sum(t.nbytes for t in snap(0))
    budget = 9

    def rotate(nconv: int, dram):
        pool = PagedKvPool(8192, 2, 8, device=torch.device("cpu"), layer_map=(0,))
        store = PrefixStore(pool, state_bytes=budget * one, dram=dram)
        convs = [list(range(c * 2000, c * 2000 + 800)) for c in range(nconv)]
        hits = 0
        for turn in range(3):
            for toks in convs:
                length = (turn + 1) * BLOCK_TOKENS * 3
                if store.lookup(toks[:length]):
                    hits += 1
                for end in range(BLOCK_TOKENS * 3, length + 1, BLOCK_TOKENS * 3):
                    store.insert(
                        toks[:end],
                        [pool.alloc_block() for _ in range(end // BLOCK_TOKENS)],
                        snap(end),
                    )
        return hits, store.stats()

    # Below the threshold the tier must not even engage: two sessions fit in nine.
    _, few = rotate(2, DramSnapshots(budget_bytes=400 * one))
    assert few["dram_promotions"] == 0, (
        f"2 sessions against a {budget}-snapshot budget promoted "
        f"{few['dram_promotions']}; nothing should have been demoted to promote"
    )

    # Above it, the tier is the difference between no reuse and complete reuse.
    plain_hits, plain = rotate(12, None)
    tier_hits, tier = rotate(12, DramSnapshots(budget_bytes=400 * one))
    assert plain["evictions"] > 0, (
        "the no-tier arm evicted nothing, so 12 sessions did not exceed the budget and "
        "this comparison has no floor"
    )
    assert tier["dram_promotions"] > 0, (
        f"12 sessions promoted nothing (demotions={tier['dram_demotions']}); the tier "
        "never served a snapshot back and the hit count below cannot be its doing"
    )
    assert tier_hits > plain_hits, (
        f"the tier bought no hits at 12 sessions: {tier_hits} against {plain_hits}"
    )


def test_the_dram_tier_demotes_instead_of_evicting():
    """Under `state_bytes` pressure the snapshot goes to the host and the entry stays.

    The snapshot is what binds, not the KV: 144 MiB at 27B against 2.125 MiB per block,
    and `build_engine` sets `state_bytes` to a quarter of free memory. Measured on the
    live V100: 54 published, **43 evicted with 64% of the block pool still free** — every
    eviction was state bytes, and every one of them threw away a reusable prefix for a
    byte the host could have held.

    Same pressure, both arms, six prefixes inserted against room for two snapshots:

    | arm | entries | evictions | hits |
    |---|---:|---:|---:|
    | no tier | 2 | 4 | 2/6 |
    | DRAM | 6 | 0 | 6/6 |

    The round trip is asserted with `torch.equal`, not `allclose`: a pinned host copy and
    a copy back is a byte-for-byte move, so anything less than exact means the tier
    reshaped or re-dtyped the snapshot on the way through.
    """
    torch.manual_seed(0)

    def snap(i: int):
        return (torch.randn(3, 4, 8, 8) + i, torch.randn(3, 2, 16))

    one = sum(t.nbytes for t in snap(0))

    def run(dram):
        pool = PagedKvPool(256, 2, 8, device=torch.device("cpu"), layer_map=(0,))
        store = PrefixStore(pool, state_bytes=2 * one, dram=dram)
        toks = list(range(400))
        kept = {}
        for k in range(1, 7):
            length = k * BLOCK_TOKENS * 2
            kept[length] = snap(k)
            store.insert(
                toks[:length],
                [pool.alloc_block() for _ in range(length // BLOCK_TOKENS)],
                kept[length],
            )
        return store, toks, kept

    plain, toks, _ = run(None)
    assert plain.stats()["evictions"] >= 1, (
        "the no-tier arm evicted nothing, so state_bytes pressure was never reached and "
        "the comparison below has no floor"
    )
    plain_hits = sum(1 for k in range(1, 7) if plain.lookup(toks[: k * BLOCK_TOKENS * 2]))

    tiered, toks, kept = run(DramSnapshots(budget_bytes=50 * one))
    st = tiered.stats()
    assert st["demoted"] >= 1, (
        f"nothing was demoted (dram_demotions={st['dram_demotions']}), so this passes for "
        "the same reason a store with no pressure would"
    )
    assert st["evictions"] == 0, (
        f"the tier still evicted {st['evictions']} entries; demotion is supposed to "
        f"relieve state-byte pressure without giving up a prefix"
    )
    tiered_hits = 0
    for length, want in kept.items():
        hit = tiered.lookup(toks[:length])
        assert hit is not None and hit.length == length, f"lost the prefix at {length}"
        tiered_hits += 1
        for got, expect in zip(hit.state, want):
            assert torch.equal(got, expect), (
                f"the snapshot at {length} came back changed: max|delta| "
                f"{(got - expect).abs().max():.3e}"
            )
    assert tiered_hits > plain_hits, (
        f"the tier bought nothing: {tiered_hits}/6 hits against {plain_hits}/6 without it"
    )


def test_the_host_tier_drops_the_oldest_snapshot_not_an_arbitrary_one():
    """The DRAM tier evicts by bytes; this asserts it evicts the OLDEST.

    `test_a_promotion_that_comes_back_empty_is_a_miss` already drives `drops >= 1` and
    checks a lookup degrades safely, but it never says WHICH snapshot went — so a tier that
    dropped the newest, or a random one, passes it. With three snapshots against a
    two-snapshot budget the first demoted must be gone and the last two held.

    Negative control: `popitem(last=True)` instead of `last=False` inverts the victim and
    fails on the first assert.
    """
    torch.manual_seed(0)
    snaps = [(torch.randn(3, 4, 8, 8) + i, torch.randn(3, 2, 16)) for i in range(3)]
    one = sum(t.nbytes for t in snaps[0])
    dram = DramSnapshots(budget_bytes=2 * one + one // 2)
    for i, snap in enumerate(snaps):
        assert dram.demote(i, snap), f"snapshot {i} was refused by a budget that fits two"
    assert dram.drops == 1, f"expected exactly one drop at this budget, got {dram.drops}"
    dev = torch.device("cpu")
    assert dram.promote(0, dev) is None, (
        "the OLDEST snapshot survived; an LRU tier must evict it first, and evicting the "
        "newest throws away what a lookup is about to ask for"
    )
    for i in (1, 2):
        back = dram.promote(i, dev)
        assert back is not None, f"snapshot {i} was dropped instead of the oldest"
        assert torch.equal(back[0], snaps[i][0]), f"snapshot {i} came back changed"


def test_a_promotion_that_comes_back_empty_is_a_miss():
    """A snapshot the tier dropped must not serve its blocks anyway.

    Adopting KV for N tokens with no recurrent state runs the GDN layers from zero over
    KV that is not zero. It is wrong and it is silent — the earlier truncated-entry bug
    was byte-identical in output with the snapshot's norm off by 1.86x — so the entry is
    dropped and the lookup falls through to shorter prefixes or misses.
    """
    torch.manual_seed(0)
    state = (torch.randn(3, 4, 8, 8), torch.randn(3, 2, 16))
    one = sum(t.nbytes for t in state)
    pool = PagedKvPool(256, 2, 8, device=torch.device("cpu"), layer_map=(0,))
    # DRAM holds one snapshot, HBM none: inserting two demotes the first, then the tier's
    # own byte LRU drops it when the second arrives.
    dram = DramSnapshots(budget_bytes=one)
    store = PrefixStore(pool, state_bytes=0, dram=dram)
    toks = list(range(200))
    for length in (BLOCK_TOKENS * 2, BLOCK_TOKENS * 4):
        store.insert(
            toks[:length],
            [pool.alloc_block() for _ in range(length // BLOCK_TOKENS)],
            (state[0].clone(), state[1].clone()),
        )
    assert dram.drops >= 1, (
        f"the tier dropped nothing (drops={dram.drops}), so no lookup below hits the "
        "empty-promotion path and this test is inert"
    )
    hit = store.lookup(toks[: BLOCK_TOKENS * 2])
    assert hit is None or hit.state is not None, (
        "a hit came back with blocks and no snapshot; the caller would prefill the GDN "
        "layers from a zero state over non-zero KV"
    )


def test_the_ssd_flag_reaches_the_store_and_the_fingerprint_covers_the_config(tmp_path):
    """`--ssd-path` must arrive at the PrefixStore, and the fingerprint must move with
    every config field.

    Two separate silent failures, so two asserts. A flag that parses and is never
    forwarded reads exactly like a working one — `dram_bytes` had no CLI entry at all and
    the review had to find that by reading `_build_engine`. And a fingerprint that misses
    a field serves KV computed under other weights after a restart, which is wrong and
    silent; the first draft of `_weight_fingerprint` hand-listed the fields and named
    `cfg.num_heads`, which does not exist on this config at all.

    The fingerprint is checked over EVERY field via `dataclasses.replace`, not a sample:
    a hand-picked list is the defect being guarded against.
    """
    import dataclasses

    from tilerl.cli import _build_engine as cli_build
    from tilerl.engine import _weight_fingerprint

    cfg = tiny()
    base = _weight_fingerprint(cfg)
    for f in dataclasses.fields(cfg):
        old = getattr(cfg, f.name)
        new = (old + 1) if isinstance(old, int) and not isinstance(old, bool) else None
        if new is None:
            continue
        with contextlib.suppress(ValueError):  # some fields validate against each other
            assert _weight_fingerprint(dataclasses.replace(cfg, **{f.name: new})) != base, (
                f"changing {f.name} left the fingerprint unchanged, so a restart would "
                "serve KV computed under a different config"
            )

    cfg, model = _build_model("tiny", seed=11)
    engine = cli_build(cfg, model, get_backend(), slots=2, blocks=64, max_ctx=256,
                       ssd_path=str(tmp_path))
    tier = engine._prefix._ssd
    assert tier is not None, (
        "--ssd-path parsed but never reached the store; the flag would read as working"
    )
    assert os.path.isdir(os.path.join(str(tmp_path), "tilerl_kvtier"))
    assert tier._fingerprint == _weight_fingerprint(cfg)

    # `--ssd-min-tokens` goes through the same `_build_engine` and had no assertion. It is
    # what every bench uses to drive the tier at a prompt shorter than the 64-token default,
    # so dropping it silently reports 0 offers -- a tier that looks dead instead of a flag
    # that was ignored. Asserted by behaviour, not just the attribute: a 32-token prefix is
    # under the default floor and over this one, so it spills only if the flag arrived.
    floor = 2 * BLOCK_TOKENS
    assert floor < 4 * BLOCK_TOKENS, "the test floor must be below KvTier's default"
    lowered = cli_build(cfg, model, get_backend(), slots=2, blocks=64, max_ctx=256,
                        ssd_path=str(tmp_path / "low"), ssd_min_tokens=floor)
    low_tier = lowered._prefix._ssd
    assert low_tier.min_tokens == floor, (
        f"--ssd-min-tokens={floor} did not reach the tier (min_tokens={low_tier.min_tokens}); "
        "every bench that lowers the floor would quietly measure the default"
    )
    toks = list(range(floor))
    pool = lowered._kv
    assert lowered._prefix.insert(toks, [pool.alloc_block() for _ in range(2)],
                                 (torch.zeros(3, 4, 8, 8), torch.zeros(3, 2, 16)))
    assert low_tier.offered == 1, (
        f"a {floor}-token prefix was not offered to a tier with min_tokens={floor}, so the "
        "flag arrived but does not take effect"
    )


def test_a_restart_faults_the_prefix_back_in_off_disk(tmp_path):
    """The whole point of the SSD tier: HBM is empty after a restart, the disk is not.

    A second store over the same directory and fingerprint is what a restart looks like
    to this layer. Its pool holds no blocks and its index no entries, so without a read
    path every lookup misses and the tier is write-only — which is what it was: `spill_kv`
    had a caller and `load_kv`/`load_state`/`has` had none.

    Three things are asserted, because two of them are the ones that fail silently:

    * the fault-in **hits** (`ssd_hits == 1`) rather than the lookup walking past it;
    * the KV comes back byte-for-byte — a `torch.save`/`load` round trip is exact, so
      anything less means the spill reshaped or re-dtyped it;
    * the snapshot comes back too, and is not None. Adopting blocks with a zero state
      runs the GDN layers from zero over non-zero KV: wrong, and byte-identical in
      output for a while.

    The negative control is the fingerprint: a mismatched one must NOT hit, or the tier
    would serve KV computed under different weights — and it must also unlink, since that
    unlink is the only path that ever reclaims this tier's disk.
    """
    torch.manual_seed(0)
    toks = list(range(4 * BLOCK_TOKENS))
    state = (torch.randn(3, 4, 8, 8), torch.randn(3, 2, 16))

    def store_at(fingerprint: str):
        pool = PagedKvPool(64, 2, 8, device=torch.device("cpu"), layer_map=(0,))
        tier = KvTier(str(tmp_path), fingerprint, min_tokens=BLOCK_TOKENS)
        return PrefixStore(pool, ssd=tier), pool, tier

    warm, pool, tier = store_at("fp-a")
    blocks = [pool.alloc_block() for _ in range(4)]
    for i, b in enumerate(blocks):
        pool.k_pool[:, b] = float(i + 1)
        pool.v_pool[:, b] = float(-i - 1)
    assert warm.insert(toks, blocks, (state[0].clone(), state[1].clone()))
    _flushed(tier)
    assert tier.stats()["ssd_entries"] == 1, (
        f"nothing reached disk (offered={tier.offered} refusals={tier.refusals}), so the "
        "restart below cannot hit and this test is inert"
    )

    cold, cold_pool, cold_tier = store_at("fp-a")
    assert cold_tier.recovered == 1, f"recovery adopted {cold_tier.recovered} entries, not 1"
    hit = cold.lookup(toks)
    st = cold.stats()
    assert hit is not None and st["ssd_hits"] == 1, (
        f"a cold lookup missed with the prefix on disk: ssd_hits={st['ssd_hits']} "
        f"ssd_faults={st['ssd_faults']} entries={st['ssd_entries']}"
    )
    assert hit.length == len(toks)
    for i, b in enumerate(hit.blocks):
        assert torch.equal(cold_pool.k_pool[:, b], torch.full_like(cold_pool.k_pool[:, b],
                                                                   float(i + 1))), (
            f"block {i} came back changed; a save/load round trip is byte-exact"
        )
    assert hit.state is not None and torch.equal(hit.state[0], state[0]), (
        "the snapshot did not survive the disk round trip, so the blocks would be adopted "
        "with a zero recurrent state"
    )
    # A re-publish of what was just faulted in must not write the bytes back.
    assert cold_tier.offered == 0, f"the fault-in re-spilled its own read ({cold_tier.offered})"

    other, _, other_tier = store_at("fp-b")
    assert other_tier.recovered == 0 and other.lookup(toks) is None, (
        "a different fingerprint served KV computed under other weights"
    )
    # And the mismatch must UNLINK, not merely decline to index. `recovered == 0` above
    # cannot see the difference: with the files left in place it is still 0, because a
    # mismatch clears `prev` either way -- the whole suite passes with the unlink deleted
    # (measured). The bytes matter because this unlink is the only thing that ever reclaims
    # the tier's disk: `invalidate()` deliberately writes one marker instead of walking a
    # 20 GiB directory inside a training step, so every generation's spill accumulates
    # until some later `_recover` mismatches and removes it.
    left = [f for f in os.listdir(os.path.join(str(tmp_path), "tilerl_kvtier"))
            if f.endswith((".kv", ".st"))]
    assert not left, (
        f"a fingerprint mismatch left {len(left)} spill files on disk; nothing else reclaims "
        "them, so every optimizer step's spill accumulates until the disk fills"
    )

    # An optimizer step calls clear(), and a tier that keeps serving afterwards hands the
    # trainer KV computed under the PREVIOUS weights -- off-policy, and silent. clear()
    # bumps the fingerprint rather than unlinking 20 GiB inside a training step, so what
    # has to be asserted is that a store built after it does not recover.
    # A fresh directory: the fp-b store above already unlinked the files, since a
    # fingerprint mismatch deletes what it cannot serve. Reusing it would make the setup
    # assert below pass for the wrong reason -- it caught exactly that.
    d2 = str(tmp_path / "clear")
    pool2 = PagedKvPool(64, 2, 8, device=torch.device("cpu"), layer_map=(0,))
    tier2 = KvTier(d2, "fp-c", min_tokens=BLOCK_TOKENS)
    warm2 = PrefixStore(pool2, ssd=tier2)
    b2 = [pool2.alloc_block() for _ in range(4)]
    assert warm2.insert(toks, b2, (state[0].clone(), state[1].clone()))
    _flushed(tier2)
    assert KvTier(d2, "fp-c", min_tokens=BLOCK_TOKENS).recovered == 1, (
        "setup: the entry must be adoptable before clear(), or the assert below is inert"
    )
    warm2.clear()
    after_tier = KvTier(d2, "fp-c", min_tokens=BLOCK_TOKENS)
    after = PrefixStore(PagedKvPool(64, 2, 8, device=torch.device("cpu"), layer_map=(0,)),
                        ssd=after_tier)
    assert after_tier.recovered == 0 and after.lookup(toks) is None, (
        f"clear() left {after_tier.recovered} entries adoptable, so an optimizer step "
        "would serve KV computed under the weights it just replaced"
    )


@pytest.mark.parametrize("suffix", [".st", ".kv"])
def test_a_spill_truncated_by_a_crash_is_a_miss_not_a_raise(tmp_path, suffix):
    """A kill between torch.save starting and finishing leaves a partial blob on disk.

    `_recover` adopts entries by FILE SIZE, so it cannot tell a truncated blob from a whole
    one -- it will happily index a half-written pair. The read then has to survive it: a
    `torch.load` raising inside `lookup` takes down the request that happened to match,
    which is a crash for a cache miss.

    ONE ARM PER GUARD, because `_fault_in` reads the state first and returns on its miss:
    truncating both files only ever reaches `load_state`, so `load_kv`'s guard can be deleted
    and a both-files test still passes (25 measured exactly that). The `.kv` arm leaves the
    state whole so the KV guard is the one that fires, and `ssd_faults` is what proves it --
    that counter is incremented only on `load_kv` returning False.

    The durability window this covers is real and bounded: the flush daemon writes off-tick,
    so an entry published within the last flush is on disk partially or not at all. Losing
    it is correct (it re-prefills); raising on it is not.
    """
    torch.manual_seed(0)
    toks = list(range(4 * BLOCK_TOKENS))
    state = (torch.randn(3, 4, 8, 8), torch.randn(3, 2, 16))
    pool = PagedKvPool(64, 2, 8, device=torch.device("cpu"), layer_map=(0,))
    tier = KvTier(str(tmp_path), "fp-trunc", min_tokens=BLOCK_TOKENS)
    store = PrefixStore(pool, ssd=tier)
    assert store.insert(toks, [pool.alloc_block() for _ in range(4)],
                        (state[0].clone(), state[1].clone()))
    _flushed(tier)

    d = os.path.join(str(tmp_path), "tilerl_kvtier")
    victims = [f for f in os.listdir(d) if f.endswith(suffix)]
    assert victims, f"nothing spilled a {suffix}, so the truncation below is inert"
    for name in victims:
        path = os.path.join(d, name)
        with open(path, "r+b") as fh:
            fh.truncate(os.path.getsize(path) // 3)

    cold_tier = KvTier(str(tmp_path), "fp-trunc", min_tokens=BLOCK_TOKENS)
    assert cold_tier.recovered >= 1, (
        "recovery skipped the truncated files, so the load path below is never reached -- "
        "it adopts by size and a truncated file still has one"
    )
    cold = PrefixStore(PagedKvPool(64, 2, 8, device=torch.device("cpu"), layer_map=(0,)),
                       ssd=cold_tier)
    hit = cold.lookup(toks)  # must not raise
    assert hit is None, f"a truncated spill served a hit of length {hit and hit.length}"
    if suffix == ".kv":
        assert cold.ssd_faults == 1, (
            "the state loaded but ssd_faults is 0, so load_kv was never reached and this arm "
            "does not exercise its guard"
        )


def test_a_spill_still_in_the_queue_is_served_from_memory(tmp_path):
    """`resident()` is two conditions and only the `_lru` one was tested.

    A spill sits in `_pending` from the moment `spill_kv` enqueues it until the daemon's
    `torch.save` returns -- ~100 ms, and the whole point of moving the save off-tick. During
    that window there is NO FILE. Both halves of `resident()` and both loads have a
    pending-table branch for it; deleting the `_pending` half of the `or` leaves all 70 tests
    in this file passing (measured), so what the tier does for the first 100 ms of every
    entry's life was unasserted.

    Getting it wrong is not a crash: `resident()` returns False, the lookup walks past a
    prefix that is in memory, and the request re-prefills. Silent, and it is the window a
    burst of same-prefix requests lands in.

    Blocking the daemon is what makes the window observable. The queue item is taken before
    the block, so `_pending` stays populated and no file is written -- asserted, or a
    passing `resident()` could just be reading the disk.
    """
    torch.manual_seed(0)
    toks = list(range(4 * BLOCK_TOKENS))
    state = (torch.randn(3, 4, 8, 8), torch.randn(3, 2, 16))
    pool = PagedKvPool(64, 2, 8, device=torch.device("cpu"), layer_map=(0,))
    tier = KvTier(str(tmp_path), "fp-pending", min_tokens=BLOCK_TOKENS)
    store = PrefixStore(pool, ssd=tier)

    gate = threading.Event()
    real_save = torch.save

    def held_save(*a, **kw):
        gate.wait(5)
        return real_save(*a, **kw)

    blocks = [pool.alloc_block() for _ in range(4)]
    for i, b in enumerate(blocks):
        pool.k_pool[:, b] = float(i + 1)
        pool.v_pool[:, b] = float(-i - 1)
    with unittest.mock.patch.object(torch, "save", held_save):
        assert store.insert(toks, blocks, (state[0].clone(), state[1].clone()))
        h = store._hash_all(tuple(toks))
        for _ in range(500):  # the enqueue is synchronous, the daemon's pickup is not
            if tier._pending and tier._pending_st:
                break
            time.sleep(0.01)
        assert tier._pending and tier._pending_st, (
            "nothing reached the queue, so the pending window below is not being tested"
        )
        d = os.path.join(str(tmp_path), "tilerl_kvtier")
        on_disk = [f for f in os.listdir(d) if f.endswith((".kv", ".st"))]
        assert not on_disk, (
            f"{len(on_disk)} files are already written, so a resident() hit below could be "
            "reading the disk rather than the pending table"
        )
        assert tier.resident(h), (
            "an entry in the queue is not resident, so a lookup in the ~100 ms before the "
            "save lands walks past a prefix that is in memory and re-prefills it"
        )
        cold_pool = PagedKvPool(64, 2, 8, device=torch.device("cpu"), layer_map=(0,))
        assert tier.load_kv(h, tuple(toks), [cold_pool.alloc_block() for _ in range(4)],
                            cold_pool), "load_kv did not serve a pending blob"
        assert tier.load_state(h, tuple(toks)) is not None, (
            "load_state did not serve a pending blob"
        )
        gate.set()
    _flushed(tier)


def test_a_drop_landing_mid_save_does_not_resurrect_the_prefix(tmp_path):
    """An eviction while the daemon is inside `torch.save` must not leave the bytes behind.

    `drop()` clears the pending tables and unlinks, but the daemon is already past that point
    with the blob in hand: its `torch.save` completes AFTER the drop and writes the file the
    drop just removed. So `_flush_loop` re-checks `table.get(k) is blob` afterwards and
    unlinks again when the entry is gone.

    Without that rollback the dropped prefix comes back: measured here, `ssd_entries` goes
    0 -> 1 and a `.kv` file is left on disk. That entry is worse than a leak -- `_track_written`
    indexes it, so a lookup can match a prefix the tier was told to forget, and after an
    `invalidate()` that means KV computed under the previous weights.

    The window is narrow and this is the only way to open it deterministically: block inside
    `torch.save`, drop, then release. `_flushed` afterwards, or the assertions race the daemon.
    """
    torch.manual_seed(0)
    toks = list(range(4 * BLOCK_TOKENS))
    state = (torch.randn(3, 4, 8, 8), torch.randn(3, 2, 16))
    pool = PagedKvPool(64, 2, 8, device=torch.device("cpu"), layer_map=(0,))
    tier = KvTier(str(tmp_path), "fp-race", min_tokens=BLOCK_TOKENS)
    store = PrefixStore(pool, ssd=tier)
    h = store._hash_all(tuple(toks))

    in_save, let_go, saved = threading.Event(), threading.Event(), threading.Event()
    real_save = torch.save

    def held_save(*a, **kw):
        in_save.set()
        let_go.wait(5)
        try:
            return real_save(*a, **kw)
        finally:
            saved.set()

    with unittest.mock.patch.object(torch, "save", held_save):
        assert store.insert(toks, [pool.alloc_block() for _ in range(4)],
                            (state[0].clone(), state[1].clone()))
        assert in_save.wait(5), (
            "the daemon never entered torch.save, so the drop below does not land mid-save "
            "and this test proves nothing"
        )
        tier.drop(h)
        let_go.set()
    # NOT `_flushed`: `drop()` empties the pending tables, so that helper returns immediately
    # while the daemon is still inside `torch.save` -- the first version of this test did that
    # and its negative control PASSED. `_q.unfinished_tasks` is no good either: nothing calls
    # `task_done()`, so it never reaches 0. Wait for the save that is in flight, then give the
    # daemon its next few instructions -- the rollback is the line after `torch.save` returns.
    assert saved.wait(5), "the held torch.save never completed"
    d = os.path.join(str(tmp_path), "tilerl_kvtier")
    for _ in range(20):
        if not [f for f in os.listdir(d) if f.endswith((".kv", ".st"))]:
            break
        time.sleep(0.01)

    left = [f for f in os.listdir(d) if f.endswith((".kv", ".st"))]
    assert not left, (
        f"a drop mid-save left {len(left)} files on disk: {left}. The save completed after the "
        "unlink, so the rollback in _flush_loop is what removes them"
    )
    assert tier.stats()["ssd_entries"] == 0, (
        f"the dropped prefix is indexed again ({tier.stats()['ssd_entries']} entries), so a "
        "lookup can match a prefix the tier was told to forget"
    )


def test_the_disk_tier_evicts_the_oldest_entry_past_its_byte_budget(tmp_path):
    """A byte budget that only reports is unbounded disk growth.

    Four prefixes spilled against a budget that holds two. The oldest entry's files must be
    gone and `resident()` False for it, while the newest is still there -- LRU, so the
    victim is the least recently touched, not the largest or the first inserted.

    The negative control is the budget itself: with it effectively unlimited, every file
    survives. Without that arm "the files are gone" would also pass for a tier that deleted
    them for some other reason.
    """
    torch.manual_seed(0)
    state = (torch.randn(3, 4, 8, 8), torch.randn(3, 2, 16))
    toks = list(range(16 * BLOCK_TOKENS))

    def spill_four(max_bytes: int, tag: str):
        d = str(tmp_path / tag)
        pool = PagedKvPool(128, 2, 8, device=torch.device("cpu"), layer_map=(0,))
        tier = KvTier(d, "fp-lru", min_tokens=BLOCK_TOKENS, max_bytes=max_bytes)
        store = PrefixStore(pool, ssd=tier)
        keys = []
        for k in range(1, 5):
            n = k * 4 * BLOCK_TOKENS
            blocks = [pool.alloc_block() for _ in range(n // BLOCK_TOKENS)]
            assert store.insert(toks[:n], blocks, (state[0].clone(), state[1].clone()))
            keys.append(store._hash_all(tuple(toks[:n])))
            _flushed(tier)
        return tier, keys, os.path.join(d, "tilerl_kvtier")

    # Negative control first, and it also measures what one entry costs on disk.
    big, _, d_big = spill_four(1 << 40, "unlimited")
    assert big.stats()["ssd_entries"] == 4, "setup: four entries should have landed"
    assert big.over_budget == 0, f"the unlimited arm evicted {big.over_budget} entries"
    assert len(os.listdir(d_big)) == 9, (  # 4 pairs + the .kvtier marker
        f"the unlimited arm lost files: {sorted(os.listdir(d_big))}"
    )
    one_pair = big.stats()["ssd_bytes"] // 4

    small, keys, _ = spill_four(2 * one_pair, "capped")
    assert small.over_budget >= 1, (
        f"nothing was evicted at a {2 * one_pair} byte budget (total "
        f"{small.stats()['ssd_bytes']}), so this test is inert"
    )
    assert not small.resident(keys[0]), "the oldest entry survived the byte budget"
    assert small.resident(keys[-1]), "the newest entry was evicted instead of the oldest"
    assert not os.path.exists(small._kv(keys[0])), (
        "the oldest entry left its .kv on disk, so the budget is not bounding the directory"
    )
    assert small.stats()["ssd_bytes"] <= 2 * one_pair, (
        f"still {small.stats()['ssd_bytes']} bytes tracked against a {2 * one_pair} budget"
    )


def _flushed(tier, tries: int = 500) -> None:
    """Wait for the flush daemon to land what is queued. Bounded, so a wedged writer
    fails the assert that follows rather than hanging the suite."""
    for _ in range(tries):
        if not tier._pending and not tier._pending_st:
            return
        time.sleep(0.01)


def test_prefix_hit_survives_evicting_its_own_entry():
    """submit()'s own evict_until_free can evict the entry it just matched;
    the snapshot must be read before that, not after.

    Unchanged by the FIFO->LRU switch: nothing here looks the matched entry up, and
    with no hits recorded LRU order IS insertion order, so the entry published
    first is still the first victim.

    # ponytail: this asserts eviction HAPPENED, not that it hit the matched entry
    # -- and measured, it does not. evict_until_free needs 2 blocks; the matched
    # entry's 2 are already retained by this request so freeing them yields
    # nothing, and the decoy's 2 satisfy the need, so the loop stops with the
    # matched entry still resident (probed: entries 2 -> 1, free 1 -> 3, and a
    # re-match after eviction still returns its snapshot). Two mutations moving
    # the snapshot read after eviction both survive. Making it bite needs the
    # matched entry to be the only source of free blocks, which then hits
    # "insufficient KV blocks" first -- a fixture problem, not a one-line fix.
    """
    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=9), get_backend(), num_blocks=8, num_slots=4)

    def publish(toks, n, state=None):
        blocks = [engine._kv.alloc_block() for _ in range(n)]
        engine._prefix.insert(list(toks), blocks, state)
        for b in blocks:
            engine._kv.free_block(b)  # the store keeps its own retain

    tokens = list(range(1, 4 * BLOCK_TOKENS + 1))
    key = tuple(tokens[: 2 * BLOCK_TOKENS])
    publish(key, 2, (engine._states.states[0].clone(), None))  # matched entry, LRU head
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


#: Backend calls a training forward makes that carry no gradient. State plumbing
#: moves the recurrent state in and out of the pool; neither is a differentiable op.
_GRADIENT_FREE = {"state_gather", "state_scatter"}


def test_every_op_the_training_forward_calls_is_on_the_tape():
    """A backend method missing from ``_BWD`` loses its gradient in silence.

    ``RecordingBackend.__getattr__`` returns the raw attribute for any name not in
    ``_BWD``, so an op added without registering it records nothing -- no error,
    no warning, and forward parity, the CPU twin and any output-level gate all
    still pass. Only a gradcheck on that specific op would catch it, and a
    gradcheck is written for the op you know about.

    This asserts the population instead: every op a training forward actually
    calls is either on the tape or named gradient-free above."""
    from tilerl.autograd import _BWD
    from tilerl.train import _training_kv

    called: list[str] = []

    class _Spy(RecordingBackend):
        def __getattr__(self, name):
            attr = super().__getattr__(name)
            if not callable(attr) or name.startswith("_"):
                return attr

            def logged(*args, **kwargs):
                called.append(name)
                return attr(*args, **kwargs)

            return logged

    backend = get_backend()
    model = build_random(tiny(), seed=0)
    ids, pos = np.array([[1, 2, 3, 4]]), np.arange(4)
    with Tape():
        model.forward(ids, pos, _training_kv(model, 1, 4, device=backend.device), _Spy(backend))

    untracked = sorted(set(called) - set(_BWD) - _GRADIENT_FREE)
    assert not untracked, (
        f"{untracked} run in a training forward and are not in _BWD, so the tape records "
        "nothing for them and their gradient is silently zero. Register a backward, or add "
        "them to _GRADIENT_FREE with the reason."
    )
    assert set(called) & set(_BWD), "the spy saw no tape op at all; it is not wrapping anything"


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
    assert len(losses) == 2 and all(math.isfinite(x) for x in losses), losses
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
    twiddled = {k for k in before
                if getattr(model.params[k], "_tl_layout", "natural") != "natural"}
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

    def forward(self, hidden, ids, positions, kv, backend, hidden_out=None, last_only=False):
        pos = np.atleast_2d(np.asarray(positions))
        logits = torch.zeros(*pos.shape, self.cfg.vocab_size, device=backend.device)
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                logits[i, j, self.expected.get(int(pos[i, j]) + 1, 0)] = 10.0
        if hidden_out is not None:
            hidden_out.append(torch.as_tensor(hidden))
        # Mirror DraftHead: reduce AFTER hidden_out, so the caller's [:, :1] read is
        # the row it asked for and not position 0 of a full-width block.
        if last_only is not False and logits.shape[1] > 1:
            idx = ([logits.shape[1] - 1] * logits.shape[0] if last_only is True
                   else [n - 1 for n in last_only])
            logits = logits[torch.arange(logits.shape[0]), torch.tensor(idx)].unsqueeze(1)
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
        # The pool dtype IS the attention/write kernel's ABI, and PagedKvPool defaults
        # to bf16 while sm70's is f32: without this the write_tokens_f32 kernel rejects
        # K with "input K dtype mismatch, expected float32" and six arms of the parity
        # test below fail on a V100 for a reason that is this helper's, not the engine's.
        kv_pool=PagedKvPool(nblk, cfg.num_kv_heads, cfg.head_dim, num_layers=1,
                            device=backend.device, layer_map=(0,),
                            dtype=getattr(backend, "io", torch.bfloat16)),
        state_pool=None, seq_q_lens=torch.tensor([n]),
    )
    return draft.forward(hid[-1][:, :n], np.array([toks[1:]]),
                         np.arange(1, n + 1), kv, backend)[0, -1].float()


def test_every_draft_call_site_is_covered_by_the_timer():
    """`_draft_ms` must see EVERY draft step, not just the ones on one code path.

    The engine calls `_draft.step` from two places — `_run_forward` (eager) and
    `_run_decode_graph`. Instrumenting only the eager one produced a number 31x too
    large on the V100: the graph path took 212 of 218 ticks, so the timer sampled the
    6 warm/mixed ticks, which carry prefill work, and reported 165.97 ms/forward
    against a subtracted 4.80-5.30. A mean over an unrepresentative subset looks
    exactly like a mean, which is why this is a gate and not a comment.

    Asserted by source, because the failure is a MISSING call and no CPU run reaches
    the graph path: every `_draft.step(` in engine.py must sit in a `_draft_ms is
    None` branch or inside the timing helper itself. Counting recorded entries at
    runtime cannot see a site that was never wired.
    """
    import re
    from pathlib import Path

    import tilerl.engine as eng_mod

    src = Path(eng_mod.__file__).read_text().split("\n")
    sites = [i for i, ln in enumerate(src) if re.search(r"self\._draft\.step\(", ln)]
    assert len(sites) >= 2, f"expected both draft call sites, found {len(sites)}"
    helper = next(i for i, ln in enumerate(src) if "def _draft_step_timed" in ln)
    for i in sites:
        if i > helper:
            continue  # the call inside the timing helper is the timed one
        # the three lines above a plain call must gate it on the timer being off
        window = "\n".join(src[max(0, i - 3):i])
        assert "_draft_ms is None" in window, (
            f"engine.py:{i + 1} calls _draft.step outside the timer's reach:\n{window}")
    # and the timed twin must exist beside it
    assert sum("_draft_step_timed(" in ln for ln in src) >= 3, (
        "each gated call site needs a _draft_step_timed twin plus the definition")


def test_the_draft_forward_counter_counts_forwards_not_ticks():
    """``DraftHead.forwards`` is the divisor a direct draft timing uses, and it must
    count actual forwards, not ticks times depth.

    The chain loop breaks when a row runs out of blocks (`spec.py:370`), so a
    depth-d tick can run fewer than d forwards. Dividing a per-tick timing by the
    configured depth would then under-price the draft, silently and in the direction
    that makes speculation look better -- which is the number
    `wins/2026-09-04-a-difference-amplifies-its-operands-noise.md` exists to stop
    being quoted loosely. Counted against a spy on the one function that does a
    draft forward, so the two cannot drift.
    """
    cfg = tiny()
    backend = get_backend()
    model = build_random(cfg, seed=5)
    draft = _random_draft(cfg, 11, model)
    depth = 3
    engine = build_engine(cfg, model, backend, num_blocks=64, num_slots=4, max_batch=4,
                          max_total_tokens=512, draft=draft, spec_depth=depth)
    calls = {"n": 0}
    inner = draft.forward

    def spy(*a, **k):
        calls["n"] += 1
        return inner(*a, **k)

    draft.forward = spy
    try:
        rid = engine.submit([3, 4, 5, 6, 7],
                            SamplingParams(temperature=0.0, max_new_tokens=12, seed=0))
        _drain(engine, [rid], 12)
    finally:
        draft.forward = inner
    assert calls["n"] > 0, "the draft never ran"
    assert draft.forwards == calls["n"], (
        f"the counter reads {draft.forwards} against {calls['n']} real forwards")
    ticks = engine.stats()["decode_forwards"]
    assert calls["n"] <= ticks * depth, (
        f"{calls['n']} forwards over {ticks} ticks exceeds {depth} per tick")


@pytest.mark.parametrize(
    "rows,plen,batched_tokens,depth",
    [
        (1, 6, 512, 1),    # one row, prompt in one chunk, one draft
        (3, 6, 512, 1),    # ragged widths: rows commit different counts after a reject
        (1, 24, 8, 1),     # chunked prefill: the prompt spans several forwards
        (1, 6, 512, 2),    # a chain, so a rejected step leaves stale KV behind it
        # A prompt that ENDS one token short of a block boundary, batched. The tick
        # after prefill drafts position `plen`, which is the first position in the
        # next block -- and the engine grows r.blocks for `decodes` only, so the row
        # does not own it yet. This is the minimal form of the bug the `manychunks`
        # case below was blamed on: one block, no chunking, fails on tick 1.
        (2, 15, 512, 1),
        # Many chunks AND more than one row. Two independent bugs live here; the first
        # is fixed, the second is not, so this case is xfail rather than deleted.
        #
        # (1) FIXED in spec.py: a row can advance several prefill ticks without
        #     drafting, so its span outruns the one-forward hidden the engine keeps. At
        #     ctx=2048 on the 27B it asked for 1535 positions against 512 of hidden and
        #     died in the fc concat as "1535 vs 511", three frames from the cause.
        # (2) ALSO FIXED in spec.py, and it was not what this case's name says. The
        #     engine grows r.blocks for `decodes` (engine.py:707) but `_draft.step`
        #     runs on EVERY row, so a row that just left prefill writes a position the
        #     trunk has no block for -- block index 1 of a 1-column table. Chunked
        #     prefill was incidental: it reproduces with 16-token prompts in one block
        #     and no chunking at all, on the first tick after prefill. `hi` is now
        #     clamped to the blocks the row owns, like the hidden span above.
        (2, 32, 8, 1),
    ],
    ids=["single", "multirow", "chunked", "depth2", "blockedge", "manychunks"],
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

    def spy(hidden, ids, positions, kv, be, hidden_out=None, last_only=False):
        out = inner(hidden, ids, positions, kv, be, hidden_out=hidden_out,
                    last_only=last_only)
        # The readout must be reduced whenever the tick is wider than one position: a
        # 512-position prefill chunk reads ONE row out of a [512, vocab] f32 readout,
        # which is 485 MiB and OOMed ctx=8192 on a 32 GB card. Correctness cannot see
        # this -- the unreduced path returns the same token.
        assert out.shape[1] == 1 or np.asarray(ids).shape[1] == 1, (
            f"draft readout is {out.shape[1]} positions wide for a "
            f"{np.asarray(ids).shape[1]}-position tick: pass last_only")
        # chain step 0 on full-batch ticks only: later steps consume the draft's
        # own hidden, and a partial batch would shift row -> request
        if step["n"] % max(depth, 1) == 0 and out.shape[0] == rows:
            pos = np.asarray(positions)
            for i in range(rows):
                # out[i, -1] is the last valid row either way: with last_only the
                # readout is already reduced to it, without it T-1 is that position.
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
        # The DRIFT is the invariant; argmax agreement is a LOTTERY on it wherever the
        # top-2 gap is narrower. Measured on sm90: `chunked` margin 0.2863 against drift
        # 0.3562 FAILED while `multirow` row 2 at margin 0.2667 / drift 0.3536 -- a worse
        # ratio -- passed. Same positions on cpu read margin 4.6552 / drift 0.2775, 16.8x
        # the other way, so the margins are per-arch and no fixed tolerance fits either.
        # `rel` is the assertion that measures the difference instead of betting on it:
        # sm90's four positions span 1.617e-02 to 5.822e-02 against the 0.1 bound, and a
        # control that moves one logit by +50 reads 3.235e-01 -- caught with 3.2x margin,
        # 5.6x above the worst real drift. So dropping the argmax compare loses no
        # coverage that was ever real. dense vs paged hidden differ ~4e-3, x10 through
        # the head; a chain-local KV reads ~1.4, which is what 0.1 was sized against.
        rel = ((full - got).norm() / full.norm()).item()
        assert rel < 0.1, (
            f"row {i} at position {pos}: norm-relative {rel:.2e}, engine drafted "
            f"{int(got.argmax())} against full context's {int(full.argmax())}"
        )


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
    # decode_graph=False by declaration: every assertion below reads a Python spy
    # that runs once at capture and never at replay, so this gate can only be an
    # eager one. Left on, capture aborts on the spy's own D2H copy and the engine
    # silently drops to eager anyway -- for every width, not just this one.
    engine = build_engine(cfg, model, backend, num_blocks=64, num_slots=1, max_batch=4,
                          max_total_tokens=512, draft=_OracleDraft(cfg, expected),
                          spec_depth=depth, decode_graph=False)
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
def test_a_verify_tick_submits_batch_times_width_rows():
    """A bench that submits ONE request measures W rows, not B*W, and the sm70 rung
    is chosen on rows: at B=1 depth 3 is 4 rows (rung 4, ncols=2 off) while serving's
    B=4 is 16 (rung 32, on). A spec A/B run that way compared a kernel against itself
    and read a flat 0.995-1.000x as "a wash" —
    errors/2026-09-03-the-spec-ncols-ab-ran-at-b1.md. Assert the row count the engine
    really submits, per concurrent request count, so the two cannot be confused again."""
    import tilerl.engine as eng

    cfg = tiny()
    model = build_random(cfg, seed=7)
    expected = dict(enumerate(range(3, 40)))
    seen: list[int] = []
    orig = eng.Engine._run_forward

    def spy(self, decodes, prefills, chunks):
        if decodes and not prefills:
            seen.append(len(decodes) * (1 + len(decodes[0].drafts)))
        return orig(self, decodes, prefills, chunks)

    engine = build_engine(
        cfg, model, get_backend(), num_blocks=32, num_slots=4, max_batch=4,
        max_total_tokens=512, draft=_OracleDraft(cfg, expected), spec_depth=3,
    )
    eng.Engine._run_forward = spy
    try:
        rids = [engine.submit([3, 4, 5, 6], SamplingParams(temperature=0.0,
                                                           max_new_tokens=12, seed=0))
                for _ in range(4)]
        _drain(engine, rids, max_new_tokens=12)
    finally:
        eng.Engine._run_forward = orig

    assert seen, "no pure-decode tick ran"
    # 4 concurrent rows at width <=4: a full tick is 16 rows, never the 4 a B=1 run sees.
    assert max(seen) > 4, f"widest tick was {max(seen)} rows: this is a B=1 measurement"
    assert max(seen) <= 16, f"tick exceeded max_batch*(1+depth): {max(seen)}"


def test_a_padded_decode_tick_needs_a_spare_state_slot():
    """A tick with fewer rows than its graph bucket permanently reserves one state slot
    for the padding rows (engine.py:827) out of the same pool, and never returns it. So
    num_slots == max_batch leaves max_batch-1 for requests and the next submit() raises
    "LinearStatePool exhausted" — which killed two 10-minute pod runs, because the engine
    swallows its own failure (`except RuntimeError: B = n`) and only the caller sees it.
    Pure bookkeeping, so it runs on the CPU target where no graph is ever captured."""
    from pathlib import Path

    from tilerl.engine import _GRAPH_BUCKETS
    def bucket(rows: int, max_batch: int) -> int:
        b = next((c for c in _GRAPH_BUCKETS if c >= rows), None)
        return rows if b is None or max_batch < b else b

    # A batch draining one request at a time hits n = max_batch-1, which pads at 4.
    pads = {n: n < bucket(n, 4) for n in (1, 2, 3, 4)}
    assert pads[3], f"n=3 must pad into the 4 bucket, else this test guards nothing: {pads}"
    assert not pads[4], f"a full batch must not pad: {pads}"

    # So any harness sizing num_slots == max_batch is one slot short. bench_ctx_decode.py
    # is the one that was, twice; assert its sizing keeps room for the pad row.
    src = (Path(__file__).resolve().parent.parent / "scripts/bench_ctx_decode.py").read_text()
    assert "slots = b + 2" in src, "bench_ctx_decode must size num_slots above max_batch"
    assert "num_slots=slots, max_batch=b" in src, "bench_ctx_decode must pass them apart"


def test_a_batch_between_rungs_warns_about_its_padding():
    """B*W strictly between two sm70 rungs launches the whole upper rung, and a padding
    row costs 3.3x the useful work on a real one (7.53 vs 2.29 ms measured), so the
    shipped max_batch=4 at depth 3 -- 16 rows on the 32 rung -- gets 42.7 tok/s where
    B=8's full rung gets 75.0. The old guard only fired PAST the top rung, so the entire
    3..7 band was silent. Pure arithmetic over LADDER_WIDTHS: runs on the CPU target,
    where the sm70 dispatch this describes never executes."""
    from pathlib import Path

    from tilerl.spec import LADDER_WIDTHS

    def rung(rows):
        return next((w for w in LADDER_WIDTHS if w >= rows), None)

    # The band the old guard missed: every one of these pays for 32 rows.
    for b in range(3, 8):
        rows = b * 4
        assert rows not in LADDER_WIDTHS, f"max_batch={b} would be silent by design"
        assert rung(rows) == 32, f"max_batch={b}: {rows} rows -> {rung(rows)}"

    # Negative controls: the two batch sizes that fill a rung must stay silent, or the
    # warning fires on the config it is telling people to use.
    assert rung(2 * 4) == 8 and 8 in LADDER_WIDTHS, "max_batch=2 fills the 8 rung"
    assert rung(8 * 4) == 32 and 32 in LADDER_WIDTHS, "max_batch=8 fills the 32 rung"

    # The suggestion the guard prints must itself fill the rung, not restate the problem.
    src = (Path(__file__).resolve().parent.parent / "src/tilerl/engine.py").read_text()
    assert "rows not in LADDER_WIDTHS" in src, "engine must warn between rungs, not only past the top"
    assert "are padding" in src, "the warning must say how much of the launch is wasted"

    # A suggestion is only possible when the verify width DIVIDES the rung. Depth 3 (W=4)
    # does, which is why testing only depth 3 hid this: at depth 2 (W=3) NO batch lands on
    # a rung, and `rung // W` names 10 -- 30 rows, which pads too. The guard must stay
    # silent there and let the depth warning carry it.
    for depth, expect_fix in ((1, True), (2, False), (3, True), (7, True)):
        w = 1 + depth
        between = [b for b in range(2, 12) if (b * w) not in LADDER_WIDTHS and b * w < 32]
        assert between, f"depth={depth}: nothing in the padding band to check"
        for b in between:
            up = next(x for x in LADDER_WIDTHS if x > b * w)
            fills = up % w == 0
            assert fills == expect_fix, f"depth={depth} max_batch={b}: divisibility {fills}"
            if fills:
                assert (up // w) * w in LADDER_WIDTHS, f"depth={depth}: suggestion still pads"
    assert 'if rung % w == 0 else ""' in src, "the guard must withhold an impossible suggestion"


def test_the_sm70_split_count_follows_the_query_width():
    """backend.py picks KVSPLIT by query width, and the two constraints sit at opposite
    ends: 32 splits are 1.20x faster at S=1 (205.1 vs 246.5 us, ctx=4096) where PO is
    3 MiB, and by S=32 the two are 1.005x apart while PO reaches 1.500 GiB and OOMs a
    32 GB card at B=8 ctx=512. So a narrow tick must get 32 and a wide one 16, and the
    threshold must sit above the widest verify a spec tick submits -- at depth 7, S=8.
    Reads the source: the dispatch is sm70-only and never executes on the CPU target."""
    from pathlib import Path

    from tilerl_kernels.registry import (
        _SM70_KERNELS,
        SM70_KVSPLIT,
        SM70_KVSPLIT_WIDE,
        sm70_kvsplit,
    )

    assert SM70_KVSPLIT_WIDE < SM70_KVSPLIT, "the wide tick must be the one that saves bytes"

    def po_gib(s, ks):  # 8 rows, 24 heads, D=256, f16 -- the shape that OOMed
        return 8 * s * 24 * ks * 256 * 2 / 1024**3

    # Call the shipped rule, don't restate it: a copy here would pass while backend drifts.
    assert sm70_kvsplit(1) == SM70_KVSPLIT, "decode must keep the faster split count"
    assert sm70_kvsplit(4) == SM70_KVSPLIT, "a depth-3 verify is still narrow"
    assert sm70_kvsplit(512) == SM70_KVSPLIT_WIDE, "a prefill-width tick must halve PO"
    # The threshold has to clear every verify width the ladder can submit, or a spec
    # tick silently takes the slower kernel. Depth 7 is the widest, S=8.
    assert sm70_kvsplit(1 + 3) == SM70_KVSPLIT, "depth 3 (S=4) must stay on the narrow count"
    # And it must actually fix the failing case, not merely differ from it.
    assert po_gib(512, sm70_kvsplit(512)) == 0.75, f"wide PO is {po_gib(512, sm70_kvsplit(512))}"
    assert po_gib(512, SM70_KVSPLIT) == 1.5, "the shipped narrow count is what OOMed"

    # The registry must hand over bare factories: a closure that pins KVSPLIT swallows
    # the call site's choice with a TypeError, which is how this was wired before.
    import inspect

    for name in ("paged_attention_split", "paged_attention_split_combine"):
        sig = inspect.signature(_SM70_KERNELS[name])
        assert "KVSPLIT" in sig.parameters, f"{name} must accept KVSPLIT from the call site"

    src = (
        Path(__file__).resolve().parent.parent
        / "packages/tilerl-kernels/src/tilerl_kernels/backend.py"
    ).read_text()
    assert "KVSPLIT=ks" in src, "backend must pass the chosen split count to both kernels"
    assert src.count("KVSPLIT=ks") == 2, "split and combine must agree, or the ABI asserts"


def test_the_cpu_kv_pool_keeps_mains_dtype_now_that_build_engine_passes_one():
    """``build_engine`` passes ``backend.io`` as the paged KV pool's dtype; origin/main
    passed NOTHING and the pool took ``PagedKvPool``'s bf16 default. So the branch
    changes the CPU parity cell's KV dtype **bf16 -> f32** — and the cause is not the
    arch split, which agrees with main on cpu (main computed io inline as
    `cuda ? bf16 : f32`, i.e. f32 on cpu). The cause is that PagedKvPool's DEFAULT
    disagrees with main's own io rule, so routing io into it moves cpu even when io
    is right.

    K and V are no longer rounded to bf16 on store, so ``paged_attention`` computes
    different values on the target that certifies every kernel in this repo, and the
    pool costs 2x the bytes. The whole suite stayed green: a parity check compares
    TileLang against a torch reference in the SAME process, so both sides moved
    together and no assertion could see it.

    Asserts the pool a CPU engine really builds, not a dtype table — a rule restated
    here would agree with itself while build_engine drifted."""
    cfg = tiny()
    backend = get_backend()
    if backend.arch != "cpu":
        pytest.skip("this pins the CPU parity cell's dtype; other arches have their own")
    model = build_random(cfg, seed=3)
    e = build_engine(cfg, model, backend, num_blocks=8, num_slots=2, max_batch=2,
                     max_total_tokens=64)
    got = e._kv.k_pool.dtype
    assert got is torch.bfloat16, (
        f"the CPU KV pool is {got} where origin/main built bfloat16. build_engine now "
        "passes backend.io (f32 on cpu) where main passed no dtype and PagedKvPool "
        "defaulted to bf16, so K/V are no longer rounded on store and the parity cell "
        "computes different attention values than main. Pass io only on cuda, or make "
        "the pool's default follow it."
    )


def test_the_draft_readout_reduction_picks_the_last_valid_row():
    """`last_only` cuts the draft readout from [rows, T, vocab] to [rows, 1, vocab] --
    1.41 GiB down to 7.6 MiB at B=8 ctx=512, which is the difference between OOM and
    running. It must select the LAST VALID position per row, and the existing parity
    test cannot check that: its spy reads out[i, -1] AFTER the reduction, so a wrong
    index inside the reduction is compared against whatever that index chose. Picking
    row 0 passes all four parity cases.

    This drives the real DraftHead.forward twice on identical input and asserts the
    reduced readout equals the full-width one at the row it claims."""
    cfg, model = _build_model("tiny", seed=0)
    backend = get_backend()
    draft = _random_draft(cfg, 21, model)

    toks = [3, 4, 5, 6, 7, 8]
    n = len(toks) - 1
    hid: list = []
    model.forward(np.array([toks]), np.arange(len(toks)),
                  _training_kv(model, 1, len(toks), device=backend.device),
                  backend, hidden_out=hid, last_only=False)
    nblk = -(-n // BLOCK_TOKENS) + 1

    def run(**kw):
        kv = BatchKv(
            block_table=torch.arange(nblk, dtype=torch.long).reshape(1, nblk),
            seq_len=torch.tensor([n]), state_slot=torch.zeros(1, dtype=torch.long),
            kv_pool=PagedKvPool(nblk, cfg.num_kv_heads, cfg.head_dim, num_layers=1,
                                device=backend.device, layer_map=(0,),
                                dtype=getattr(backend, "io", torch.bfloat16)),
            state_pool=None, seq_q_lens=torch.tensor([n]),
        )
        return draft.forward(hid[-1][:, :n], np.array([toks[1:]]),
                             np.arange(1, n + 1), kv, backend, **kw)

    full = run()
    assert full.shape[:2] == (1, n), full.shape
    # Positions must differ, or "picked the right row" is unfalsifiable.
    assert not torch.allclose(full[:, 0], full[:, n - 1], atol=1e-4), \
        "this head gives every position the same logits; the test proves nothing"

    for want, kw in ((n - 1, {"last_only": True}), (n - 2, {"last_only": [n - 1]})):
        got = run(**kw)
        assert got.shape == (1, 1, full.shape[-1]), got.shape
        assert torch.allclose(got[0, 0], full[0, want], atol=1e-3), (
            f"{kw}: reduction returned argmax {int(got[0, 0].argmax())}, "
            f"position {want} has {int(full[0, want].argmax())}"
        )


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


def test_a_mixed_tick_pays_the_widest_rows_width_on_every_row():
    """A tick's activations are `len(rows) x width`, not `sum(tokens)` -- so one prefill
    row makes the whole tick as wide as itself.

    `engine.py:741-742` builds one rectangle from the WIDEST row:

        width = ceil(max(seq_q) / _PREFILL_BUCKET) * _PREFILL_BUCKET  if chunk > 1
        input_ids = np.zeros((len(rows), width))

    `max_num_batched_tokens` bounds `sum(chunks)`, so it caps prompt tokens per tick and
    says nothing about the rectangle they are padded into. At B=8 on the 27B that cost
    272 MiB in `silu_mul` -- 8 rows x 512, against the 102 MiB the token budget suggests --
    and OOMed a 32 GiB card at spec depth 1. The ceiling is not "B=8 needs more memory",
    it is "a mixed tick pays the widest row's width on every row".

    Silent until it OOMs: shapes and tokens are both correct, only the padding is wasted,
    and a tiny model's rectangle fits anywhere. Asserted on the width RULE rather than on
    a reconstructed tick, because reconstructing from outside `step()` reads state that
    `_build_plan` has not yet promoted -- measured, it reports 0.00x waste. The e2e half
    below only has to show a mixed tick happens at all; `mixed_forwards` had no coverage.
    """
    bucket = _PREFILL_BUCKET

    def rect(seq_q, chunks=()):
        """The rectangle _run_forward materializes, mirroring engine.py:740-742.

        The bucket branch is gated on `max(chunks)` -- the PREFILL chunks -- not on
        `max(seq_q)`. A verify tick has no prefills, so it takes the exact branch even at
        width 5; only a tick carrying a multi-token prefill pays a bucket. Getting this
        wrong is how the first version of this test asserted `rect([5]*8) == 40` and got
        512.
        """
        chunk = max(chunks, default=0)
        width = -(-max(seq_q) // bucket) * bucket if chunk > 1 else max(seq_q)
        return len(seq_q) * width

    # A decode row (1 position) beside a prefill chunk: the decode row is padded from 1
    # to the bucket, so the rectangle is 2x the bucket for bucket+1 real tokens.
    assert rect([1, bucket], chunks=[bucket]) == 2 * bucket
    assert rect([1, bucket], chunks=[bucket]) / (1 + bucket) > 1.9, "a mixed tick must waste ~2x on 2 rows"
    # Eight rows, one of them prefilling: the OOM shape, 8x the widest row.
    assert rect([1] * 7 + [512], chunks=[512]) == 8 * 512
    assert rect([1] * 7 + [512], chunks=[512]) / (7 + 512) > 6.0, "the 27B OOM shape wastes >6x"
    # Decode-only ticks are NOT padded: chunk == 1 takes the exact-width branch.
    assert rect([1] * 8) == 8, "a decode-only tick must not pay a bucket"
    # Bucket rounding, on a width that is NOT already a multiple -- the cases above use
    # 64 and 512, where rounding is a no-op, so they cannot see it. Checked: mutating the
    # rounding away leaves every one of them passing.
    assert rect([1, 96], chunks=[96]) == 2 * 128, "96 must round up to the 128 bucket"
    assert rect([1, 65], chunks=[65]) == 2 * 128
    assert rect([1, 63], chunks=[63]) == 2 * 64
    # A verify tick is exact: no prefill chunk, so no bucket, even at width 5.
    assert rect([5] * 8) == 8 * 5
    # And a SINGLE-token prefill chunk does not trigger the bucket either (chunk > 1).
    assert rect([1, 1], chunks=[1]) == 2

    engine = _build_engine(seed=91)
    try:
        rng = np.random.default_rng(5)
        params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=1)
        short = engine.submit(rng.integers(3, 320, size=4).astype(np.int64), params)
        for _ in range(4):
            engine.step()
            if any(r.phase == _PHASE_DECODE for r in engine._running):
                break
        assert any(r.phase == _PHASE_DECODE for r in engine._running), "short prompt never decoded"
        long_ = engine.submit(rng.integers(3, 320, size=96).astype(np.int64), params)
        for _ in range(24):
            engine.step()
            if engine.stats()["mixed_forwards"]:
                break
        assert engine.stats()["mixed_forwards"], (
            "no decode+prefill tick occurred, so the rectangle above was never reached "
            "in a real run; the arithmetic asserts still hold but nothing exercises them"
        )
        _drain(engine, [short, long_], max_new_tokens=8)
    finally:
        engine.shutdown()


def test_the_draft_prefill_width_is_bucketed_like_the_trunks():
    """Every distinct prompt length used to give the draft a new kernel shape.

    `engine.py:740` buckets the trunk's prefill width to `_PREFILL_BUCKET`, and
    `spec.py` took `w = max(hi - lo + 1)` raw. The draft runs the same two kernels
    that take `seq_q_lens` (`write_tokens_f32`, `paged_attention_split`), both of
    which bake their shape, so a prompt nobody had asked before compiled them
    inline: measured on the live V100 server, a first visit at a new prompt length
    paid **14 compiles / 15.5 s** and read **4.4 tok/s** where the identical prompt
    repeated read **45.0**.

    Asserting on the widths the draft's own forward SEES, not on a formula --
    a mirror of the arithmetic passes even when `spec.py` stops calling it.
    """
    cfg = tiny()
    model = build_random(cfg, seed=31)
    draft = _random_draft(cfg, 32, model)
    engine = build_engine(
        cfg, model, backend=get_backend(), num_blocks=64, num_slots=4, max_batch=4,
        max_total_tokens=512, draft=draft, spec_depth=1,
    )
    widths: list[int] = []
    tables: list[int] = []
    inner = draft.forward

    def spy(hidden, ids, positions, kv, be, hidden_out=None, last_only=False):
        widths.append(int(np.asarray(ids).shape[1]))
        tables.append(int(kv.block_table.shape[1]))
        return inner(hidden, ids, positions, kv, be, hidden_out=hidden_out,
                     last_only=last_only)

    draft.forward = spy
    try:
        rng = np.random.default_rng(7)
        params = SamplingParams(temperature=0.0, max_new_tokens=4, seed=1)
        # Three prompt lengths that are NOT bucket multiples and are pairwise
        # distinct mod the bucket -- unbucketed they give three shapes, bucketed one.
        for plen in (19, 37, 53):
            _drain(engine, [engine.submit(rng.integers(3, 320, size=plen).astype(np.int64),
                                          params)], max_new_tokens=4)
    finally:
        draft.forward = inner
        engine.shutdown()

    wide = sorted({w for w in widths if w > 1})
    assert wide, f"the draft never ran a multi-position tick: {widths}"
    assert wide == [_PREFILL_BUCKET], (
        f"draft prefill widths {wide} are not all the bucket ({_PREFILL_BUCKET}): "
        "three distinct prompt lengths compiled three kernel shapes"
    )
    # The block-table width is the SECOND shape axis, and bucketing w alone left it
    # varying: on the served path that still cost 4-8 compiles per new prompt length,
    # all at S=64 with tables [1,1] / [1,3] / [1,4] / [1,5] / [1,6]. `Mb` is compiled
    # in (engine.py:666 says so for the trunk), so it must be the pool size, not the
    # blocks a row happens to own.
    assert len(set(tables)) == 1, (
        f"draft block-table widths {sorted(set(tables))} vary: Mb is a compiled-in "
        "dimension, so each width is another kernel"
    )
