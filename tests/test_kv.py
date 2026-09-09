"""Hermetic CPU tests for tilerl.kv_cache.

Covers: roundtrip, prefix hit/miss incl. hash-collision, refcount/CoW,
state pool, pool exhaustion, and the shared-prefix fork lifecycle.
"""

import pytest
import torch

from tilerl.kv_cache import (
    BLOCK_TOKENS,
    BatchKv,
    LinearStatePool,
    PagedKvPool,
    PrefixStore,
    _nbytes,
)


def _kv(seed: int, n: int, heads: int = 1, dim: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    # Match PagedKvPool's default device (the backend target device): the pool
    # lands on mps under TILERL_TARGET=metal, and write_block/readback compares
    # against these tensors directly.
    from tilerl_kernels.backend import get_backend

    g = torch.Generator().manual_seed(seed)
    k = torch.randn(heads, n, dim, generator=g, dtype=torch.bfloat16).to(get_backend().device)
    return k, k.clone()


# --------------------------------------------------------------------- roundtrip


def test_pool_shapes_and_dtype():
    pool = PagedKvPool(num_blocks=4, num_kv_heads=2, head_dim=8, num_layers=3, device="cpu")
    assert pool.k_pool.shape == (3, 4, 2, BLOCK_TOKENS, 8)
    assert pool.v_pool.shape == (3, 4, 2, BLOCK_TOKENS, 8)
    assert pool.k_pool.dtype == torch.bfloat16
    assert pool.v_pool.dtype == torch.bfloat16
    assert pool.free_blocks == 4
    assert pool.used_blocks == 0


def test_pool_layer_map_is_dense():
    """``layer_map`` packs full-attn layers densely and maps a GLOBAL layer
    index to its plane; a non-full-attn layer raises (regression: the pool
    shaped on ``num_layers``, so 3/4 of the 27B pool was permanently zero)."""
    pool = PagedKvPool(4, 1, 4, layer_map=(1, 3))
    assert pool.k_pool.shape[0] == 2 and pool.num_layers == 2
    b = pool.alloc_block()
    k1, v1 = _kv(0, BLOCK_TOKENS)
    k3, v3 = _kv(1, BLOCK_TOKENS)
    pool.write_block(b, 0, k1, v1, layer=1)
    pool.write_block(b, 0, k3, v3, layer=3)
    kp1, vp1 = pool.kv_layer(1)
    kp3, vp3 = pool.kv_layer(3)
    assert torch.equal(kp1[b, :, :BLOCK_TOKENS], k1)
    assert torch.equal(vp1[b, :, :BLOCK_TOKENS], v1)
    assert torch.equal(kp3[b, :, :BLOCK_TOKENS], k3)
    assert torch.equal(vp3[b, :, :BLOCK_TOKENS], v3)
    with pytest.raises(KeyError):
        pool.kv_layer(0)  # global 0 is not a full-attn plane here
    with pytest.raises(KeyError):
        pool.write_block(b, 0, k1, v1, layer=0)


def test_write_read_roundtrip():
    pool = PagedKvPool(4, 2, 8)
    b = pool.alloc_block()
    k, v = _kv(0, BLOCK_TOKENS, heads=2, dim=8)
    pool.write_block(b, 0, k, v)
    assert torch.equal(pool.k_pool[0, b, :, :BLOCK_TOKENS], k)
    assert torch.equal(pool.v_pool[0, b, :, :BLOCK_TOKENS], v)
    # partial slice at a nonzero offset on a fresh block; leading region stays zero
    b2 = pool.alloc_block()
    k2, v2 = _kv(1, 3, heads=2, dim=8)
    pool.write_block(b2, 5, k2, v2)
    assert torch.equal(pool.k_pool[0, b2, :, 5:8], k2)
    assert torch.equal(pool.v_pool[0, b2, :, 5:8], v2)
    assert torch.all(pool.k_pool[0, b2, :, :5] == 0)


def test_write_block_bounds_and_shapes():
    pool = PagedKvPool(2, 1, 4)
    b = pool.alloc_block()
    k, v = _kv(0, 5)
    with pytest.raises(ValueError):
        pool.write_block(b, 14, k, v)  # 14 + 5 > 16
    with pytest.raises(ValueError):
        pool.write_block(b, 0, torch.randn(2, 5, 4, dtype=torch.bfloat16), v)  # wrong heads
    with pytest.raises(ValueError):
        pool.write_block(b, 0, k, torch.randn(1, 4, 4, dtype=torch.bfloat16))  # k/v mismatch


def test_paged_gather_mimics_paged_attention():
    """The pool layout must gather by block id, as paged_attention will."""
    pool = PagedKvPool(4, 2, 8)
    b0 = pool.alloc_block()
    b1 = pool.alloc_block()
    k0, _ = _kv(0, BLOCK_TOKENS, heads=2, dim=8)
    k1, _ = _kv(1, BLOCK_TOKENS, heads=2, dim=8)
    pool.write_block(b0, 0, k0, k0)
    pool.write_block(b1, 0, k1, k1)
    gathered = torch.cat([pool.k_pool[0, b] for b in (b0, b1)], dim=1)
    assert torch.equal(gathered, torch.cat([k0, k1], dim=1))


# -------------------------------------------------------------- refcount


def test_refcount_lifecycle():
    pool = PagedKvPool(4, 1, 4)
    b = pool.alloc_block()
    assert pool.refcount[b] == 1
    assert pool.free_blocks == 3
    pool.retain(b)
    assert pool.is_shared(b)
    pool.free_block(b)  # one owner gone, one remains -> not recycled
    assert pool.refcount[b] == 1
    assert pool.free_blocks == 3
    pool.free_block(b)  # last owner -> recycled
    assert pool.refcount[b] == 0
    assert pool.free_blocks == 4
    with pytest.raises(RuntimeError):
        pool.free_block(b)  # double free
    with pytest.raises(RuntimeError):
        pool.retain(b)  # retaining a free block would corrupt the free list






# -------------------------------------------------------------- prefix store


def test_prefix_longest_match():
    pool = PagedKvPool(8, 1, 4)
    store = PrefixStore(pool)
    # engine pattern: insert each completed block span
    blocks = [pool.alloc_block() for _ in range(3)]
    toks = list(range(48))
    store.insert(toks[:16], blocks[:1])
    store.insert(toks[:32], blocks[:2])
    store.insert(toks, blocks)
    # exact hit
    hit = store.lookup(toks)
    assert hit is not None and hit.length == 48 and hit.blocks == tuple(blocks)
    # longest stored prefix of a longer query
    hit = store.lookup(toks + [99, 100])
    assert hit.length == 48
    # shorter query resolves to the block-boundary prefix
    hit = store.lookup(toks[:20])
    assert hit.length == 16 and hit.blocks == (blocks[0],)
    # cold miss
    assert store.lookup(list(range(100, 148))) is None
    s = store.stats()
    assert s["entries"] == 3 and s["lookups_matched"] == 3 and s["lookups_missed"] == 1


def test_prefix_hash_collision_is_verified():
    pool = PagedKvPool(8, 1, 4)
    # constant hash: every token sequence collides; verification must do the work
    store = PrefixStore(pool)
    store._roll = lambda *_: 0
    toks = list(range(10, 42))
    blocks = (pool.alloc_block(), pool.alloc_block())
    store.insert(toks, blocks)
    # same length, different tokens, same hash -> miss
    other = list(range(100, 132))
    assert store.lookup(other) is None
    # second sequence under the same hash chain; both must resolve correctly
    oblocks = (pool.alloc_block(), pool.alloc_block())
    store.insert(other, oblocks)
    assert store.lookup(toks).blocks == blocks
    assert store.lookup(other).blocks == oblocks
    # same-length collision: a 16-token entry under the same constant hash must
    # be rejected for a different 16-token query (verification, not hash, decides)
    store.insert(toks[:16], blocks[:1])
    assert store.lookup(other[:16]) is None
    assert store.lookup(toks[:16]).blocks == blocks[:1]


def test_prefix_eviction_releases_blocks():
    pool = PagedKvPool(4, 1, 4)
    store = PrefixStore(pool, capacity=2)
    b1 = pool.alloc_block()
    store.insert(list(range(16)), [b1])
    pool.free_block(b1)  # slot gone; store is now the sole owner
    b2 = pool.alloc_block()
    store.insert(list(range(16, 32)), [b2])
    pool.free_block(b2)
    assert pool.free_blocks == 2
    b3 = pool.alloc_block()
    store.insert(list(range(32, 48)), [b3])  # evicts the oldest entry
    pool.free_block(b3)
    assert pool.refcount[b1] == 0  # eviction released it
    assert pool.free_blocks == 2
    assert store.stats()["evictions"] == 1
    assert store.lookup(list(range(16))) is None  # evicted entry is gone


def test_prefix_eviction_reacts_to_block_pressure():
    pool = PagedKvPool(2, 1, 4)
    store = PrefixStore(pool)
    blocks = [pool.alloc_block() for _ in range(2)]
    for i, block in enumerate(blocks):
        store.insert(list(range(i * 16, (i + 1) * 16)), [block])
        pool.free_block(block)

    assert pool.free_blocks == 0
    store.evict_until_free(1)
    assert pool.free_blocks == 1
    assert store.stats()["evictions"] == 1


def test_prefix_duplicate_insert_is_noop():
    pool = PagedKvPool(4, 1, 4)
    store = PrefixStore(pool)
    b = pool.alloc_block()
    store.insert(list(range(16)), [b])
    rc_after_first = pool.refcount[b]
    store.insert(list(range(16)), [b])
    assert pool.refcount[b] == rc_after_first
    assert store.stats()["entries"] == 1


# --------------------------------------------------------------- state pool


def test_state_pool_lifecycle():
    sp = LinearStatePool(3, 2, 2, 4)
    assert sp.states.shape == (3, 2, 2, 4, 4)
    assert sp.states.dtype == torch.bfloat16
    s0 = sp.alloc_slot()
    assert torch.all(sp.states[s0] == 0)  # zeroed on alloc
    sp.states[s0].fill_(1.0)
    s1 = sp.alloc_slot()
    sp.free_slot(s0)
    s2 = sp.alloc_slot()  # LIFO: recycles s0, re-zeroed
    assert s2 == s0
    assert torch.all(sp.states[s2] == 0)
    sp.free_slot(s1)
    sp.free_slot(s2)
    slots = [sp.alloc_slot() for _ in range(3)]
    with pytest.raises(RuntimeError):
        sp.alloc_slot()  # exhausted
    for s in slots:
        sp.free_slot(s)
    with pytest.raises(RuntimeError):
        sp.free_slot(slots[0])  # double free


def test_a_failed_alloc_slot_does_not_consume_the_slot():
    """``alloc_slot`` pops before it zeroes, and the zeroing can raise -- an OOM on a
    27B is not hypothetical. Without the unwind the slot has left ``_free`` and reached
    no caller, so nothing can ``free_slot`` it and the pool is one slot smaller for the
    life of the process. Measured before the fix: 3 slots became 2."""
    sp = LinearStatePool(3, 1, 1, 8)

    class _Boom:
        def zero_(self):
            raise RuntimeError("simulated OOM inside zero_")

    victim = sp._free[-1]  # alloc_slot pops from the end
    real = sp.states
    sp.states = {victim: _Boom()}
    try:
        with pytest.raises(RuntimeError, match="OOM"):
            sp.alloc_slot()
    finally:
        sp.states = real
    # The slot came back, so the pool still offers all three.
    assert len(sp._free) == 3, f"leaked: {sp._free}"
    assert [sp.alloc_slot() for _ in range(3)]


# ----------------------------------------------------------------- exhaustion


def test_alloc_raises_under_pressure():
    pool = PagedKvPool(4, 1, 4)
    for _ in range(4):
        pool.alloc_block()
    with pytest.raises(RuntimeError):
        pool.alloc_block()


# ------------------------------------------------- shared-prefix publication


def test_a_partial_block_cannot_be_published():
    """The engine publishes whole blocks only, which is why a shared block is
    never appended to. Stated as a guard here because nothing else states it:
    without it a partial publish silently shares a page a slot keeps writing."""
    pool = PagedKvPool(8, 1, 4)
    store = PrefixStore(pool)
    blocks = [pool.alloc_block(), pool.alloc_block()]
    with pytest.raises(ValueError, match="partial block"):
        store.insert(list(range(20)), blocks)  # 2 blocks cover 32 tokens, 20 given
    assert store.insert(list(range(32)), blocks) is True




@pytest.mark.xfail(strict=True, reason="open: LRU evicts the shared prefix first, "
                   "errors/2026-09-07-a-prompts-own-publishes-evict-its-shared-prefix.md")
def test_a_prompts_own_publishes_evict_the_prefix_it_shares():
    """One conversation publishes a nested family of keys -- 6 at prefill chunk ends plus
    one per 16 generated tokens -- and LRU keeps the LONGEST, evicting the shared header its
    own tail displaced. strict=True: this goes red the day it is fixed."""
    def run(entries: int) -> tuple[int | None, int]:
        pool = PagedKvPool(64, 1, 4, device=torch.device("cpu"))
        snap = (torch.zeros(8, 8, 8), None)
        store = PrefixStore(pool, state_bytes=entries * _nbytes(snap))
        header = list(range(2 * BLOCK_TOKENS))
        convo = header + list(range(500, 500 + 6 * BLOCK_TOKENS))
        for i in range(1, 9):                                 # 8 boundary publishes
            blocks = [pool.alloc_block() for _ in range(i)]
            store.insert(convo[: i * BLOCK_TOKENS], blocks,
                         (torch.zeros(8, 8, 8), None))
            for b in blocks:
                pool.free_block(b)
        hit = store.lookup(header + list(range(9000, 9000 + 3 * BLOCK_TOKENS)))
        return (None if hit is None else hit.length), store.stats()["evictions"]

    # The control first: with room for every publish the second session HITS, so the header
    # entry is published correctly and the failure below is eviction, not absence.
    length, evictions = run(99)
    assert (length, evictions) == (2 * BLOCK_TOKENS, 0), (
        f"control: an ample budget must serve the shared header, got {length} with "
        f"{evictions} evictions -- if this fails the probe is wrong, not the store"
    )

    length, evictions = run(3)
    assert evictions > 0, "fixture: a 3-entry budget evicted nothing, so nothing is under test"
    assert length == 2 * BLOCK_TOKENS, (
        f"a second session sharing the {2 * BLOCK_TOKENS}-token header got "
        f"{'a miss' if length is None else f'length {length}'} after one conversation's own "
        f"{evictions} evictions: LRU ranked this prompt's tail above a prefix another "
        "session can use"
    )


def test_prefix_state_budget_evicts():
    pool = PagedKvPool(4, 1, 4)
    store = PrefixStore(pool, state_bytes=1000)
    for i in range(3):
        b = pool.alloc_block()
        store.insert(list(range(16 * i, 16 * (i + 1))), [b], (torch.zeros(100), None))  # 400 B
        pool.free_block(b)
    assert store.stats()["entries"] == 2 and store.stats()["state_bytes"] == 800
    assert store.lookup(list(range(16))) is None
    assert store.lookup(list(range(16, 32))).state[0].shape == (100,)
    # `entries_capacity` must track the byte budget, not restate the count cap.
    assert store.stats()["entries_capacity"] == 2, (
        f"1000 B of budget at 400 B per snapshot holds 2, not "
        f"{store.stats()['entries_capacity']}; the derived cap is not reading the budget"
    )
    assert store.stats()["capacity"] > store.stats()["entries_capacity"], (
        "the fixture's count cap does not exceed its byte cap, so this cannot tell a derived "
        "value from a restated one"
    )


def test_entries_capacity_counts_the_host_tier():
    """A demoted entry stays matchable, so the host tier's bytes are capacity too.

    `state_bytes=0` with a tier is a real config (`_demote_one` moves every snapshot to the
    host and the entry keeps its tokens and blocks). Reading only `state_bytes` reported 0
    while the store held 3 entries a lookup could match -- an operator sizing against it
    would see a store that cannot hold anything.
    """
    from tilerl.kv_cache import DramSnapshots

    state = (torch.randn(3, 4, 8, 8), torch.randn(3, 2, 16))
    one = _nbytes(state)

    def run(state_bytes: int, dram_entries: int | None) -> tuple[int, int]:
        pool = PagedKvPool(256, 2, 8, device=torch.device("cpu"), layer_map=(0,))
        dram = None if dram_entries is None else DramSnapshots(budget_bytes=dram_entries * one)
        store = PrefixStore(pool, state_bytes=state_bytes * one, dram=dram)
        toks = list(range(400))
        for length in (BLOCK_TOKENS * 2, BLOCK_TOKENS * 4, BLOCK_TOKENS * 6):
            store.insert(toks[:length],
                         [pool.alloc_block() for _ in range(length // BLOCK_TOKENS)],
                         (state[0].clone(), state[1].clone()))
        st = store.stats()
        return st["entries"], st["entries_capacity"]

    # HBM holds none, the host holds 5: the entries are all demoted and all matchable.
    entries, cap = run(0, 5)
    assert (entries, cap) == (3, 5), (
        f"state_bytes=0 with a 5-entry host tier: {entries} resident against a reported "
        f"capacity of {cap}. A capacity below the resident count is not a ceiling"
    )
    assert cap >= entries, "capacity below the resident count"

    # The no-tier arm is the control: without it, returning `capacity` unconditionally
    # would satisfy the assert above and still ignore both budgets.
    entries, cap = run(2, None)
    assert (entries, cap) == (2, 2), (
        f"no tier, 2 entries of budget: expected (2, 2), got ({entries}, {cap})"
    )

    # Both budgets sum: 2 in HBM + 3 on the host.
    entries, cap = run(2, 3)
    assert cap == 5, f"HBM 2 + host 3 should report 5, got {cap}"


def test_a_block_costs_2125_kib_at_the_27b_shape():
    """Pin the per-block byte cost the pool-sizing arithmetic is written against.

    bench_ctx_decode.py sizes its pool from this number and prints it, and a wrong
    value is invisible: the run still works, it just reserves the wrong amount and
    every "how much headroom did that arm have" comparison across runs is off. The
    comment claimed 0.92 MB for months -- 2.42x low, because it was bf16 and left
    out the draft's mirrored plane.
    """
    from tilerl.config import qwen38_27b

    cfg = qwen38_27b()
    n = 8
    # f32: sm70's attention IO dtype, which the pool matches (engine.py:1202).
    trunk = PagedKvPool(n, cfg.num_kv_heads, cfg.head_dim, device="cpu",
                        layer_map=cfg.full_attn_layers, dtype=torch.float32)
    draft = PagedKvPool(n, cfg.num_kv_heads, cfg.head_dim, num_layers=1, device="cpu",
                        layer_map=(0,), dtype=torch.float32)
    per_block = sum(
        t.numel() * t.element_size()
        for t in (trunk.k_pool, trunk.v_pool, draft.k_pool, draft.v_pool)
    ) / n
    assert per_block == 2.125 * 2**20, (
        f"a block is {per_block / 2**20:.4f} MiB; bench_ctx_decode.py sizes and prints its "
        f"pool at 2.125 MiB/block, so both that number and its comment are now wrong"
    )


def test_a_rejected_submit_does_not_release_the_prefix_stores_blocks():
    """The unwind must free what this request incremented, not what it hoped to adopt.

    submit() seeded its block list from the prefix hit before retaining any of it,
    so an alloc_slot() failure -- an engine at slot capacity, which is ordinary --
    ran the handler over blocks it never retained. free_block decrements without
    ownership tracking, so those decrements are indistinguishable from releases:
    the store's refcount falls, the block reaches the free list while the store
    still lists it, and a later request is handed a page holding someone else's KV.

    The hit needs a LONGER prompt sharing the prefix: _match_prefix treats
    matched >= len(tokens) as a miss, so resubmitting the same tokens never hits.
    """
    from tilerl.cli import _build_model
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.testing import RefBackend

    cfg, model = _build_model("tiny", seed=0)
    engine = build_engine(cfg, model, RefBackend(), num_blocks=32, num_slots=2,
                          max_batch=2)
    base = list(range(1, 49))
    rid = engine.submit(base, SamplingParams(max_new_tokens=2, seed=0))
    for _ in range(40):
        if rid in engine.poll():
            break
        engine.step()

    longer = base + list(range(100, 116))
    for _ in range(2):  # fill the slots, and hit the prefix while doing it
        engine.submit(longer, SamplingParams(max_new_tokens=8, seed=0))
    # step() to admit: the prefix match moved from submit to the planner, because a hit found
    # at submit can be evicted before the request is admitted.
    engine.step()
    assert engine.stats()["prefill_forwards"] > 0, "no forward ran; the test is inert"
    assert engine.stats()["prefix_hits"] >= 1, "no prefix hit; the test is inert"

    before = {b: n for b, n in enumerate(engine._kv.refcount) if n > 0}
    # A third request no longer raises -- it waits for capacity. What must still hold is the
    # invariant this test is named for: an admission that could not complete releases nothing
    # the store holds. Asserted on the refcounts, not on an exception.
    engine.submit(longer, SamplingParams(max_new_tokens=1, seed=0))
    engine.step()
    after = {b: engine._kv.refcount[b] for b in before}
    assert after == before, f"a blocked admission released blocks: {before} -> {after}"
    assert len(engine._waiting) == 1, "the third request was neither admitted nor queued"



def test_a_second_client_waits_for_capacity_instead_of_503ing():
    """A prompt that does not fit YET must wait, not be refused.

    The live defect: a 30,485-token prompt took 1906 of the V100's 2048 blocks (93.1%), and
    the next request 503-ed with `insufficient KV blocks for request` -- permanently, because
    `submit` allocated up front and had no later tick to retry on. Eviction was already being
    called one line above the refusal; what it could not do is reclaim blocks a LIVE request
    retains, since `free_block` is a refcount decrement. Measured: 2 entries dropped, 0 blocks
    freed.

    The fixture matches the RATIO, not the absolute size: 60 of 64 blocks is 93.8% against the
    live 93.1%. A 3/64 fixture cannot express the mechanism at any pool size.

    DRIVEN DIRECTLY, no loop thread. With `engine.run()` this arm raced and went red on
    macos-14 while ubuntu passed, taking an unrelated PR's CI with it: A can finish and release
    before the test reads `_waiting`, both are then admitted sequentially, and the vacuity guard
    fires correctly on a timing the test does not control. `step()` per tick makes contention a
    property of the pool rather than of the scheduler's speed, and lets the precondition be
    asserted BEFORE the arm instead of hoped for.
    """
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.model import build_random
    from tilerl.testing import RefBackend

    cfg = tiny(max_position_embeddings=4096)
    engine = build_engine(cfg, build_random(cfg, seed=9), RefBackend(), num_blocks=64,
                          num_slots=4, max_batch=2, max_total_tokens=4096)
    big = [1 + (j % 150) for j in range(944)]          # 59 blocks of 64 = 92.2%
    other = [200 + (j % 100) for j in range(944)]
    # Wrapped so a drop can be attributed to an admission attempt rather than to the process.
    admit_drops: list[int] = []
    _real_admit = engine._admit

    def _watched(req):
        before = engine._prefix.stats()["evictions"]
        ok = _real_admit(req)
        admit_drops.append(engine._prefix.stats()["evictions"] - before)
        return ok

    engine._admit = _watched
    a = engine.submit(big, SamplingParams(max_new_tokens=24, seed=0))
    b = engine.submit(other, SamplingParams(max_new_tokens=4, seed=0))

    # One tick admits A and must REFUSE B: the precondition, checked before the arm runs.
    engine.step()
    assert engine.stats()["prefill_forwards"] > 0, "no forward ran; the arm proves nothing"
    assert [r.req_id for r in engine._running] == [a], (
        f"expected only A admitted, got {[r.req_id for r in engine._running]} "
        f"(free_blocks {engine._kv.free_blocks} of 64)")
    assert [r.req_id for r in engine._waiting] == [b], (
        f"B was not left waiting; the pool is not contended and the arm is vacuous "
        f"(free_blocks {engine._kv.free_blocks} of 64)")

    try:
        # A's prefill chunks, then its decode, with B refused on every planner tick.
        for _ in range(400):
            if not engine._waiting:
                break
            engine.step()
        # The store must be left ALONE by the BLOCKED ADMISSIONS: its blocks are pinned by the
        # live request, so evicting them frees nothing and would flush every other client's
        # prefix cache once per planner tick.
        #
        # Attributed per admission attempt, not from the global counter. Two earlier forms of
        # this assert were wrong: `entries >= entries` passed with the guard REMOVED (the count
        # rises as the live request publishes its own chunk boundaries), and comparing
        # `evictions` against a baseline was red on CORRECT code, because decode growth at
        # engine.py:886 evicts legitimately for the running request and `insert` trims at
        # capacity. Neither is the waiting request's doing.
        # The BLOCKED attempts, i.e. every one that returned False. The final attempt -- the
        # one that succeeds after the first request finishes -- legitimately evicts, because by
        # then the store's blocks ARE reclaimable; measured [0]*26 + [3], and asserting on all
        # attempts would have called that correct eviction a defect.
        blocked = admit_drops[:-1] if not engine._waiting else admit_drops
        assert len(blocked) > 2, f"too few blocked attempts to check: {admit_drops}"
        assert all(d == 0 for d in blocked), (
            f"a blocked admission evicted the store: drops per attempt {admit_drops}")

        out = {}
        for _ in range(600):
            out.update(engine.poll())
            if a in out and b in out:
                break
            engine.step()
    finally:
        engine.shutdown()

    assert a in out, "the first request never finished"
    assert b in out, "the second request was never served; it waited forever or was refused"
    # Not `free_blocks == 64`: the prefix store legitimately keeps both prompts' blocks after
    # they finish, which is the point of the store. What must be zero is what the REQUESTS
    # hold -- `blocks_used` is net of retains (engine.py:447), so it is the leak test here.
    assert engine._blocks_used == 0, f"requests leaked blocks: {engine._blocks_used}"
    assert engine._slots_used == 0, f"requests leaked state slots: {engine._slots_used}"
    # And the store's blocks must still be reclaimable now that nothing is live.
    assert engine._prefix.reclaimable_blocks() == engine._kv.used_blocks, (
        f"{engine._kv.used_blocks} blocks in use but only "
        f"{engine._prefix.reclaimable_blocks()} reclaimable with nothing running")


def test_a_prompt_larger_than_an_empty_pool_still_refuses_at_submit():
    """Waiting is for a prompt that fits LATER. One that never fits must still 400.

    Ruling 4, and the arm that proves moving allocation to the planner did not lose the
    `blocks_for_tokens(total + width - 1) > usable_blocks` refusal: without it such a request
    would sit in `_waiting` until its timeout instead of being told immediately.
    """
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.model import build_random
    from tilerl.testing import RefBackend

    cfg = tiny(max_position_embeddings=8192)
    engine = build_engine(cfg, build_random(cfg, seed=11), RefBackend(), num_blocks=16,
                          num_slots=2, max_batch=1, max_total_tokens=8192)
    with pytest.raises(ValueError, match="KV pool capacity"):
        engine.submit(list(range(1, 600)), SamplingParams(max_new_tokens=8, seed=0))
    assert not engine._waiting, "an impossible prompt was queued instead of refused"


def test_a_failed_admission_returns_every_refcount_it_took():
    """An admission that raises mid-way must leave the store's refcounts exactly as they were.

    The dangerous shape, from the old `submit`: `blocks` seeded with the prefix hit's blocks
    meant an `alloc_slot` failure ran `free_block` over blocks the PrefixStore still held, and
    `free_block` cannot tell that from a release -- the store keeps listing a page that has
    gone back to the free list, and a later request is handed someone else's KV.

    Asserted on the REFCOUNTS, not on "no exception escaped": the unwind can be wrong in both
    directions (freeing too much, or leaking what it took) and only the counts show which.
    """
    import time

    from tilerl.config import tiny
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.model import build_random
    from tilerl.testing import RefBackend

    cfg = tiny(max_position_embeddings=4096)
    engine = build_engine(cfg, build_random(cfg, seed=13), RefBackend(), num_blocks=64,
                          num_slots=4, max_batch=2, max_total_tokens=4096)
    base = [1 + (j % 150) for j in range(160)]
    rid = engine.submit(base, SamplingParams(max_new_tokens=2, seed=0))
    engine.run()
    try:
        for _ in range(600):
            if rid in engine.poll():
                break
            time.sleep(0.02)
        engine.take(rid)
    finally:
        engine.shutdown()
    assert engine.stats()["prefill_forwards"] > 0, "no forward ran; the arm proves nothing"
    assert engine._prefix.stats()["entries"] > 0, "nothing published; there is no hit to adopt"

    before = {b: n for b, n in enumerate(engine._kv.refcount) if n > 0}
    slots_before = engine._states.free_slots

    # Fail AFTER the hit blocks are retained: alloc_block is what runs next.
    calls = {"n": 0}
    real_alloc = engine._kv.alloc_block

    def boom():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("injected: allocation failed mid-admission")
        return real_alloc()

    engine._kv.alloc_block = boom
    longer = base + [200 + (j % 50) for j in range(160)]
    engine.submit(longer, SamplingParams(max_new_tokens=4, seed=0))
    with pytest.raises(RuntimeError, match="injected"):
        engine._admit(engine._waiting[0])
    engine._kv.alloc_block = real_alloc

    after = {b: engine._kv.refcount[b] for b in before}
    assert after == before, f"the unwind did not restore the refcounts: {before} -> {after}"
    assert engine._states.free_slots == slots_before, (
        f"the unwind leaked a state slot: {slots_before} -> {engine._states.free_slots}")
    # And every block it allocated before the failure went back, not just the retained ones.
    assert engine._kv.free_blocks == 64 - sum(1 for n in before.values() if n > 0), (
        f"blocks leaked by the unwind: {engine._kv.free_blocks} free")


def test_every_key_the_store_publishes_reaches_health_or_is_named_as_dropped():
    """`_build_stats` forwards a hand-picked subset, so a new store counter vanishes silently.

    This is asserted by ROUTE, not by name. A name test passes for the wrong reason on
    `hits`/`misses`: `/health` carries `prefix_hits`, but it comes from the engine's own
    `_prefix_hits` (`engine.py:652`, counted per admission), not from the store's counter of
    the same name -- so "the key exists" is true while the store's value goes nowhere.
    """
    from tilerl.config import tiny
    from tilerl.engine import _STORE_STATS_INTERNAL, build_engine
    from tilerl.kv_cache import PrefixStore
    from tilerl.model import build_random
    from tilerl.testing import RefBackend

    cfg = tiny(max_position_embeddings=512)
    engine = build_engine(cfg, build_random(cfg, seed=3), RefBackend(), num_blocks=16,
                          num_slots=2, max_batch=1, max_total_tokens=512)
    assert isinstance(engine._prefix, PrefixStore), "needs a real store, not the null one"

    # Non-trivial first: equal zeros cannot tell a forwarded value from a hardcoded one.
    toks = list(range(1, 33))
    blks = [engine._kv.alloc_block() for _ in range(PagedKvPool.blocks_for_tokens(len(toks)))]
    assert engine._prefix.insert(toks, blks, None), "fixture: insert refused"
    for b in blks:
        engine._kv.free_block(b)
    engine._prefix.lookup(toks)                 # moves the store's lookups_matched
    engine._prefix.clear()                      # moves evictions and blocks_freed
    st = engine._prefix.stats()
    assert st["evictions"] and st["blocks_freed"] and st["lookups_matched"], (
        f"fixture left the counters at zero, so the value check cannot discriminate: {st}")

    published = set(engine._prefix.stats())
    health = engine.stats()
    # By value, not by name: `prefix_hits` exists but carries the ENGINE's counter, so a name
    # check is true for a store key nothing forwards -- and cannot see a hardcoded zero either.
    store_vals = engine._prefix.stats()
    unrouted = []
    for k in sorted(published):
        if k in _STORE_STATS_INTERNAL or k.startswith(("dram_", "ssd_")):
            continue
        wire = f"prefix_{k}"
        if wire not in health or health[wire] != store_vals[k]:
            unrouted.append(f"{k}={store_vals[k]} vs {wire}={health.get(wire, '<absent>')}")
    assert not unrouted, (
        f"the store publishes {unrouted} and /health does not carry the value; add a "
        "prefix_<k> entry in _build_stats or name the key in _STORE_STATS_INTERNAL")

    # The drop list may not name a key the store does not publish: a stale entry there would
    # silence a future key that happens to reuse the name.
    stale = sorted(set(_STORE_STATS_INTERNAL) - published)
    assert not stale, f"_STORE_STATS_INTERNAL names keys the store does not publish: {stale}"


def test_blocks_freed_moves_on_the_wire_when_the_store_frees_a_block():
    """The mutation arm: a name check cannot tell a forwarded key from a hardcoded zero.

    Drives a real eviction through `_drop` and asserts `/health`'s value moves with the
    store's. Without the forwarding line this fails on the KeyError, which is the state the
    #221 merge shipped.
    """
    from tilerl.config import tiny
    from tilerl.engine import build_engine
    from tilerl.model import build_random
    from tilerl.testing import RefBackend

    cfg = tiny(max_position_embeddings=512)
    engine = build_engine(cfg, build_random(cfg, seed=5), RefBackend(), num_blocks=16,
                          num_slots=2, max_batch=1, max_total_tokens=512)
    store = engine._prefix
    tokens = list(range(1, 65))
    blocks = [engine._kv.alloc_block()
              for _ in range(PagedKvPool.blocks_for_tokens(len(tokens)))]
    assert store.insert(tokens, blocks, None), "fixture: insert refused"
    for b in blocks:                       # hand the store sole ownership
        engine._kv.free_block(b)

    # Membership first: a missing key must be REPORTED, not raise a KeyError that reads as a
    # broken test.
    health = engine.stats()
    assert "prefix_blocks_freed" in health, (
        "prefix_blocks_freed is not on the wire; _build_stats is not forwarding it")
    before = health["prefix_blocks_freed"]
    assert before == store.stats()["blocks_freed"], "the wire disagrees with the store"
    store.clear()                          # goes through _drop, which measures the free list
    after_store = store.stats()["blocks_freed"]
    assert after_store > before, f"fixture freed nothing: {before} -> {after_store}"
    assert engine.stats().get("prefix_blocks_freed") == after_store, (
        f"/health says {engine.stats().get('prefix_blocks_freed')}, store says {after_store}: "
        "the counter is not reaching the wire")


#: (n, budget), the open ones wrapped in a strict xfail. `budget` is
#: `max_num_batched_tokens - len(decodes)`, so a decode row sharing the tick lowers it and
#: c14511b's guard -- which gives up one whole block -- stops working: at or below
#: BLOCK_TOKENS there is no block to give up, and a ragged budget shifts every later chunk
#: end so the boundary helper (which takes n alone) predicts a different walk.
#: errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md
_OPEN = pytest.mark.xfail(strict=True, reason="open: the boundary helper does not take the "
                          "budget -- n=961 is 944 at 512 and 448 at 511")
_BOUNDARY_CASES = [
    (65, 512), (129, 512), (513, 512), (1025, 512),
    *(pytest.param(n, b, marks=_OPEN)
      for n, b in ((33, 16), (18, 17), (49, 8), (962, 511), (961, 504))),
]


@pytest.mark.parametrize("n,budget", _BOUNDARY_CASES)
def test_last_prefill_boundary_is_a_real_chunk_end(n, budget):
    """`_last_prefill_boundary(n)` must name a position the planner ends a chunk at.

    When it does not, `last` never fires (engine.py:1045) and NOTHING from that prompt is
    offered to the disk tier -- no error, no counter, the spill just does not happen. The
    512-budget lengths did exactly that: at n=65 the 64-alignment lands on 64, so the 1-token
    tail is a chunk of its own, `short` computes to 0, and the tail back-off cannot run.

    strict=True on the open rows: they go RED the day the budget-dependent half is fixed,
    which a bare `pytest.xfail()` call would not -- that skips unconditionally and can never
    report the fix.
    """
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams, _last_prefill_boundary, build_engine
    from tilerl.model import build_random
    from tilerl.testing import RefBackend

    cfg = tiny(max_position_embeddings=4096)
    engine = build_engine(cfg, build_random(cfg, seed=5), RefBackend(), num_blocks=512,
                          num_slots=2, max_batch=1, max_total_tokens=4096,
                          max_num_batched_tokens=budget)
    engine.submit(list(range(n)), SamplingParams(max_new_tokens=1, seed=0))
    ends, at = [], 0
    while at < n:                          # drive the planner, the only source of chunk ends
        _, prefills, chunks = engine._build_plan()
        assert prefills, f"planner stalled at {at} of {n}"
        assert chunks[0] > 1, f"1-token chunk at {at}: reaches the kernels with 0 blocks"
        at += chunks[0]
        ends.append(at)
        prefills[0].prefill_from = at      # advance without running a forward
    assert at == n, f"chunks summed to {at}, not {n}"
    lb = _last_prefill_boundary(n)
    assert lb in ends, (
        f"n={n} budget={budget}: _last_prefill_boundary says {lb}, but chunks end at "
        f"{ends[-4:]} -- `last` never fires and nothing reaches the disk tier")


# --------------------------------------------------------------- inputs_for


def test_batchkv_inputs_for_covers_every_forward_input():
    """inputs_for clones every tensor the forward reads for a row. The key set
    is the enumeration; the mutation loop proves each source reaches the
    snapshot. A new forward input not added to inputs_for leaves the key set
    assertion as the only thing standing between it and a silent gap."""
    pool = PagedKvPool(4, 2, 8, num_layers=3, device="cpu")
    state = LinearStatePool(2, 2, 2, 4, device="cpu", conv_window=4, conv_dim=16)
    kv = BatchKv(
        block_table=torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]]),
        seq_len=torch.tensor([100, 40]),
        state_slot=torch.tensor([0, 1]),
        kv_pool=pool,
        state_pool=state,
        seq_q_lens=torch.tensor([1, 1]),
    )
    ids = torch.tensor([[5], [7]])
    pos = torch.tensor([[99], [39]])
    snap = kv.inputs_for(ids, pos, 0)
    assert set(snap) == {
        "ids", "pos", "block_table", "seq_len", "state_slot", "seq_q_lens",
        "k", "v", "states", "conv_windows", "win_parity",
    }
    # clones: mutating a return leaves the source untouched
    snap["k"].fill_(999)
    assert not torch.any(pool.k_pool[:, [1, 2]] == 999)
    # coverage: each source the forward reads, when mutated, reaches the snapshot
    pool.k_pool[:, [1, 2]] = 1
    assert torch.any(kv.inputs_for(ids, pos, 0)["k"] == 1)
    pool.v_pool[:, [1, 2]] = 2
    assert torch.any(kv.inputs_for(ids, pos, 0)["v"] == 2)
    state.states[0].fill_(3)
    assert torch.any(kv.inputs_for(ids, pos, 0)["states"] == 3)
    state.conv_windows[0].fill_(4)
    assert torch.any(kv.inputs_for(ids, pos, 0)["conv_windows"] == 4)
    state.win_parity[0] = 1
    assert kv.inputs_for(ids, pos, 0)["win_parity"].item() == 1
    ids[0, 0] = 11
    assert kv.inputs_for(ids, pos, 0)["ids"].item() == 11
    pos[0, 0] = 12
    assert kv.inputs_for(ids, pos, 0)["pos"].item() == 12


def test_batchkv_inputs_for_fp8_scales():
    """An fp8 pool's scales are read by the forward and must be in the snapshot."""
    pool = PagedKvPool(4, 2, 8, num_layers=2, device="cpu", kv_fp8=torch.float8_e4m3fn)
    state = LinearStatePool(1, 1, 1, 4, device="cpu")
    kv = BatchKv(
        block_table=torch.tensor([[1, 0, 0, 0]]),
        seq_len=torch.tensor([20]),
        state_slot=torch.tensor([0]),
        kv_pool=pool,
        state_pool=state,
    )
    snap = kv.inputs_for(torch.tensor([[5]]), torch.tensor([[19]]), 0)
    assert "k_scale" in snap and "v_scale" in snap
    pool.k_scale[:, [1]] = 7
    assert torch.any(
        kv.inputs_for(torch.tensor([[5]]), torch.tensor([[19]]), 0)["k_scale"] == 7
    )
