"""Hermetic CPU tests for tilerl.kv_cache.

Covers: roundtrip, prefix hit/miss incl. hash-collision, refcount/CoW,
state pool, pool exhaustion, and the shared-prefix fork lifecycle.
"""

import pytest
import torch

from tilerl.kv_cache import (
    BLOCK_TOKENS,
    LinearStatePool,
    PagedKvPool,
    PrefixStore,
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
    assert s["entries"] == 3 and s["hits"] == 3 and s["misses"] == 1


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
    assert engine.stats()["prefix_hits"] >= 1, "no prefix hit; the test is inert"

    before = {b: n for b, n in enumerate(engine._kv.refcount) if n > 0}
    with pytest.raises(RuntimeError):
        engine.submit(longer, SamplingParams(max_new_tokens=1, seed=0))
    after = {b: engine._kv.refcount[b] for b in before}
    assert after == before, f"the rejected submit released blocks: {before} -> {after}"

