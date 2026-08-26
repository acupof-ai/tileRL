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
    from tilerl.ops.backend import get_backend

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


# -------------------------------------------------------------- refcount / CoW


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


def test_cow_duplicates_shared_block():
    pool = PagedKvPool(4, 1, 4)
    b = pool.alloc_block()
    k, v = _kv(0, 6)
    pool.write_block(b, 0, k, v)
    pool.retain(b)  # second owner (e.g. prefix store) -> shared
    assert pool.is_shared(b)
    # writing a shared block is a hard error
    with pytest.raises(RuntimeError):
        pool.write_block(b, 6, *_kv(1, 1))
    nb = pool.cow_for_append(b)
    assert nb != b
    assert pool.refcount[b] == 1  # caller's ownership moved off it
    assert pool.refcount[nb] == 1  # exclusive copy
    # contents copied across every layer
    assert torch.equal(pool.k_pool[0, nb, :, :6], k)
    assert torch.equal(pool.v_pool[0, nb, :, :6], v)
    # mutating the copy leaves the shared original byte-stable
    k2, v2 = _kv(2, 1)
    pool.write_block(nb, 6, k2, v2)
    assert torch.equal(pool.k_pool[0, b, :, :6], k)
    assert torch.equal(pool.k_pool[0, nb, :, 6:7], k2)


def test_cow_exclusive_block_is_noop():
    pool = PagedKvPool(2, 1, 4)
    b = pool.alloc_block()
    assert pool.cow_for_append(b) == b
    assert pool.free_blocks == 1


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


# ----------------------------------------------------------------- exhaustion


def test_alloc_raises_under_pressure():
    pool = PagedKvPool(4, 1, 4)
    for _ in range(4):
        pool.alloc_block()
    with pytest.raises(RuntimeError):
        pool.alloc_block()


# ------------------------------------------------- shared-prefix fork lifecycle


def test_shared_prefix_fork_cow():
    """Two sequences share a cached prefix, then fork: each append CoWs."""
    pool = PagedKvPool(8, 1, 4)
    store = PrefixStore(pool)

    # Sequence A: 20-token prompt in [a0 (full), a1 (4 tokens)].
    toks = list(range(20))
    a0 = pool.alloc_block()
    a1 = pool.alloc_block()
    blocks_a = [a0, a1]
    k0, v0 = _kv(0, BLOCK_TOKENS)
    k1, v1 = _kv(1, 4)
    pool.write_block(a0, 0, k0, v0)
    pool.write_block(a1, 0, k1, v1)
    store.insert(toks, blocks_a)  # store co-owns both
    assert pool.refcount[a0] == 2 and pool.refcount[a1] == 2

    # Sequence B: identical prompt -> prefix hit, adopts the same blocks.
    hit = store.lookup(toks)
    assert hit is not None and hit.length == 20 and hit.blocks == (a0, a1)
    blocks_b = list(hit.blocks)
    for b in hit.blocks:
        pool.retain(b)
    assert pool.refcount[a1] == 3  # A + store + B

    # B appends one token: a1 is shared AND partial -> CoW.
    assert pool.is_shared(blocks_b[-1])
    b1 = pool.cow_for_append(blocks_b[-1])
    blocks_b[-1] = b1
    kb, vb = _kv(2, 1)
    pool.write_block(b1, 4, kb, vb)
    assert pool.refcount[a1] == 2  # B's ref moved to b1
    assert pool.refcount[b1] == 1

    # A appends too: a1 is still shared (A + store) -> A CoWs as well.
    assert pool.is_shared(blocks_a[-1])
    a1n = pool.cow_for_append(blocks_a[-1])
    blocks_a[-1] = a1n
    ka, va = _kv(3, 1)
    pool.write_block(a1n, 4, ka, va)
    assert pool.refcount[a1] == 1  # store only now

    # The store's cached prefix is byte-stable; each fork reads its own token.
    assert torch.equal(pool.k_pool[0, a1, :, :4], k1)
    assert torch.equal(pool.v_pool[0, a1, :, :4], v1)
    assert torch.equal(pool.k_pool[0, b1, :, 4:5], kb)
    assert torch.equal(pool.k_pool[0, a1n, :, 4:5], ka)

    # Cleanup: B releases its fork plus its ref on the shared head block;
    # A releases its fork and head. The store keeps the prefix until evicted.
    pool.free_block(b1)
    pool.free_block(a0)  # B's ref on the shared head
    assert pool.refcount[b1] == 0  # recycled
    assert pool.refcount[a0] == 2  # A + store
    pool.free_block(a1n)
    pool.free_block(a0)
    assert pool.refcount[a0] == 1 and pool.refcount[a1] == 1  # store only
    assert pool.free_blocks == 6  # 8 total - 2 store-held
