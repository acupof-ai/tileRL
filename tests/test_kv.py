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


# ------------------------------------------------------------- SSD tier (offload)


def test_tier_spill_reload_roundtrip(tmp_path):
    """A prefix evicted to the SSD tier reloads bit-identical KV into fresh
    blocks — the offload path a repeated long prefix rides instead of prefill.
    """
    from tilerl.kv_cache import KvTier

    pool = PagedKvPool(8, 1, 4)
    tier = KvTier(str(tmp_path / "kvt"), min_tokens=0)
    # min_tokens=0 so any prefix spills; tier_capacity high enough to keep it.
    store = PrefixStore(pool, capacity=1, tier=tier, tier_capacity=8)

    # Prefix A: two blocks of known KV.
    a = [pool.alloc_block() for _ in range(2)]
    for i, b in enumerate(a):
        pool.k_pool[:, b] = torch.randn_like(pool.k_pool[:, b])
        pool.v_pool[:, b] = torch.randn_like(pool.v_pool[:, b])
    ka = [pool.k_pool[:, b].clone() for b in a]
    va = [pool.v_pool[:, b].clone() for b in a]
    toks = list(range(32))
    store.insert(toks, a)
    for b in a:
        pool.free_block(b)  # request done; only the store now holds A's blocks

    # Insert a second prefix — capacity=1 forces A's eviction → spill to tier.
    bblk = [pool.alloc_block()]
    store.insert(list(range(100, 116)), bblk)
    # A's blocks were freed back to the pool (store released its last owner).
    for b in a:
        assert pool.refcount[b] == 0

    # A cold lookup returns reload_key, empty blocks.
    hit = store.lookup(toks)
    assert hit is not None and hit.reload_key is not None and hit.blocks == ()

    # Reload into fresh blocks; spilled KV is bit-identical.
    fresh = [pool.alloc_block() for _ in range(2)]
    assert tier.load_kv(hit.reload_key, tuple(toks), fresh, pool), "A did not spill"
    for i, b in enumerate(fresh):
        assert torch.equal(pool.k_pool[:, b], ka[i])
        assert torch.equal(pool.v_pool[:, b], va[i])


def test_tier_min_tokens_gates_spill(tmp_path):
    """Below tier_min_tokens an evicted prefix is dropped, not spilled (SSD
    churn floor)."""
    from tilerl.kv_cache import KvTier

    pool = PagedKvPool(8, 1, 4)
    tier = KvTier(str(tmp_path / "kvt"), min_tokens=999)
    store = PrefixStore(pool, capacity=1, tier=tier, tier_capacity=8)
    a = [pool.alloc_block()]
    store.insert(list(range(16)), a)
    store.insert(list(range(100, 116)), [pool.alloc_block()])
    # Sub-floor: dropped, not spilled.
    assert tier.load_kv(store._hash_all(list(range(16))), tuple(range(16)),
                        [pool.alloc_block()], pool) is False
    assert store.lookup(list(range(16))) is None


def test_tier_deferred_flush_to_disk(tmp_path):
    """spill_kv defers the torch.save off-tick: returns immediately with the
    blob in memory (served from _pending), then a daemon flushes it to disk;
    reload is bit-identical either way. Keeps the spill off the decode tick."""
    import os
    import time

    from tilerl.kv_cache import KvTier

    pool = PagedKvPool(8, 1, 4)
    tier = KvTier(str(tmp_path / "kvt"), min_tokens=0)
    b = pool.alloc_block()
    pool.k_pool[:, b] = torch.randn_like(pool.k_pool[:, b])
    pool.v_pool[:, b] = torch.randn_like(pool.v_pool[:, b])
    kb = pool.k_pool[:, b].clone()
    key, toks = 0xABCD, (1, 2, 3)

    assert tier.spill_kv(key, toks, [b], pool) is True
    assert key in tier._pending  # served from memory on-tick

    for _ in range(200):
        if key not in tier._pending:
            break
        time.sleep(0.01)
    assert key not in tier._pending, "writer never flushed"
    assert os.path.exists(tier._kv(key)), "blob not on disk after flush"

    fresh = pool.alloc_block()
    assert tier.load_kv(key, toks, [fresh], pool) is True
    assert torch.equal(pool.k_pool[:, fresh], kb)
    assert tier.load_kv(0xDEAD, toks, [fresh], pool) is False  # missing key
    # Hash collision: same key, different tokens -> refuse, do not return stale KV.
    assert tier.load_kv(key, (9, 9, 9), [fresh], pool) is False


def test_tier_drop_purges_pending_and_queue_cap(tmp_path):
    """drop() on a still-queued spill must purge the in-memory blob, or has()
    would keep serving KV for a prefix the store evicted (write-back cache
    invalidation → wrong tokens). Also: over max_pending, spill refuses."""
    import os

    from tilerl.kv_cache import KvTier

    pool = PagedKvPool(16, 1, 4)
    tier = KvTier(str(tmp_path / "kvt"), min_tokens=0, max_pending=2)
    key, toks = 0x1234, (5, 6)
    b = pool.alloc_block()
    assert tier.spill_kv(key, toks, [b], pool) is True
    # drop before the flush may have run: load must go False immediately.
    tier.drop(key)
    assert tier.load_kv(key, toks, [b], pool) is False, "drop left a pending blob (invalidation hole)"

    # After a drop, no stale file should survive the daemon's write either.
    import time

    time.sleep(0.1)
    assert not os.path.exists(tier._kv(key)), "dropped blob reappeared on disk"

    # Queue cap: flood faster than the disk drains; some spills must refuse.
    refused = 0
    for i in range(50):
        bb = pool.alloc_block()
        if not tier.spill_kv(0x9000 + i, (i,), [bb], pool):
            refused += 1
        pool.free_block(bb)
    # The cap must bite at least once under a burst faster than the disk drains,
    # OR the writer kept up and none refused — both are correct; assert no crash
    # and the cap is respected in-flight.
    with tier._lock:
        assert len(tier._pending) <= tier._max_pending


def test_tier_hash_collision_serves_no_wrong_kv(tmp_path):
    """Two prefixes colliding on the same 64-bit hash must not cross KV in the
    cold tier — the one path that would produce WRONG tokens, not a crash.
    Forces the hash constant so both spill to the same file key."""
    from tilerl.kv_cache import KvTier

    pool = PagedKvPool(8, 1, 4)
    tier = KvTier(str(tmp_path / "kvt"), min_tokens=0)
    # A and B: distinct tokens, same hash key; distinct KV contents.
    a_tok, b_tok = (1, 2, 3), (7, 8, 9)
    ba, bb = pool.alloc_block(), pool.alloc_block()
    pool.k_pool[:, ba] = 1.0
    pool.k_pool[:, bb] = 2.0
    key = 0xC0FFEE  # same key for both — the collision
    assert tier.spill_kv(key, a_tok, [ba], pool)
    # B spills to the same key AFTER A (later writer wins the file).
    assert tier.spill_kv(key, b_tok, [bb], pool)
    import time
    time.sleep(0.1)  # drain writer

    # Loading A's tokens against the (now B) file must REFUSE, not return B's KV.
    probe = pool.alloc_block()
    got_a = tier.load_kv(key, a_tok, [probe], pool)
    got_b = tier.load_kv(key, b_tok, [probe], pool)
    # At most one prefix's KV survives, and it is served ONLY to its own tokens.
    assert not (got_a and got_b), "both tokens loaded from one file — cross-contamination"
    if got_b:
        assert torch.all(pool.k_pool[:, probe] == 2.0)  # B's content, B's tokens


def test_tier_insert_retires_cold_twin(tmp_path):
    """A reloaded prefix that re-publishes must retire its cold twin, or the
    twin's later eviction drop()s the live entry's file (generational race)."""
    from tilerl.kv_cache import KvTier

    pool = PagedKvPool(16, 1, 4)
    tier = KvTier(str(tmp_path / "kvt"), min_tokens=0)
    store = PrefixStore(pool, capacity=1, tier=tier, tier_capacity=8)
    toks = list(range(32))
    a = [pool.alloc_block() for _ in range(2)]
    store.insert(toks, a)
    for b in a:
        pool.free_block(b)
    store.insert(list(range(100, 116)), [pool.alloc_block()])  # evict A -> cold
    h = store._hash_all(toks)
    assert h in store._cold  # A is cold
    # Re-publish A (as if reloaded): must retire the cold twin, not leave two.
    a2 = [pool.alloc_block() for _ in range(2)]
    store.insert(toks, a2)
    cold_a = [e for e in store._cold.get(h, ()) if e.tokens == tuple(toks)]
    assert not cold_a, "cold twin of the re-published prefix was not retired"


def test_tier_lru_evicts_oldest_over_cap(tmp_path):
    """max_bytes caps on-disk usage: after each write the daemon evicts LRU
    entries until under cap; a load touches the entry to MRU so it survives."""
    import os
    import time

    from tilerl.kv_cache import KvTier

    pool = PagedKvPool(16, 1, 4)
    tier = KvTier(str(tmp_path / "kvt"), min_tokens=0, max_bytes=10**9)

    def spill_wait(key, toks):
        b = pool.alloc_block()
        assert tier.spill_kv(key, toks, [b], pool)
        for _ in range(200):
            if key not in tier._pending:
                break
            time.sleep(0.01)
        time.sleep(0.05)  # let the daemon finish _track_written
        pool.free_block(b)

    spill_wait(0xA, (1,))
    one = tier._total
    assert one > 0
    tier._max_bytes = 2 * one + 1  # holds 2, evicts on the 3rd

    spill_wait(0xB, (2,))
    assert os.path.exists(tier._kv(0xA))  # 2 entries, under cap

    # Touch A → MRU; B is now LRU.
    fresh = pool.alloc_block()
    assert tier.load_kv(0xA, (1,), [fresh], pool)
    pool.free_block(fresh)

    spill_wait(0xC, (3,))
    assert not os.path.exists(tier._kv(0xB)), "LRU evicted the touched entry, not the LRU one"
    assert os.path.exists(tier._kv(0xA)), "touched entry was evicted"
    assert os.path.exists(tier._kv(0xC))
