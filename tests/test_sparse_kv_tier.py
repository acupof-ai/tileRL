"""Unit C: sparse-KV page tiering. A selected page lives on the device; the rest
move to a pinned host tier (HostKvPages) on PagedKvPool.demote_page/promote_page,
through the same path DramSnapshots uses for state. A demoted page frees its
device block back to the SAME pool; promotion allocates a fresh block.

Two end-to-end properties, selector-agnostic:
1. a demoted page promoted back reads byte-equal across EVERY plane, fp8 scale
   planes included;
2. decode attention tokens are identical with and without demoting every
   unselected page, after a demote -> promote round trip moves block ids.
"""

from __future__ import annotations

import os

import torch

from tilerl.kv_cache import BLOCK_TOKENS, HostKvPages, PagedKvPool
from tilerl.testing import RefBackend


#: cpu by default (the hermetic gate); TILERL_TARGET=cuda puts the pools on the
#: card so demote/promote exercise the real pinned D2H/H2D copies. RefBackend's
#: torch paged_attention runs on whatever device the pools are on.
def _device() -> torch.device:
    return torch.device("cuda") if os.environ.get("TILERL_TARGET") == "cuda" else torch.device("cpu")


def _kv(seed: int, p: int, hkv: int, d: int):
    torch.manual_seed(seed)
    return (torch.randn(p, hkv, BLOCK_TOKENS, d, dtype=torch.bfloat16),
            torch.randn(p, hkv, BLOCK_TOKENS, d, dtype=torch.bfloat16))


def test_a_demoted_page_promotes_byte_equal_across_every_plane():
    """bf16 pool, two layers: the K and V of BOTH planes must come back identical."""
    p, hkv, d = 4, 2, 8
    pool = PagedKvPool(p + 1, hkv, d, num_layers=2, device=_device())
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    kk, vv = _kv(0, p, hkv, d)
    b = pool.alloc_block()
    for plane in range(2):
        pool.write_block(b, 0, kk[0], vv[0], layer=plane)
    before = (pool.k_pool[:, b].clone(), pool.v_pool[:, b].clone())
    loc = pool.page_location(b)
    assert loc == "device", loc

    n = pool.demote_page(b)
    assert pool.page_location(b) == "host"
    assert b in pool._free, "demotion must return the block to the same pool"
    assert n == sum(t.numel() * t.element_size() for t in before), n

    # take the freed frame for something else, so promotion must allocate elsewhere
    other = pool.alloc_block()
    assert other == b, "the freed frame is reclaimable; occupying it forces a new one"
    nb = pool.promote_page(b)
    assert nb != b, "promotion must allocate a fresh block, not the occupied frame"
    assert pool.page_location(nb) == "device"
    assert b not in pool.cold, "the host copy is released on promote"
    assert torch.equal(pool.k_pool[:, nb], before[0]), "K planes differ after round trip"
    assert torch.equal(pool.v_pool[:, nb], before[1]), "V planes differ after round trip"


def test_fp8_scale_planes_round_trip_with_the_page():
    """The fp8 K/V are 1 B/value; without their f32 per-token scale planes the
    promoted page reloads plausible garbage. All four tensors must be byte-equal."""
    p, hkv, d = 4, 2, 16
    pool = PagedKvPool(p + 1, hkv, d, num_layers=2, device=_device(),
                       kv_fp8=torch.float8_e4m3fn)
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    kk, vv = _kv(1, p, hkv, d)
    b = pool.alloc_block()
    for plane in range(2):
        pool.write_block(b, 0, kk[0], vv[0], layer=plane)
    snap = (pool.k_pool[:, b].clone(), pool.v_pool[:, b].clone(),
            pool.k_scale[:, b].clone(), pool.v_scale[:, b].clone())

    pool.demote_page(b)
    nb = pool.promote_page(b)
    assert torch.equal(pool.k_pool[:, nb], snap[0])
    assert torch.equal(pool.v_pool[:, nb], snap[1])
    assert torch.equal(pool.k_scale[:, nb], snap[2]), "k_scale plane lost"
    assert torch.equal(pool.v_scale[:, nb], snap[3]), "v_scale plane lost"


def test_narrow_cold_f16_round_trips_exactly_for_f16_values():
    """sm70's f32 pool narrows a demoted page's K/V to f16 on the host (the host has
    half the room); values that ARE f16-representable must return byte-identical, and
    the held blob is half the f32 size. The fp8 scale planes stay f32."""
    p, hkv, d = 4, 2, 8
    pool = PagedKvPool(p + 1, hkv, d, num_layers=2, device=_device(),
                       dtype=torch.float32, cold_dtype=torch.float16)
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    # powers of two and small integers are exact in f16
    k = torch.arange(p * hkv * BLOCK_TOKENS * d, dtype=torch.float32).reshape(
        p, hkv, BLOCK_TOKENS, d) * 0.25
    v = -torch.arange(p * hkv * BLOCK_TOKENS * d, dtype=torch.float32).reshape(
        p, hkv, BLOCK_TOKENS, d) * 0.25
    b = pool.alloc_block()
    for plane in range(2):
        pool.write_block(b, 0, k[0].float(), v[0].float(), layer=plane)
    before = (pool.k_pool[:, b].clone(), pool.v_pool[:, b].clone())

    n = pool.demote_page(b)
    blob = pool.cold.take(b)
    # re-hold it so promote_page has a blob
    assert pool.cold.hold(b, blob, n)
    assert blob["k"].dtype == torch.float16 and blob["v"].dtype == torch.float16
    f32_bytes = before[0].numel() * 4 + before[1].numel() * 4
    held_bytes = blob["k"].numel() * 2 + blob["v"].numel() * 2
    assert held_bytes * 2 == f32_bytes, (held_bytes, f32_bytes)
    assert n == held_bytes

    nb = pool.promote_page(b)
    assert torch.equal(pool.k_pool[:, nb], before[0]), "f16-exact K values changed"
    assert torch.equal(pool.v_pool[:, nb], before[1]), "f16-exact V values changed"


def test_a_prefix_shared_page_cannot_be_demoted():
    """A page retained by the prefix store is read-only wherever it lives; moving
    its device frame away would corrupt every other request sharing it."""
    pool = PagedKvPool(4, 2, 8, num_layers=1, device=_device())
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    b = pool.alloc_block()
    pool.retain(b)  # a second owner (prefix store)
    kk, vv = _kv(2, 4, 2, 8)
    pool.write_block(b, 0, kk[0], vv[0])
    try:
        pool.demote_page(b)
        raise AssertionError("a shared page was demoted")
    except RuntimeError:
        pass
    assert pool.page_location(b) == "device"
    pool.free_block(b)


def test_demoting_every_unselected_page_leaves_decode_identical():
    """Decode over a context: select only one page, demote the rest, promote the
    named one back (its block id changes), then run the REAL paged_attention over
    the rebuilt full table. The output and the sampled token must equal a dense run
    that never demoted. Drives the actual selector-independent mechanism."""
    torch.manual_seed(5)
    p, hkv, d = 4, 2, 16
    hq = 4
    pool = PagedKvPool(p + 5, hkv, d, num_layers=1, device=_device())
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    blocks = [pool.alloc_block() for _ in range(p)]
    pages = [_kv(10 + i, 1, hkv, d) for i in blocks]
    for i, b in enumerate(blocks):
        kk, vv = pages[i]
        pool.write_block(b, 0, kk[0], vv[0])
    dev = _device()
    q = torch.randn(1, 1, hq, d, device=dev)
    scale = 1.0 / d ** 0.5
    seq_len = torch.tensor([p * BLOCK_TOKENS], device=dev)

    def run(table):
        return RefBackend().paged_attention(
            q, pool.k_pool[0], pool.v_pool[0], table.to(dev), seq_len, scale)

    dense_table = torch.tensor([blocks], device=dev)
    y_dense = run(dense_table)

    # selector picks page 2 only; demote the other three (their device frames leave)
    keep = blocks[2]
    for i, b in enumerate(blocks):
        if b != keep:
            pool.demote_page(b)
    assert pool.page_location(keep) == "device"
    assert all(pool.page_location(b) == "host" for b in blocks if b != keep)

    # a later selector names the whole context again: promote each demoted page to a
    # fresh block at its original logical position
    rebuilt = [0] * p
    for i, b in enumerate(blocks):
        rebuilt[i] = keep if b == keep else pool.promote_page(b)
    y_round = run(torch.tensor([rebuilt]))

    assert torch.equal(y_round, y_dense), "decode output changed across tiering"
    assert torch.equal(y_round.argmax(-1), y_dense.argmax(-1)), "sampled token changed"
    # the round trip really moved ids: the rebuilt table is not the dense one
    assert rebuilt != blocks


def _drain(engine, rid, n):
    done = {}
    for _ in range(512):
        done.update(engine.poll())
        if rid in done and len(done[rid]) >= n:
            return done[rid][:n]
        engine.step()
    raise TimeoutError("engine did not finish in 512 ticks")


def test_engine_decode_tokens_equal_across_a_full_demote_promote_round_trip():
    """The gate the design names for the live engine: decode tokens on the tiny
    model are identical with and without tiering. A dense engine decodes a
    prompt; an identical engine, after prefill, moves EVERY private page to the
    host (keep empty) and then selects the whole context again (every cold page
    promoted to a fresh block at its logical index) before decoding. The rebuild
    must be lossless even though the device block ids all changed.

    CPU-only by construction: it builds through RefBackend, the model-forward
    parity cell. The pinned D2H/H2D transfer path on a real card is the four pool
    gates above run with TILERL_TARGET=cuda."""
    if _device().type != "cpu":
        import pytest

        pytest.skip("model-forward parity gate is the CPU cell; card runs the pool gates")
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.model import build_random

    def engine(cold_bytes, cold_format="native"):
        cfg = tiny()
        return build_engine(
            cfg, build_random(cfg, seed=11), RefBackend(), num_blocks=64,
            num_slots=4, max_batch=1, max_total_tokens=2048,
            prefix_store=NoPrefixStore(), kv_cold_bytes=cold_bytes,
            cold_format=cold_format), cfg

    import numpy as np

    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)  # 6 pages
    params = SamplingParams(temperature=0.0, max_new_tokens=6, seed=0)

    dense, cfg = engine(0)
    rid_d = dense.submit(prompt, params)
    tok_d = _drain(dense, rid_d, 6)

    cold, _ = engine(1 << 30)
    rid = cold.submit(prompt, params)
    # step until this request reaches decode (prefill finished, no token drained)
    req = None
    for _ in range(64):
        for r in cold._running:
            if r.req_id == rid and r.phase == 2:
                req = r
        if req is not None:
            break
        cold.step()
    live = list(req.blocks)
    assert len(live) == 6, live
    # selector selects NOTHING: all private pages demote (pool freed)
    d, p = cold.sparse_retier(frozenset())
    assert d == 6 and p == 0 and not req.blocks
    old = {b for _, b in req.cold_pages}
    assert old == set(live)
    # while demoted, stats memory shows the moved bytes as one host kv_cold allocation
    mid = cold.stats()
    rows = [r for r in mid["memory"] if r["owner"] == "kv_cold"]
    assert len(rows) == 1 and rows[0]["tier"] == "host" and rows[0]["kind"] == "allocation"
    assert rows[0]["derived"] == mid["kv_cold_bytes"] > 0
    # selector selects the whole context: every cold page fetches back once
    d, p = cold.sparse_retier(frozenset(old))
    assert p == 6 and d == 0 and len(req.blocks) == 6
    # ids may be reused (LIFO) or fresh; what matters is the rebuilt context is lossless
    assert req.cold_pages == []
    tok_c = _drain(cold, rid, 6)

    assert tok_c == tok_d, f"tiered decode {tok_c} != dense {tok_d}"
    st = cold.stats()
    assert st["kv_cold_pages"] == 0 and st["kv_cold_promotions"] == 6
    mem = [r for r in st["memory"] if r["owner"] == "kv_cold"]
    assert not mem, "a fully promoted tier leaves no host row"
    dense.shutdown(); cold.shutdown()


def test_narrow_f16_path_prices_half_and_decodes_like_dense():
    """The sm70 path end to end on the f32 CPU cell: an engine whose cold tier stores
    f16 (cold_format='f16') demotes every unselected page through the narrow D2H cast,
    promotes it, and decodes. The held kv_cold row is priced at the f16 width and its
    derived bytes equal the measured pinned bytes; greedy tokens equal the dense
    continuation on this model (f16 holds the attention values to greedy precision).

    CPU-only for the same reason as the native engine gate; the card runs the pool's
    pinned f16 cast gate above."""
    if _device().type != "cpu":
        import pytest

        pytest.skip("model-forward parity gate is the CPU cell; card runs the pool gates")
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.memory import per_cold_kv_block_bytes
    from tilerl.model import build_random

    cfg = tiny()
    dense = build_engine(cfg, build_random(cfg, seed=11), RefBackend(), num_blocks=64,
                         num_slots=4, max_batch=1, max_total_tokens=2048,
                         prefix_store=NoPrefixStore())
    cold = build_engine(cfg, build_random(cfg, seed=11), RefBackend(), num_blocks=64,
                        num_slots=4, max_batch=1, max_total_tokens=2048,
                        prefix_store=NoPrefixStore(), kv_cold_bytes=1 << 30,
                        cold_format="f16")
    import numpy as np

    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)
    params = SamplingParams(temperature=0.0, max_new_tokens=6, seed=0)
    tok_d = _drain(dense, dense.submit(prompt, params), 6)

    rid = cold.submit(prompt, params)
    req = None
    for _ in range(64):
        for r in cold._running:
            if r.req_id == rid and r.phase == 2:
                req = r
        if req is not None:
            break
        cold.step()
    d, p = cold.sparse_retier(frozenset())
    assert d == 6 and p == 0
    # the held row is priced at the f16 cold width and matches the bytes actually held
    mid = cold.stats()
    row = [r for r in mid["memory"] if r["owner"] == "kv_cold"][0]
    expected = 6 * per_cold_kv_block_bytes(cfg, torch.float32, None, torch.float16)
    assert row["derived"] == expected == row["measured"], (row, expected)
    cold_pages = {b for _, b in req.cold_pages}
    cold.sparse_retier(frozenset(cold_pages))
    tok_c = _drain(cold, rid, 6)
    assert tok_c == tok_d, f"f16-narrow decode {tok_c} != dense {tok_d}"
    dense.shutdown(); cold.shutdown()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
