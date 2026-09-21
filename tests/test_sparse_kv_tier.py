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

import pytest
import torch

from tilerl.kv_cache import BLOCK_TOKENS, PagedKvPool
from tilerl.kv_tiers import HostKvPages
from tilerl.testing import RefBackend


#: cpu by default (the hermetic gate); TILERL_TARGET=cuda puts the pools on the
#: card so demote/promote exercise the real pinned D2H/H2D copies. RefBackend's
#: torch paged_attention runs on whatever device the pools are on.
def _device() -> torch.device:
    return (
        torch.device("cuda") if os.environ.get("TILERL_TARGET") == "cuda" else torch.device("cpu")
    )


def _kv(seed: int, p: int, hkv: int, d: int):
    torch.manual_seed(seed)
    return (
        torch.randn(p, hkv, BLOCK_TOKENS, d, dtype=torch.bfloat16),
        torch.randn(p, hkv, BLOCK_TOKENS, d, dtype=torch.bfloat16),
    )


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
    pool = PagedKvPool(p + 1, hkv, d, num_layers=2, device=_device(), kv_fp8=torch.float8_e4m3fn)
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    kk, vv = _kv(1, p, hkv, d)
    b = pool.alloc_block()
    for plane in range(2):
        pool.write_block(b, 0, kk[0], vv[0], layer=plane)
    snap = (
        pool.k_pool[:, b].clone(),
        pool.v_pool[:, b].clone(),
        pool.k_scale[:, b].clone(),
        pool.v_scale[:, b].clone(),
    )

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
    pool = PagedKvPool(
        p + 1, hkv, d, num_layers=2, device=_device(), dtype=torch.float32, cold_dtype=torch.float16
    )
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    # powers of two and small integers are exact in f16
    k = (
        torch.arange(p * hkv * BLOCK_TOKENS * d, dtype=torch.float32).reshape(
            p, hkv, BLOCK_TOKENS, d
        )
        * 0.25
    )
    v = (
        -torch.arange(p * hkv * BLOCK_TOKENS * d, dtype=torch.float32).reshape(
            p, hkv, BLOCK_TOKENS, d
        )
        * 0.25
    )
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


def test_cold_byte_row_matches_held_blob_for_every_width():
    """per_cold_kv_block_bytes must equal the bytes demote actually holds, for every
    width. The fp8 case is the one that was wrong: kv_format already prices the
    per-token f32 scales, so adding a scale term double-counted them (1792 derived
    vs 1280 held)."""
    from tilerl.config import tiny
    from tilerl.memory import per_cold_kv_block_bytes

    cfg = tiny()
    H, D = cfg.num_kv_heads, cfg.head_dim
    L = len(cfg.full_attn_layers)

    def held(fp8, cold):
        pool = PagedKvPool(
            8,
            H,
            D,
            num_layers=L,
            device=_device(),
            dtype=torch.float32,
            kv_fp8=fp8,
            cold_dtype=cold,
        )
        pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
        b = pool.alloc_block()
        return pool.demote_page(b)

    assert per_cold_kv_block_bytes(cfg, torch.float32, None, None) == held(None, None)
    assert per_cold_kv_block_bytes(cfg, torch.float32, None, torch.float16) == held(
        None, torch.float16
    )
    assert per_cold_kv_block_bytes(cfg, torch.float32, torch.float8_e4m3fn, None) == held(
        torch.float8_e4m3fn, None
    )


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
    scale = 1.0 / d**0.5
    seq_len = torch.tensor([p * BLOCK_TOKENS], device=dev)

    def run(table):
        return RefBackend().paged_attention(
            q, pool.k_pool[0], pool.v_pool[0], table.to(dev), seq_len, scale
        )

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
    from tilerl.build import build_engine
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.model import build_random

    def engine(cold_bytes, cold_format="native"):
        cfg = tiny()
        return build_engine(
            cfg,
            build_random(cfg, seed=11),
            RefBackend(),
            num_blocks=64,
            num_slots=4,
            max_batch=1,
            max_total_tokens=2048,
            prefix_store=NoPrefixStore(),
            sparse_k=0,
            kv_cold_bytes=cold_bytes,
            cold_format=cold_format,
        ), cfg

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
    dense.shutdown()
    cold.shutdown()


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
    from tilerl.build import build_engine
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.memory import per_cold_kv_block_bytes
    from tilerl.model import build_random

    cfg = tiny()
    dense = build_engine(
        cfg,
        build_random(cfg, seed=11),
        RefBackend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=2048,
        prefix_store=NoPrefixStore(),
        sparse_k=0,
    )
    cold = build_engine(
        cfg,
        build_random(cfg, seed=11),
        RefBackend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=2048,
        prefix_store=NoPrefixStore(),
        sparse_k=0,
        kv_cold_bytes=1 << 30,
        cold_format="f16",
    )
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
    dense.shutdown()
    cold.shutdown()


def test_pages_past_the_host_budget_spill_to_ssd_and_promote_byte_equal(tmp_path):
    """Cold-tier SSD spill: a host budget that holds only two pages pushes the
    third demotion to one mmap'd file (opaque key -> file slot, no on-disk
    index). Every spilled page promotes back through the SAME demote/promote
    seam byte-equal across all planes, and the tier reports host and SSD bytes
    separately."""
    p, hkv, d, layers = 5, 2, 8, 2
    pool = PagedKvPool(p + 1, hkv, d, num_layers=layers, device=_device())
    per = (
        sum(pool.k_pool[0, 0].numel() * pool.k_pool.element_size() for _ in (0,)) * layers
    )  # K planes only; the tier also holds V below
    per = (
        pool.k_pool[0, 0].numel() * pool.k_pool.element_size()
        + pool.v_pool[0, 0].numel() * pool.v_pool.element_size()
    ) * layers
    ssd = str(tmp_path / "cold_spill.bin")
    pool.attach_cold(HostKvPages(budget_bytes=per * 2, ssd_path=ssd))
    blocks = [pool.alloc_block() for _ in range(4)]  # hold all so frames are distinct
    pages = {}
    for b in blocks:
        kk, vv = _kv(b, p, hkv, d)
        for plane in range(layers):
            pool.write_block(b, 0, kk[0], vv[0], layer=plane)
        pages[b] = (pool.k_pool[:, b].clone(), pool.v_pool[:, b].clone())
        pool.demote_page(b)
    st = pool.cold.stats()
    assert st["kv_cold_pages"] == 2 and st["kv_cold_ssd_pages"] == 2, st
    assert pool.cold.bytes_held == per * 2 and pool.cold.ssd_bytes == per * 2, st
    # each live key owns one stride-sized logical slot; the file is grown one
    # extent at a time, so its size is the extent-rounded capacity, not n*stride
    assert len(pool.cold._ssd) == 2
    assert os.path.getsize(ssd) == pool.cold._ssd.HEADER + per * pool.cold._ssd._cap
    assert pool.cold._ssd._cap >= 2 and pool.cold._ssd._cap < 2 + pool.cold._ssd.GROWTH_SLOTS
    for b in blocks:
        nb = pool.promote_page(b)
        assert torch.equal(pool.k_pool[:, nb], pages[b][0]), f"K page {b}"
        assert torch.equal(pool.v_pool[:, nb], pages[b][1]), f"V page {b}"
    assert pool.cold.bytes_held == 0 and pool.cold.ssd_bytes == 0


def test_ssd_capacity_counts_toward_admission(tmp_path):
    """A request whose pages exceed the pinned host budget but fit within
    host+SSD must be admitted; cold_capacity_blocks adds the countable SSD
    budget. Without the SSD term admission refuses a context SSD spill serves.
    Explicit ssd_capacity_bytes keeps this independent of free disk space."""
    p, hkv, d, layers = 5, 2, 8, 2
    pool = PagedKvPool(p + 1, hkv, d, num_layers=layers, device=_device())
    per = (
        pool.k_pool[0, 0].numel() * pool.k_pool.element_size()
        + pool.v_pool[0, 0].numel() * pool.v_pool.element_size()
    ) * layers
    ssd = str(tmp_path / "cold_cap.bin")
    # host holds 2 pages; SSD budget holds another 8
    pool.attach_cold(HostKvPages(budget_bytes=per * 2, ssd_path=ssd, ssd_capacity_bytes=per * 8))
    assert pool.cold_capacity_blocks() == 10
    # spill actually reaches the file: 3 pages past host land on SSD
    blocks = [pool.alloc_block() for _ in range(3)]
    for b in blocks:
        kk, vv = _kv(b, p, hkv, d)
        for plane in range(layers):
            pool.write_block(b, 0, kk[0], vv[0], layer=plane)
        pool.demote_page(b)
    assert pool.cold.stats()["kv_cold_ssd_pages"] >= 1
    # no spill file -> only the host budget counts (SSD capacity ignored)
    pool2 = PagedKvPool(p + 1, hkv, d, num_layers=layers, device=_device())
    pool2.attach_cold(HostKvPages(budget_bytes=per * 2))
    assert pool2.cold_capacity_blocks() == 2


def test_a_recycled_frame_spills_two_pages_to_ssd_under_distinct_keys(tmp_path):
    """The phys-id collision on the spill file: a pool of two frames reissues
    the freed id while the first page's blob is still cold, so a slot keyed by
    the physical id aliases the two pages (#528 fixes the same class one tier
    up). Both pages demote under their own opaque (req, page) keys, land in
    DISTINCT file slots, and promote back byte-equal. budget=0 forces the SSD
    path directly so this cannot pass via host RAM."""
    hkv, d, layers = 2, 8, 2
    pool = PagedKvPool(2, hkv, d, num_layers=layers, device=_device())
    per = (
        pool.k_pool[0, 0].numel() * pool.k_pool.element_size()
        + pool.v_pool[0, 0].numel() * pool.v_pool.element_size()
    ) * layers
    ssd = str(tmp_path / "recycle.bin")
    pool.attach_cold(HostKvPages(budget_bytes=0, ssd_path=ssd))

    saved = {}

    def write_page(key, seed):
        b = pool.alloc_block()
        kk, vv = _kv(seed, 1, hkv, d)
        for plane in range(layers):
            pool.write_block(b, 0, kk[0], vv[0], layer=plane)
        saved[key] = (pool.k_pool[:, b].clone(), pool.v_pool[:, b].clone())
        pool.demote_page(b, key=key)
        return b

    b0 = write_page((0, 0), 11)
    b1 = write_page((0, 1), 22)
    assert b1 == b0, "the freed physical frame must be recycled for the collision to bite"
    st = pool.cold.stats()
    assert st["kv_cold_ssd_pages"] == 2, st
    assert len(pool.cold._ssd) == 2
    assert os.path.getsize(ssd) == pool.cold._ssd.HEADER + per * pool.cold._ssd._cap
    assert pool.cold._ssd._cap >= 2 and pool.cold._ssd._cap < 2 + pool.cold._ssd.GROWTH_SLOTS

    for key in ((0, 0), (0, 1)):
        nb = pool.promote_keyed(key)
        assert torch.equal(pool.k_pool[:, nb], saved[key][0]), f"K {key}"
        assert torch.equal(pool.v_pool[:, nb], saved[key][1]), f"V {key}"


def test_a_spill_file_is_one_process_not_boot(tmp_path):
    """The serving spill keys presence in memory: a reopened tier over the same
    file does NOT resurrect pages (that is KvBootStore's prefix-keyed job)."""
    p, hkv, d = 1, 2, 8
    pool = PagedKvPool(p + 1, hkv, d, num_layers=1, device=_device())
    per = 2 * pool.k_pool[0, 0].numel() * pool.k_pool.element_size()
    ssd = str(tmp_path / "c.bin")
    pool.attach_cold(HostKvPages(budget_bytes=0, ssd_path=ssd))
    kk, vv = _kv(0, p, hkv, d)
    b = pool.alloc_block()
    pool.write_block(b, 0, kk[0], vv[0], layer=0)
    pool.demote_page(b)
    assert pool.cold.ssd_bytes == per
    reopened = HostKvPages(budget_bytes=0, ssd_path=ssd)
    assert reopened.take(b) is None


def test_batched_promotions_copy_many_pages_but_sync_once():
    """A decode tick that fetches N cold pages must pay ONE device sync, not N —
    the per-page cuda.synchronize inside promote was the 6.6x decode slowdown.
    Demote three pages, then promote all three inside pool.promotions() on a pool
    whose device we pretend is cuda: exactly one synchronize must be observed and
    every page must come back byte-equal."""
    import unittest.mock as mock

    p, hkv, d = 4, 2, 8
    pool = PagedKvPool(p + 8, hkv, d, num_layers=2, device=_device())
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    snaps = {}
    for i in range(3):
        b = pool.alloc_block()
        kk, vv = _kv(100 + i, 1, hkv, d)
        for plane in range(2):
            pool.write_block(b, 0, kk[0], vv[0], layer=plane)
        snaps[i] = (pool.k_pool[:, b].clone(), pool.v_pool[:, b].clone())  # keyed logical i
        pool.demote_page(b, key=("r", i))

    # On the CPU cell copies are synchronous clones, so drive the cuda branch by
    # pretending the device is cuda and counting synchronize calls.
    real_device = pool.device
    pool.device = torch.device("cuda")
    try:
        with mock.patch("tilerl.kv_cache.torch.cuda.synchronize") as sync_fn, pool.promotions():
            new_blocks = [pool.promote_keyed(("r", i)) for i in range(3)]
            # still inside the batch: the single sync happens only at context exit
            assert sync_fn.call_count == 0, sync_fn.call_count
        assert sync_fn.call_count == 1, f"expected one batched sync, got {sync_fn.call_count}"
    finally:
        pool.device = real_device
    for i, nb in enumerate(new_blocks):
        assert torch.equal(pool.k_pool[:, nb], snaps[i][0])
        assert torch.equal(pool.v_pool[:, nb], snaps[i][1])


def test_batched_demotions_copy_many_pages_but_sync_once():
    """A decode tick that evicts N pages must pay ONE device sync, not N — the
    demote mirror of the promotions gate. Three live pages demote inside one
    pool.demotions() on a pool whose device we pretend is cuda: exactly one
    synchronize, and every page's pinned host blob is byte-equal to the device
    frame it copied. Frames stay live until that sync."""
    import unittest.mock as mock

    p, hkv, d = 4, 2, 8
    pool = PagedKvPool(p + 8, hkv, d, num_layers=2, device=_device())
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    blocks = [pool.alloc_block() for _ in range(3)]
    snaps = {}
    for i, b in enumerate(blocks):
        kk, vv = _kv(200 + i, 1, hkv, d)
        for plane in range(2):
            pool.write_block(b, 0, kk[0], vv[0], layer=plane)
        snaps[b] = (pool.k_pool[:, b].clone(), pool.v_pool[:, b].clone())

    real_device = pool.device
    pool.device = torch.device("cuda")
    try:
        with mock.patch("tilerl.kv_cache.torch.cuda.synchronize") as sync_fn, pool.demotions():
            for i, b in enumerate(blocks):
                pool.demote_page(b, key=("r", i))
                # inside the batch: frames not freed and sync not yet issued
                assert pool.refcount[b] == 1, "frame stays live until end sync"
            assert sync_fn.call_count == 0, sync_fn.call_count
        assert sync_fn.call_count == 1, f"one batched sync, got {sync_fn.call_count}"
    finally:
        pool.device = real_device
    for i, b in enumerate(blocks):
        assert pool.refcount[b] == 0, "frame returned to the pool only after sync"
        blob = pool.cold.take(("r", i))
        assert blob is not None
        assert torch.equal(blob["k"], snaps[b][0]), f"K {i}"
        assert torch.equal(blob["v"], snaps[b][1]), f"V {i}"


def test_batched_demotion_survives_frame_recycling():
    """The collision case: two pages demoted in one batch on a 2-frame pool must
    not alias even though their physical ids are recycled. Frames stay live until
    the single end-of-batch sync, so the second D2H cannot overwrite the first;
    both pinned blobs promote back byte-equal through a real CPU pool."""
    hkv, d = 2, 8
    pool = PagedKvPool(2, hkv, d, num_layers=2, device=_device())
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    saved = {}

    def write_page(key, seed):
        b = pool.alloc_block()
        kk, vv = _kv(seed, 1, hkv, d)
        for plane in range(2):
            pool.write_block(b, 0, kk[0], vv[0], layer=plane)
        saved[key] = (b, pool.k_pool[:, b].clone(), pool.v_pool[:, b].clone())
        return b

    with pool.demotions():
        b0 = write_page((0, 0), 31)
        pool.demote_page(b0, key=(0, 0))
        b1 = write_page((0, 1), 42)
        assert b1 != b0, "retained frames must not be reused mid-batch"
        pool.demote_page(b1, key=(0, 1))
    assert pool.refcount[b0] == 0 and pool.refcount[b1] == 0

    for key in ((0, 0), (0, 1)):
        nb = pool.promote_keyed(key)
        assert torch.equal(pool.k_pool[:, nb], saved[key][1]), f"K {key}"
        assert torch.equal(pool.v_pool[:, nb], saved[key][2]), f"V {key}"


def test_budget_enforcement_is_constant_work_per_hold(tmp_path):
    """The O(1) LRU: once the host budget binds, a hold that evicts one page must
    not snapshot/scan every RAM-resident entry. With the old list(_ram_order.items())
    inner loop a 16k fill against a 32-page budget scanned ~n*budget dict items;
    the OrderedDict-front implementation pops each page at most once and never
    calls items() on the eviction path."""
    from collections import OrderedDict

    per = 256

    class CountingOrder(OrderedDict):
        def __init__(self):
            super().__init__()
            self.items_calls = 0
            self.popitem_calls = 0

        def items(self):  # noqa: D401
            self.items_calls += 1
            return super().items()

        def popitem(self, last=True):  # noqa: D401
            self.popitem_calls += 1
            return super().popitem(last)

    cold = HostKvPages(budget_bytes=per * 32, ssd_path=str(tmp_path / "o1.bin"))
    cold._ram_order = CountingOrder()
    n = 16384
    for i in range(n):
        blob = {"x": torch.arange(per, dtype=torch.uint8).reshape(2, -1)}
        assert cold.hold(i, blob, per)
        assert cold.bytes_held <= per * 32 + per
    # exactly the evicted pages were popped, once each; the 32 survivors never are
    assert cold._ram_order.popitem_calls == n - 32, cold._ram_order.popitem_calls
    assert cold._ram_order.items_calls == 0, "eviction scanned a dict snapshot"
    st = cold.stats()
    assert st["kv_cold_pages"] == 32 and st["kv_cold_ssd_pages"] == n - 32, st
    cold.close()


def test_stale_lru_entries_are_popped_not_rescanned(tmp_path):
    """A private LRU entry whose blob is already gone (promoted/forgotten between
    enqueue and sweep) and a shared entry with no live record are removed from the
    LRU on the next sweep. The old code `continue`d past them and left them in the
    OrderedDict, so every later hold scanned the same dead records forever."""
    per = 128
    cold = HostKvPages(budget_bytes=per, ssd_path=str(tmp_path / "stale.bin"))
    cold.hold("a", {"x": torch.zeros(per, dtype=torch.uint8)}, per)
    # two dead records ahead of the live one: no blob / no shared record
    cold._ram_order.clear()
    cold._ram_order[("p", 999)] = per  # private: not in _blobs/_held
    cold._ram_order[("s", 888)] = per  # shared: not in _shared
    cold._ram_order[("p", "a")] = per
    cold.hold("b", {"x": torch.zeros(per, dtype=torch.uint8)}, per)
    assert ("p", 999) not in cold._ram_order and ("s", 888) not in cold._ram_order
    assert cold.bytes_held == per
    # the budget loop is genuinely settled, not wedged on a dead record
    cold.hold("c", {"x": torch.zeros(per, dtype=torch.uint8)}, per)
    assert cold.bytes_held == per and ("p", 999) not in cold._ram_order
    cold.close()


def test_spill_file_grows_one_extent_and_round_trips_bytes(tmp_path):
    """Growth is one ftruncate+remap per extent (not per high-water slot), and the
    zero-copy numpy-view write preserves every byte across a remap, including the
    slots written before the file grew."""
    from tilerl.kv_tiers import ColdSsdFile

    spec = [("x", (64,), "float32", 256)]
    f = ColdSsdFile(str(tmp_path / "ext.bin"), spec)
    remaps = []
    orig = f._remap

    def counted(cap):
        remaps.append(cap)
        return orig(cap)

    f._remap = counted
    n_slots = ColdSsdFile.GROWTH_SLOTS + 5
    blobs = {
        i: {"x": torch.arange(64, dtype=torch.float32) * (i + 1) * 0.5} for i in range(n_slots)
    }
    for i, blob in blobs.items():
        f.write(i, blob)
    # one ftruncate+remap per extent boundary (64, then 128), never per slot
    assert remaps == [ColdSsdFile.GROWTH_SLOTS, 2 * ColdSsdFile.GROWTH_SLOTS], remaps
    assert f._cap == 2 * ColdSsdFile.GROWTH_SLOTS
    assert os.path.getsize(str(tmp_path / "ext.bin")) == f.HEADER + f._cap * 256
    for i, blob in blobs.items():  # early slots survive the remap, byte-exact
        assert torch.equal(f.read(i, False)["x"], blob["x"]), i
    f.close()


def test_unspillable_shared_pages_pin_in_ram_without_wedging_the_lru():
    """No spill file: a refcounted shared page can neither spill nor drop, so under
    budget pressure the LRU parks it and returns instead of looping on an entry it
    already popped (the front-pop rewrite must re-insert, not move_to_end a missing
    key). A later private hold still evicts the private page it can spill-drop."""
    per = 128
    cold = HostKvPages(budget_bytes=per)  # no ssd_path
    blob = {"x": torch.zeros(per, dtype=torch.uint8)}
    cold.share_hold(7, dict(blob), per)  # the only RAM page, pinned shared
    # budget already bound; a private page forces the loop past the unspillable
    # shared entry once and must return, not raise/loop forever
    cold.hold(1, {"x": torch.zeros(per, dtype=torch.uint8)}, per)
    assert cold.share_take(7) is not None  # shared page retained
    assert cold.bytes_held >= per


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))


def test_ssd_spill_with_prefix_sharing_keeps_host_bytes_under_budget(tmp_path):
    """256k host OOM (#553): every dropped page was cloned once for private
    storage and again for the shared prefix index, and _shared had no byte
    budget and never spilled, so anonymous RSS grew a second full KV copy.
    Demote thousands of pages through the SSD tier with the prefix SHARE path
    active and assert TOTAL host bytes (private + shared + spilled-read cache)
    stay within the host budget plus one page — a third copy cannot hide."""
    p, hkv, d, layers = 32, 2, 8, 2
    pool = PagedKvPool(p, hkv, d, num_layers=layers, device=_device())
    per = (
        pool.k_pool[0, 0].numel() * pool.k_pool.element_size()
        + pool.v_pool[0, 0].numel() * pool.v_pool.element_size()
    ) * layers
    budget = per * 4
    ssd = str(tmp_path / "cold_spill.bin")
    cold = HostKvPages(budget_bytes=budget, ssd_path=ssd)
    pool.attach_cold(cold)

    def host_bytes():
        # EVERY anonymous host container: private RAM + shared RAM
        return cold.bytes_held

    n = 4096
    for i in range(n):
        b = pool.alloc_block()
        pool.k_pool[:, b].fill_(i % 7)
        pool.v_pool[:, b].fill_(-(i % 5))
        pool.demote_page(b, key=i)
        # mimic the engine's drop-only publish: transfer the PRIVATE blob to a
        # content key (no clone), with a small bounds tensor attached
        bound = torch.zeros(2, dtype=torch.float16)
        key = 1_000_000 + i
        cold.share_hold_kv(i, key, extra={"bounds": bound})
        # the running shared-RAM counter must equal the real RAM-resident sum
        # through every hold/spill/read-through transition
        assert cold._shared_ram == sum(
            rec[0] for k, rec in cold._shared.items() if k in cold._shared_blobs
        )
        assert host_bytes() <= budget + per, (i, host_bytes(), budget)

    st = cold.stats()
    # every transferred page still EXISTS: private SSD + shared RAM/file total n
    accounted = st["kv_cold_pages"] + st["kv_cold_ssd_pages"] + st["kv_cold_shared_pages"]
    assert accounted >= n, (st, n)
    # read-through: a spilled shared page still resolves
    blob = cold.share_take(1_000_000 + n - 1)
    assert blob is not None
    # releasing all refs leaves no shared host residue
    for key in range(1_000_000, 1_000_000 + n):
        cold.share_release(key)
    assert cold.shared_bytes() == 0 and cold._shared_ssd_bytes == 0
    cold.close()


def test_shared_transfer_of_an_already_spilled_private_page_reads_back(tmp_path):
    """256k case: when the contiguous frontier closes the page's private blob is
    already on the private SSD (past the host budget). The private->shared
    transfer must still produce a readable shared blob - lifted off the private
    file and written to the prefix spill file, with bounds attached, adding
    nothing to host RAM."""
    p, hkv, d, layers = 8, 2, 8, 2
    pool = PagedKvPool(p, hkv, d, num_layers=layers, device=_device())
    per = (
        pool.k_pool[0, 0].numel() * pool.k_pool.element_size()
        + pool.v_pool[0, 0].numel() * pool.v_pool.element_size()
    ) * layers
    ssd = str(tmp_path / "cold.bin")
    cold = HostKvPages(budget_bytes=per, ssd_path=ssd)  # one page host budget
    pool.attach_cold(cold)

    blocks = [pool.alloc_block() for _ in range(p)]
    for i, b in enumerate(blocks):
        pool.k_pool[:, b].fill_(i)
        pool.v_pool[:, b].fill_(-i)
        pool.demote_page(b, key=i)
    # every page beyond the first is on the private SSD
    assert cold.ssd_bytes >= per * (p - 1)
    ram_before = cold.bytes_held

    bound = torch.zeros(2, dtype=torch.float16)
    n = cold.share_hold_kv(p - 1, 777, extra={"bounds": bound})
    assert n > 0
    # host RAM did not grow to hold the lifted blob
    assert cold.bytes_held <= ram_before + per
    # the shared page resolves (read-through) carrying its bound and K/V
    blob = cold.share_take(777)
    assert blob is not None and "bounds" in blob
    assert torch.all(blob["k"] == (p - 1)) and torch.all(blob["v"] == -(p - 1))
    # the private copy is gone
    assert cold.take(p - 1) is None
    cold.close()


def test_two_spilled_publishers_of_one_content_key_share_one_slot(tmp_path):
    """52 CHANGE-REQ: two identical-prompt requests whose pages are BOTH already
    on the private SSD transfer to the same content key. The second transfer must
    ref++, not reset refs to 1, allocate a second prefix-file slot, or let one
    release forget the slot the other publisher still serves."""
    p, hkv, d, layers = 8, 2, 8, 2
    pool = PagedKvPool(p, hkv, d, num_layers=layers, device=_device())
    per = (
        pool.k_pool[0, 0].numel() * pool.k_pool.element_size()
        + pool.v_pool[0, 0].numel() * pool.v_pool.element_size()
    ) * layers
    cold = HostKvPages(budget_bytes=per, ssd_path=str(tmp_path / "cold.bin"))
    pool.attach_cold(cold)

    blocks = [pool.alloc_block() for _ in range(p)]
    for i, b in enumerate(blocks):
        pool.k_pool[:, b].fill_(i)
        pool.v_pool[:, b].fill_(-i)
        pool.demote_page(b, key=i)
    assert cold.ssd_bytes >= per * (p - 1)

    bound = torch.zeros(2, dtype=torch.float16)
    cold.share_hold_kv(p - 1, 777, extra={"bounds": bound})
    cold.share_hold_kv(p - 2, 777, extra={"bounds": bound})

    assert cold._shared[777][1] == 2  # refs, not reset to 1
    assert len(cold._shared_ssd) == 1  # exactly one prefix-file slot
    # read-through by field does not consume the spilled slot
    assert cold.share_take_field(777, "bounds") is not None
    assert len(cold._shared_ssd) == 1
    cold.share_release(777)
    assert 777 in cold.share_keys()  # B's entry still served
    assert ("s", 777) in cold._shared_ssd  # its slot survived A's release
    assert cold.share_take(777) is not None  # full read-through resolves
    cold.share_release(777)
    assert 777 not in cold.share_keys()
    cold.close()


def test_an_unwritable_spill_path_refuses_to_construct(tmp_path):
    """Fail fast: an unwritable --cold-ssd-path (and its .prefix.bin sibling)
    must be rejected when HostKvPages is built, not after the host budget binds
    mid-decode. Red on main: the constructor never probed the path."""
    import pytest

    from tilerl.kv_tiers import SpillWriteError

    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)  # r-x, no write
    try:
        with pytest.raises(SpillWriteError):
            HostKvPages(budget_bytes=1 << 20, ssd_path=str(ro / "cold.bin"))
    finally:
        ro.chmod(0o700)


def test_a_shared_spill_failure_stays_in_ram_and_disables_spill(tmp_path):
    """Shared prefix spill is a cache. If the sibling write raises, the page
    stays in RAM, shared SSD spill turns off for the process, and the caller
    never sees — private spill is untouched. Red on main: the OSError escaped."""
    from tilerl.kv_tiers import ColdSsdFile

    per = 2 * 2 * BLOCK_TOKENS
    cold = HostKvPages(budget_bytes=per * 2, ssd_path=str(tmp_path / "c.bin"))

    def boom(*_a, **_k):
        raise OSError(13, "Permission denied")

    orig = ColdSsdFile.write
    ColdSsdFile.write = boom  # fire on BOTH files; private failure must still raise
    try:
        blob = {"k": torch.zeros(1), "v": torch.zeros(1), "bounds": torch.zeros(1)}
        n = 3
        cold.share_hold(11, blob, n)
        # Force a shared eviction: it must not raise and the page is retained.
        ok = cold._shared_evict_ram(11)
        assert ok is False
        assert cold.shared_spill_disabled is True
        assert cold.share_take(11) is not None  # still served from RAM

        # a PRIVATE page past the tiny budget still fails loudly (live row needs it)
        import pytest

        from tilerl.kv_tiers import SpillWriteError

        big = {f"k{i}": torch.zeros(per) for i in range(4)}
        with pytest.raises(SpillWriteError):
            cold.hold(("r", 9), big, per * 3)
    finally:
        ColdSsdFile.write = orig
        cold.close()


def test_bounded_shared_spill_refuses_pages_past_the_cap_and_keeps_them_in_ram(
    tmp_path, monkeypatch
):
    """TILERL_COLD_PREFIX_SSD_CAP=1 bounds the publish-only .prefix.bin by the
    same --cold-ssd-bytes admission the private spill reports (observed
    unbounded: 13.6 GiB logical / 25 GiB physical vs an 8 GiB cap). A shared
    page past the cap is not written (False) and stays in host RAM instead of
    filling the disk; the counter never exceeds the cap. Off by default."""
    monkeypatch.setenv("TILERL_COLD_PREFIX_SSD_CAP", "1")
    per = 2 * 2 * BLOCK_TOKENS
    ssd = str(tmp_path / "c.bin")
    # host holds 1 page; explicit SSD cap holds only 2 shared pages
    cold = HostKvPages(budget_bytes=per, ssd_path=ssd, ssd_capacity_bytes=per * 2)
    assert cold.prefix_spill_bounded is True
    assert cold.stats()["kv_cold_shared_ssd_bounded"] == 1
    try:

        def page(i):
            return {
                "k": torch.full((per,), i, dtype=torch.uint8),
                "v": torch.full((per,), i, dtype=torch.uint8),
                "bounds": torch.zeros(1, dtype=torch.uint8),
            }

        # three holds against a 1-page RAM budget: LRU spills the first two to the
        # shared SSD (filling the 2-page cap); the third stays in RAM
        for i in (1, 2, 3):
            cold.share_hold(i, page(i), per)
        assert cold._shared_ssd_bytes == per * 2, cold._shared_ssd_bytes
        # the RAM-resident third page cannot spill past the cap -> stays RAM
        assert cold._shared_evict_ram(3) is False
        assert cold._shared_ssd_bytes == per * 2
        # the page is still served from RAM (a cache miss, not a lost publish)
        assert cold.share_take(3) is not None
    finally:
        cold.close()


def test_host_tier_wires_reclaim_to_the_shared_spill_only_by_env(tmp_path, monkeypatch):
    """Wiring-layer gate (red on main): the reclaim flag must reach the LAZILY
    created SHARED ColdSsdFile from the env, and forgetting a trailing live page
    must actually shrink that file; the PRIVATE spill file is never reclaiming.
    Constructing ColdSsdFile(reclaim=True) directly does not prove the host tier
    passes it when it lazily opens .prefix.bin — this exercises that seam."""
    per = 2 * 2 * BLOCK_TOKENS
    ssd = str(tmp_path / "w.bin")

    def page(i):
        return {
            "k": torch.full((per,), i, dtype=torch.uint8),
            "v": torch.full((per,), i, dtype=torch.uint8),
            "bounds": torch.zeros(1, dtype=torch.uint8),
        }

    # env ON: a shared spill exists after LRU eviction and is reclaiming; release
    # of its only live slot truncates the trailing extent back.
    monkeypatch.setenv("TILERL_COLD_PREFIX_SSD_CAP", "1")
    cold = HostKvPages(budget_bytes=per, ssd_path=ssd, ssd_capacity_bytes=per * 8)
    try:
        cold.share_hold(1, page(1), per)
        cold.share_hold(2, page(2), per)  # evicts 1 to the shared SSD
        assert cold._shared_ssd is not None
        assert cold._shared_ssd._reclaim is True, "env on must wire reclaim=True to shared"
        size_before = os.path.getsize(cold._shared_ssd._path)
        cold.share_release(1)  # last ref of the spilled page -> forget + tail reclaim
        assert os.path.getsize(cold._shared_ssd._path) < size_before
    finally:
        cold.close()

    # env OFF: the same shared spill is created with reclaim=False and never shrinks.
    monkeypatch.delenv("TILERL_COLD_PREFIX_SSD_CAP", raising=False)
    cold2 = HostKvPages(budget_bytes=per, ssd_path=ssd, ssd_capacity_bytes=per * 8)
    try:
        cold2.share_hold(1, page(1), per)
        cold2.share_hold(2, page(2), per)
        assert cold2._shared_ssd is not None and cold2._shared_ssd._reclaim is False
        size_before = os.path.getsize(cold2._shared_ssd._path)
        cold2.share_release(1)
        assert os.path.getsize(cold2._shared_ssd._path) == size_before
    finally:
        cold2.close()

    # The PRIVATE spill file is NEVER reclaiming, regardless of the env.
    monkeypatch.setenv("TILERL_COLD_PREFIX_SSD_CAP", "1")
    pool = PagedKvPool(2, 2, 8, num_layers=2, device=_device())
    pool.attach_cold(HostKvPages(budget_bytes=0, ssd_path=str(tmp_path / "p.bin")))
    kk, vv = _kv(7, 1, 2, 8)
    b = pool.alloc_block()
    pool.write_block(b, 0, kk[0], vv[0], layer=0)
    pool.write_block(b, 0, kk[0], vv[0], layer=1)
    pool.demote_page(b, key=(3, 0))
    assert pool.cold._ssd is not None and pool.cold._ssd._reclaim is False
    pool.cold.close()

    # default OFF (env removed): the same write past the cap is allowed
    monkeypatch.delenv("TILERL_COLD_PREFIX_SSD_CAP", raising=False)
    cold2 = HostKvPages(budget_bytes=per, ssd_path=ssd, ssd_capacity_bytes=per)
    try:
        assert cold2.prefix_spill_bounded is False
        b = {
            "k": torch.zeros(per, dtype=torch.uint8),
            "v": torch.zeros(per, dtype=torch.uint8),
            "bounds": torch.zeros(1, dtype=torch.uint8),
        }
        cold2.share_hold(9, {k: t.clone() for k, t in b.items()}, per)
        assert cold2._shared_evict_ram(9) is True
        assert cold2._shared_ssd_bytes == per
    finally:
        cold2.close()


def test_spill_file_reclaims_freed_trailing_extents_to_disk(tmp_path, monkeypatch):
    """reclaim=True truncates the file back when the last live extent empties, so
    a release wave returns physical disk instead of leaving the spill at its
    high-water mark forever (the #735 extent growth reuses slots but never
    shrank). Mid-file free extents cycle through the LIFO free list; only the
    fully-free TAIL collapses."""
    from tilerl.kv_tiers import ColdSsdFile

    monkeypatch.setenv("TILERL_COLD_PREFIX_SSD_CAP", "1")
    spec = [("x", (64,), "float32", 256)]
    f = ColdSsdFile(str(tmp_path / "r.bin"), spec, reclaim=True)
    g = ColdSsdFile.GROWTH_SLOTS
    grown = f.HEADER + 2 * g * 256
    for i in range(g + 2):  # force growth into the 2nd extent
        f.write(i, {"x": torch.arange(64, dtype=torch.float32) + i})
    assert os.path.getsize(str(tmp_path / "r.bin")) == grown
    # release just one early (extent-0) page: no tail collapse (extent 1 full)
    f.forget(0)
    assert os.path.getsize(str(tmp_path / "r.bin")) == grown
    # release every extent-1 page (the tail) -> file collapses to one extent
    for i in range(g, g + 2):
        f.forget(i)
    one_extent = f.HEADER + g * 256
    assert f._cap == g
    assert os.path.getsize(str(tmp_path / "r.bin")) == one_extent
    # remaining extent-0 slots (except 0) still round-trip byte-exact
    for i in range(1, g):
        assert torch.equal(f.read(i, False)["x"], torch.arange(64, dtype=torch.float32) + i)
    # releasing the last live extent collapses the file to the header only
    for i in range(1, g):
        f.forget(i)
    assert f._cap == 0
    assert os.path.getsize(str(tmp_path / "r.bin")) == f.HEADER
    f.close()

    # reclaim=False (default private file / gate off): forgetting never shrinks
    f2 = ColdSsdFile(str(tmp_path / "r2.bin"), spec, reclaim=False)
    for i in range(g + 2):
        f2.write(i, {"x": torch.zeros(64, dtype=torch.float32)})
    for i in range(g + 2):
        f2.forget(i)
    assert f2._cap == 2 * g and os.path.getsize(str(tmp_path / "r2.bin")) == grown
    f2.close()


# ---------------------------------------------------------------- background publish


def _cold_blob(fill=1.0):
    return {
        "k": torch.full((2, 4), fill, dtype=torch.float16),
        "v": torch.full((2, 4), -fill, dtype=torch.float16),
        "bounds": torch.zeros(2),
    }


def _warm_blob(fill=2.0):
    b = _cold_blob(fill)
    b["dk"] = torch.full((2, 4), fill, dtype=torch.float16)
    b["dv"] = torch.full((2, 4), -fill, dtype=torch.float16)
    return b


def test_shared_spill_holds_heterogeneous_warm_and_cold_blobs(tmp_path):
    """A 3-field cold blob (k/v/bounds) and a 5-field warm blob (plus dk/dv)
    share the SAME prefix spill file. The file layout is frozen from whichever
    blob first created it, so today:
      cold-first -> the warm blob's dk/dv are silently dropped, share_take_field
                    returns a wrong-value/absent field instead of a clean miss;
      warm-first -> spilling the cold blob raises KeyError('dk').
    A bucketed/signature-tagged layout must store both: a missing field on a
    record that does not own it reads as None (cache miss), never KeyError, and
    owned fields round-trip their own bytes in both creation orders."""

    def nbytes(b):
        return sum(t.numel() * t.element_size() for t in b.values())

    # --- cold-first ---
    d = tmp_path / "coldfirst"
    d.mkdir()
    cold = HostKvPages(budget_bytes=64, ssd_path=str(d / "x.bin"))
    try:
        cold.share_hold(100, _cold_blob(1.0), nbytes(_cold_blob()))
        cold.share_hold(200, _warm_blob(2.0), nbytes(_warm_blob()))
        # budget 64: the 40-byte cold page and the 72-byte warm page both spill,
        # each into its own signature bucket
        assert 100 not in cold._shared_blobs and 200 not in cold._shared_blobs
        # a field the cold record does not own is a clean miss, not a KeyError
        assert cold.share_take_field(100, "dk") is None
        # the warm record owns dk/dv and must read them back (fill 2.0)
        dk = cold.share_take_field(200, "dk")
        assert dk is not None and torch.all(dk == 2.0)
        warm = cold.share_take(200)
        assert torch.all(warm["k"] == 2.0) and torch.all(warm["dk"] == 2.0)
    finally:
        cold.close()

    # --- warm-first ---
    d2 = tmp_path / "warmfirst"
    d2.mkdir()
    warmfirst = HostKvPages(budget_bytes=64, ssd_path=str(d2 / "y.bin"))
    try:
        warmfirst.share_hold(300, _warm_blob(2.0), nbytes(_warm_blob()))
        warmfirst.share_hold(400, _cold_blob(3.0), nbytes(_cold_blob()))
        assert 300 not in warmfirst._shared_blobs  # 72-byte warm page spilled
        c400 = warmfirst.share_take(400)  # must not KeyError on the 3->5 layout
        assert c400 is not None and torch.all(c400["k"] == 3.0)
        # the cold record owns no dk/dv -> miss, not a KeyError
        assert warmfirst.share_take_field(400, "dk") is None
        assert torch.all(warmfirst.share_take(300)["dk"] == 2.0)
    finally:
        warmfirst.close()


def test_spill_io_on_publish_thread_is_billed_separately(tmp_path):
    """ColdSsdFile.ssd_ms keeps step-thread spill IO; IO done on the
    tilerl-cold-publish thread accrues to ssd_ms_worker so the worker's disk
    time cannot be drained into a step tick and masquerade as a close stall.
    Thread identity is the only discriminator, so drive the real write/read on a
    thread renamed to the worker."""
    import threading

    from tilerl.kv_tiers import ColdSsdFile

    spec = [("x", (4,), "float32", 16)]
    f = ColdSsdFile(str(tmp_path / "spill.bin"), spec, step_timing=object())
    blob = {"x": torch.arange(4, dtype=torch.float32)}
    try:
        f.write(("a", 1), blob)
        assert f.ssd_ms > 0.0 and f.ssd_ms_worker == 0.0
        step_before = f.ssd_ms
        got = {}

        def _worker_io():
            f.write(("b", 2), blob)
            got["blob"] = f.read(("b", 2), False)["x"]
            got["name"] = threading.current_thread().name

        th = threading.Thread(target=_worker_io, name=ColdSsdFile.PUBLISH_THREAD)
        th.start()
        th.join()
        assert got["name"] == ColdSsdFile.PUBLISH_THREAD
        assert torch.equal(got["blob"], blob["x"])
        # the worker's two IO calls did not touch the step bucket ...
        assert f.ssd_ms == step_before
        assert f.ssd_ms_worker > 0.0
        # the two accumulators reset independently when read.
        wms = f.ssd_ms_worker
        f.ssd_ms_worker = 0.0
        assert f.ssd_ms_worker == 0.0 and wms > 0.0
    finally:
        f.close()


def test_shared_bucket_rejects_dtype_and_shape_mismatch_to_ram(tmp_path):
    """Defect B (frozen-spec mismatch): a bucket is keyed by field-name signature
    but freezes ONE concrete dtype+shape. An equal-byte-width dtype swap must not
    be silently reinterpreted as the frozen dtype on readback, and a shape
    mismatch must not raise out of share_hold (which would wedge the tick and
    leak a slot). Both fall back to RAM and are counted. Mutation-red on removing
    the spec comparison (the dtype swap reads back as wrong values)."""
    ssd = str(tmp_path / "spec.bin")
    cold = HostKvPages(budget_bytes=64, ssd_path=ssd, ssd_capacity_bytes=1 << 30)
    cold.prefix_spill_bounded = True
    try:
        # establish the cold bucket frozen to fp16: a second 40-byte page pushes
        # the first past the 64-byte budget and LRU-spills it into the bucket.
        cold.share_hold(
            100, _cold_blob(1.0), sum(t.numel() * t.element_size() for t in _cold_blob().values())
        )
        cold.share_hold(
            101, _cold_blob(2.0), sum(t.numel() * t.element_size() for t in _cold_blob().values())
        )
        bucket = cold._shared_ssd
        assert bucket is not None

        # same field names+shapes, EQUAL WIDTH (bf16 is 2 bytes) but different
        # dtype: rejected to RAM, never reinterpreted through the fp16 spec
        bf = {
            "k": torch.full((2, 4), 7.0, dtype=torch.bfloat16),
            "v": torch.full((2, 4), -7.0, dtype=torch.bfloat16),
            "bounds": torch.zeros(2),
        }
        nbf = sum(t.numel() * t.element_size() for t in bf.values())
        cold.share_hold(200, bf, nbf)
        # the 40-byte bf page fits the 64-byte budget, so force it through the
        # LRU spill path by admitting a second page: the bf page is oldest, its
        # bucket write is spec-rejected (stays RAM), then the matching page spills.
        cold.share_hold(
            201, _cold_blob(5.0), sum(t.numel() * t.element_size() for t in _cold_blob().values())
        )
        assert cold.shared_spec_failures >= 1
        assert 200 in cold._shared_blobs  # kept in RAM ...
        blob = cold.share_take(200)
        assert blob is not None and blob["k"].dtype == torch.bfloat16
        assert torch.all(blob["k"] == 7.0)  # ... with its TRUE dtype, not garbage
        assert ("s", 200) not in bucket

        # different SHAPE under the same signature: share_hold must NOT raise; the
        # page stays in RAM and no slot/reservation is leaked.
        odd = {
            "k": torch.zeros(2, 8, dtype=torch.float16),
            "v": torch.zeros(2, 8, dtype=torch.float16),
            "bounds": torch.zeros(2),
        }
        nodd = sum(t.numel() * t.element_size() for t in odd.values())
        cold.share_hold(300, odd, nodd)  # must not raise
        assert ("s", 300) not in bucket
        # One-time synchronous spill never reserves a slot for a rejected page:
        # it stays in RAM and is absent from every signature bucket on disk.
        assert 300 not in bucket
        assert 300 in cold._shared_blobs
    finally:
        cold.close()


def test_cold_spill_file_check_blob_spec_raises_at_the_source(tmp_path):
    """Defect B assertion anchored at the ColdSsdFile raise point itself (rev
    nit): the two end-to-end gates above prove the RAM-fallback behaviour, but
    this pins the exact validator — a frozen spec rejects an equal-width dtype
    swap and a shape mismatch with SpillSpecError, and accepts an exact match —
    so a refactor that only relaxed the host-level catch could not silently drop
    the real check."""
    import pytest

    from tilerl.kv_tiers import ColdSsdFile, SpillSpecError, _blob_spec

    base = {
        "k": torch.zeros(2, 4, dtype=torch.float16),
        "v": torch.zeros(2, 4, dtype=torch.float16),
        "bounds": torch.zeros(2),
    }
    f = ColdSsdFile(str(tmp_path / "b.bin"), _blob_spec(base))
    try:
        f._check_blob_spec(base)  # exact match: no raise
        same_shape_other_dtype = {
            "k": torch.zeros(2, 4, dtype=torch.bfloat16),
            "v": torch.zeros(2, 4, dtype=torch.bfloat16),
            "bounds": torch.zeros(2),
        }
        other_shape = {
            "k": torch.zeros(2, 8, dtype=torch.float16),
            "v": torch.zeros(2, 8, dtype=torch.float16),
            "bounds": torch.zeros(2),
        }
        missing_field = {
            "k": torch.zeros(2, 4, dtype=torch.float16),
            "v": torch.zeros(2, 4, dtype=torch.float16),
        }
        with pytest.raises(SpillSpecError):
            f._check_blob_spec(same_shape_other_dtype)
        with pytest.raises(SpillSpecError):
            f._check_blob_spec(other_shape)
        with pytest.raises(SpillSpecError):
            f._check_blob_spec(missing_field)
    finally:
        f.close()


def test_step_thread_shared_write_failure_keeps_page_in_ram_no_leak(tmp_path):
    """Defect D on the inline (step-thread) path: a non-OSError failure writing
    the shared bucket must not propagate out of share_hold (which would wedge the
    close tick holding _tlock), must not strand an allocated slot, and must keep
    the page in RAM. Deterministic by forcing the inline bucket write to raise.
    Mutation-red on the inline write not catching BaseException (it escapes
    share_hold) or on _write not returning its post-alloc slot to the free list."""
    ssd = str(tmp_path / "dinline.bin")
    cold = HostKvPages(budget_bytes=64, ssd_path=ssd, ssd_capacity_bytes=1 << 30)
    cold.prefix_spill_bounded = True
    # establish a frozen cold bucket
    cold.share_hold(
        100, _cold_blob(1.0), sum(t.numel() * t.element_size() for t in _cold_blob().values())
    )
    cold.share_hold(
        101, _cold_blob(2.0), sum(t.numel() * t.element_size() for t in _cold_blob().values())
    )
    bucket = cold._shared_ssd
    assert bucket is not None
    before = len(bucket._free_slots)
    real_write = bucket._write

    def boom_write(key, blob):
        raise RuntimeError("simulated inline copy failure")

    bucket._write = boom_write
    # force a fresh spill attempt of a RAM-resident page
    nb = sum(t.numel() * t.element_size() for t in _cold_blob().values())
    cold.share_hold(200, _cold_blob(3.0), nb)
    try:
        cold._shared_evict_ram(200)  # must NOT raise
    finally:
        bucket._write = real_write
    # page retained in RAM, no slot stranded
    assert cold.share_take(200) is not None
    assert len(bucket._free_slots) == before
    cold.close()


