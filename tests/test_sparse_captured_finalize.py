"""Coverage for the captured-tick finalize guard
(errors/2026-09-16-sm90-sparse-graph-no-speedup-finalize-d2h).

A captured CUDA sparse decode tick skips the device→host pin readback in
``SparseRuntime.finalize``; residency self-prunes via ``evict_victim`` and the
1-in-N eager refresh tick reconciles the pin set and publishes drops. The guard
reads only ``sf.device.type`` and ``sf.device_select``, so a CPU test drives
the real runtime with a fake-cuda ``SparseForward`` (SimpleNamespace) and an
eager ``SparseForward`` whose ``selected_pages`` returns a controlled set.

Red control: dropping the ``continue`` makes the captured tick reconcile with
the eager set and this test fails (captured tick would demote + offer).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tilerl.engine import SamplingParams
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.sparse_runtime import SparseRuntime


def _decode_engine():
    from tilerl.build import build_engine
    from tilerl.config import tiny
    from tilerl.model import build_random
    from tilerl.testing import RefBackend

    e = build_engine(
        cfg=tiny(),
        model=build_random(tiny(), seed=11),
        backend=RefBackend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        prefix_store=NoPrefixStore(),
        sparse_k=2,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
    )
    # 10 pages of context: decode selection then drops/resurrects far pages.
    prompt = np.arange(7, 7 + 10 * BLOCK_TOKENS, dtype=np.int64)
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=128, seed=0))
    # Step until the request is in decode (phase 2) with a populated resident set,
    # but stop well before it finishes so the tracker is still attached.
    for _ in range(600):
        if (any(x.req_id == rid and x.phase == 2 for x in e._running)
                and e._sparse.tracker.resident.get(rid)):
            break
        d = e.poll()
        if rid in d and len(d[rid]) >= 128:
            break
        e.step()
    return e, rid


def _captured_sf(rows):
    # device_select=True + faked cuda: the captured-tick branch. selected_pages
    # must NOT be called on this path; if it is, it loudly returns a wrong set.
    def boom(bi):
        raise AssertionError("captured tick must not read selected_pages() to host")

    return SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        device_select=True,
        rows=rows,
        selected_pages=boom,
    )


def _eager_sf(rows, keep_pages):
    def selected_pages(bi):
        return set(keep_pages[bi])

    return SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        device_select=False,
        rows=rows,
        selected_pages=selected_pages,
    )


def test_captured_tick_defers_reconcile_eager_tick_demotes_once():
    e, rid = _decode_engine()
    try:
        sp: SparseRuntime = e._sparse
        tr = sp.tracker
        pool = e._kv
        assert rid in tr.resident, "precondition: a live decode request with residency"
        r = next(x for x in e._running if x.req_id == rid)
        live_before = dict(tr.resident[rid])
        assert len(live_before) >= 3, live_before
        pages = sorted(live_before)
        # q_hi on a block boundary keeps complete == q_hi//BLOCK and bounds_count
        # already covers every resident page, so finalize's bounds-store loop is a
        # no-op and only the pin section is under test.
        q_hi = (r.seq_len - 1 + BLOCK_TOKENS) // BLOCK_TOKENS * BLOCK_TOKENS
        srows = [{"req": r, "req_id": rid, "q_hi": q_hi}]
        rows = [r]  # finalize iterates _Req objects; geometry lives on sf.rows

        # --- captured CUDA tick: must NOT demote/offer/reorder -----------------
        blocks_before = list(r.blocks)
        d0, p0 = pool.cold.demotions, pool.cold.promotions
        offers = sp.finalize(_captured_sf(srows), rows)
        assert offers == [], "captured tick published drops"
        assert pool.cold.demotions == d0 and pool.cold.promotions == p0, (
            "captured tick moved cold pages"
        )
        assert dict(tr.resident[rid]) == live_before, "captured tick changed live set"
        assert list(r.blocks) == blocks_before, "captured tick reordered blocks"

        # --- following EAGER tick reconciles: drop all but the window ----------
        keep = [pages[-2:]]  # keep only the two newest pages
        d1 = pool.cold.demotions
        offers = sp.finalize(_eager_sf(srows, keep), rows)
        dropped_rows = [p for _r, ps in offers for p in ps]
        expected_dropped = [p for p in pages if p not in set(keep[0])]
        assert sorted(dropped_rows) == sorted(expected_dropped), (dropped_rows, expected_dropped)
        # each dropped page demoted exactly once
        assert pool.cold.demotions - d1 == len(expected_dropped)
        # live == selected, blocks in logical order, dropped pages tracked cold
        assert set(tr.resident[rid]) == set(keep[0])
        phys = {p: live_before[p] for p in keep[0]}
        assert list(r.blocks) == [phys[p] for p in sorted(phys)]
        assert set(r.cold_pages) >= set(expected_dropped)
    finally:
        e.shutdown()


def test_multiple_captured_ticks_then_one_refresh_leaks_no_pin():
    e, rid = _decode_engine()
    try:
        sp: SparseRuntime = e._sparse
        tr = sp.tracker
        r = next(x for x in e._running if x.req_id == rid)
        live0 = dict(tr.resident[rid])
        pages = sorted(live0)
        q_hi = (r.seq_len - 1 + BLOCK_TOKENS) // BLOCK_TOKENS * BLOCK_TOKENS
        srows = [{"req": r, "req_id": rid, "q_hi": q_hi}]
        rows = [r]

        # several captured ticks back to back: residency untouched each time
        for _ in range(3):
            assert sp.finalize(_captured_sf(srows), rows) == []
            assert dict(tr.resident[rid]) == live0
        # one eager refresh collapses to the newest page: nothing pinned twice,
        # every non-kept page demoted exactly once.
        keep = [[pages[-1]]]
        d1 = e._kv.cold.demotions
        offers = sp.finalize(_eager_sf(srows, keep), rows)
        dropped = sorted(p for _r, ps in offers for p in ps)
        assert sorted(set(dropped)) == dropped, "a page offered more than once"
        assert dropped == pages[:-1]
        assert e._kv.cold.demotions - d1 == len(pages) - 1
        assert set(tr.resident[rid]) == {pages[-1]}
    finally:
        e.shutdown()
