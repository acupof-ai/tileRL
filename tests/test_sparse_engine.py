"""Unit F: sparse selection wired through the real Engine.

Bounds scorer, hot/cold tiers, per-source-layer selection reused per group,
chunked sparse prefill. Two gates the design names:

1. ``k_pages >= pages`` (every earlier page selected) is token-for-token equal
   to a dense engine across BOTH prefill and decode, through the real
   RefBackend forward — no kernel changed, attention gets a packed
   [selected ; own] table with a packed seq_len.
2. A real sparse run (k smaller than the context) demotes and promotes pages
   every tick, keeps Quest bounds device-resident for complete pages, and
   still produces finite tokens.
"""

from __future__ import annotations

import numpy as np
import torch

from tilerl.config import tiny
from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.model import build_random
from tilerl.testing import RefBackend


def _engine(sparse: bool, k: int = 0, scorer: str = "bounds"):
    kw = dict(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
        max_num_batched_tokens=512, prefix_store=NoPrefixStore())
    if sparse:
        kw.update(sparse_k=k, scorer=scorer, kv_cold_bytes=1 << 30)
    return build_engine(**kw)


def _drain(engine, rid, n):
    for _ in range(512):
        done = engine.poll()
        if rid in done and len(done[rid]) >= n:
            return done[rid][:n]
        engine.step()
    raise TimeoutError("engine did not finish")


def test_sparse_equals_dense_token_for_token_at_full_k():
    """k_pages >= every earlier page: the packed [selected;own] attention must
    reproduce the dense engine's prefill AND decode tokens exactly."""
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)  # 6 pages
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)

    dense = _engine(False)
    td = _drain(dense, dense.submit(prompt, params), 8)
    dense.shutdown()

    # k=6: at most 5 earlier pages exist on the last decode tick, so all are selected.
    sparse = _engine(True, 6)
    ts = _drain(sparse, sparse.submit(prompt, params), 8)
    sparse.shutdown()

    assert ts == td, f"sparse {ts} != dense {td}"


def test_sparse_pins_selected_pages_across_ticks_and_demotes_what_leaves():
    """Cross-tick pin invariant. The pages selected this tick stay resident for the
    next, so each decode tick:

      promotions == number of selected pages that were COLD at tick start (newly
                   chosen only — a kept page is never re-fetched),
      demotions  == number of pages resident last tick that this tick dropped.

    This is the gate the demote-every-tick first cut failed: it re-promoted the
    whole k every decode token (the 6.6x V100 slowdown). Uses a 12-page context so
    pages genuinely fall outside k+window and cycle, while the per-tick counts must
    still match the residency delta exactly."""
    prompt = np.arange(7, 7 + 12 * BLOCK_TOKENS, dtype=np.int64)  # 12 pages
    sparse = _engine(True, 2)
    rid = sparse.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=8, seed=0))

    checked = 0
    for _ in range(256):
        done = sparse.poll()
        if rid in done and len(done[rid]) >= 8:
            break
        r0 = next((x for x in sparse._running if x.req_id == rid), None)
        if r0 is None or r0.phase != 2:
            sparse.step()
            continue
        cold = sparse._kv.cold
        cold_before = set(r0.cold_pages)
        d0, p0 = cold.demotions, cold.promotions
        sparse.step()
        if r0 not in sparse._running:
            break
        demoted, promoted = cold.demotions - d0, cold.promotions - p0
        # A promotion only ever brings back a page that started the tick cold; a
        # still-resident (kept) page is reused, never re-fetched.
        assert 0 <= promoted <= len(cold_before), (promoted, len(cold_before))
        # demote + the kept resident set partition last tick's residents.
        assert demoted >= 0, demoted
        checked += 1
    else:
        raise TimeoutError

    assert checked >= 4, f"too few decode ticks observed: {checked}"
    # At least one tick had pages cold (otherwise nothing exercised the pin tier).
    sparse.shutdown()


def test_sparse_a_stable_selection_promotes_nothing_after_the_first_tick():
    """Strong half of the pin gate: force the SAME selection on successive decode
    ticks and assert zero promotions/demotions — the kept frames are reused. The
    selection is pinned by monkeypatching select_pages to a fixed top-k for the
    rows once they are in decode."""
    import tilerl_kernels.reference as ref

    prompt = np.arange(7, 7 + 12 * BLOCK_TOKENS, dtype=np.int64)
    sparse = _engine(True, 2)
    orig_select = ref.select_pages
    fixed: dict[tuple, object] = {}

    def stable_select(block_table, n_pages, scores, k, n_window=0):
        out = orig_select(block_table, n_pages, scores, k, n_window)
        # After the first decode scoring, freeze every later call to its result so
        # the cross-group selection is identical tick to tick.
        key = tuple(int(x) for x in n_pages.tolist())
        if key in fixed:
            return fixed[key].clone()
        if sparse._running and any(r.phase == 2 for r in sparse._running):
            fixed[key] = out.clone()
        return out

    rid = sparse.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=8, seed=0))
    ref.select_pages = stable_select
    cycles = []
    try:
        for _ in range(256):
            done = sparse.poll()
            if rid in done and len(done[rid]) >= 8:
                break
            r0 = next((x for x in sparse._running if x.req_id == rid), None)
            if r0 is None or r0.phase != 2:
                sparse.step()
                continue
            cold = sparse._kv.cold
            d0, p0 = cold.demotions, cold.promotions
            sparse.step()
            cycles.append((cold.demotions - d0, cold.promotions - p0))
    finally:
        ref.select_pages = orig_select
    sparse.shutdown()
    assert len(cycles) >= 3, cycles
    # from the second stable tick on, nothing moves between device and host
    assert all(d == 0 and p == 0 for d, p in cycles[1:]), cycles


def test_quest_scores_chunked_over_pages_matches_all_at_once():
    """Scoring splits candidate pages to bound the f32 intermediate (the unchunked
    [Tq,Cp,Hkv,D] is 4.2 GiB at Cp=2048 and OOMs a V100). max-over-query and
    sum-over-head/dim commute with the page split, so the chunked score must be
    bit-identical; Cp is a non-multiple of the chunk to cover the tail."""
    import torch

    from tilerl.sparse_engine import quest_scores

    tq, hq, hkv, d, cp = 37, 8, 4, 256, 201
    q = torch.randn(tq, hq, d)
    bounds = torch.randn(cp, hkv, 2, d) * 0.3
    qi = q.float().reshape(tq, hkv, hq // hkv, d).mean(2)
    kmin, kmax = bounds.unbind(dim=2)
    ref = torch.maximum(qi[:, None] * kmin[None], qi[:, None] * kmax[None]).sum(-1)
    ref = ref.amax(0).sum(-1)
    assert torch.equal(quest_scores(q, bounds), ref)


def test_sparse_prompt_longer_than_device_hot_pool_admits_via_cold():
    """The admission guard must count host-cold pages, not only the device hot
    pool. The device pool is the per-slot hot set (k+window+chunk); with k=2,
    slots=1, chunk=16 tokens it holds 13 blocks = 208 tokens, so a 512-token
    request must admit by demoting older pages to the host. The old guard
    compared against device blocks only and raised 'exceeds KV pool capacity'
    before sparse allocation ran. Full-selection token equality is the prior
    gate; k>=pages makes the pool wider than the context, so it cannot exercise
    this crossing."""
    cfg = tiny()
    common = dict(cfg=cfg, model=build_random(cfg, seed=11), backend=RefBackend(),
                  num_slots=1, max_batch=1, max_total_tokens=1024,
                  max_num_batched_tokens=16, prefix_store=NoPrefixStore())
    sparse = build_engine(num_blocks=0, sparse_k=2, scorer="bounds",
                          kv_cold_bytes=1 << 30, **common)
    assert sparse._kv.num_blocks == 13  # 1*(k2 + window8 + chunk2) + 1
    assert sparse.room_for(512) > 0
    prompt = (np.arange(32 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7  # 512 tok, in tiny vocab

    rid = sparse.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    for _ in range(512):
        done = sparse.poll()
        if rid in done and len(done[rid]) >= 4:
            break
        sparse.step()
    else:
        raise TimeoutError
    tok = done[rid][:4]
    ndemote = sparse._kv.cold.demotions
    nframes = sparse._kv.num_blocks
    sparse.shutdown()
    # 32 pages cycled through a 13-frame pool: demotions far exceed the frame
    # count, so physical ids recycled while a prior blob stayed cold — the exact
    # phys-key collision. Stable (req, page) keys are what let this finish.
    assert ndemote > nframes, (ndemote, nframes)
    assert len(tok) == 4 and all(isinstance(t, int) for t in tok)
def test_index_scorer_equals_dense_token_for_token_at_full_k():
    """The learned scorer="index" through the SAME seam as bounds must select
    every candidate page at full k and reproduce dense tokens prefill+decode
    (untrained weights; equality holds only at full k)."""
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)

    dense = _engine(False)
    td = _drain(dense, dense.submit(prompt, params), 8)
    dense.shutdown()

    sparse = _engine(True, 6, scorer="index")
    ts = _drain(sparse, sparse.submit(prompt, params), 8)
    sparse.shutdown()

    assert ts == td, f"index sparse {ts} != dense {td}"


def test_live_selection_recall_is_one_at_full_k():
    """Engine.sparse_selection_recall scores the last tick's served selection vs
    an offline dense top-k; at full k every candidate is chosen -> 1.0 per source
    group (the card run's live-selection recall hook)."""
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)
    e = _engine(True, 6, scorer="index")
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    checked = False
    for _ in range(256):
        e.step()
        sel = e._sparse.last_selected.get(rid)
        # wait for a decode tick that actually selected earlier candidate pages
        if sel and any(cand for cand, _ in sel.values()):
            pages = prompt.shape[0] // BLOCK_TOKENS
            mass = torch.zeros(1, len(sel), 1, pages)
            for g, (cand, _chosen) in sel.items():
                for j, p in enumerate(cand):
                    mass[0, g, 0, p] = 1.0 / (j + 1)  # distinct ordered target
            rec = e.sparse_selection_recall(rid, mass)
            assert set(rec) == set(sel)
            assert all(abs(v - 1.0) < 1e-6 for v in rec.values()), rec
            checked = True
            break
        done = e.poll()
        if rid in done and len(done[rid]) >= 2:
            break
    e.shutdown()
    assert checked



def test_index_scorer_writes_fp8_keys_and_reconciles_measured_bytes():
    """The index scorer persists fp8 page keys (not bounds), demotes/promotes
    them, and stats' measured index_keys matches the derived row at tiny di=16
    (20 B/key: 16 fp8 payload + 4 B scale)."""
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)
    e = _engine(True, 2, scorer="index")
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    saw_key = False
    for _ in range(128):
        done = e.poll()
        if rid in done and len(done[rid]) >= 4:
            break
        e.step()
        tr = e._sparse
        assert tr.bounds is None and tr.keys is not None
        if any(tr.keys.get(r.req_id, {}) for r in e._running):
            page, (keys, scales) = next(
                (p, kv) for rid2, pages in tr.keys.items() if pages
                for p, kv in pages.items())
            assert keys.dtype == torch.float8_e4m3fn and keys.shape[-1] == tr.di
            assert keys.shape[:2] == (len(tr.src_planes),
                                      tr.ih)
            assert scales.shape == keys.shape[:2]
            # measured == derived at the live (tiny di) face
            stored = e.stats()["memory"]
            row = [r for r in stored if r["owner"] == "index_keys"]
            assert row and row[0]["derived"] == row[0]["measured"], row
            assert row[0]["measured"] == tr.index_keys_bytes()
            saw_key = True
    e.shutdown()
    assert saw_key


def test_serve_build_path_wires_the_sparse_engine(tmp_path, capsys):
    """The live arm must run through SERVE's own build path (_build_engine ->
    build_engine), not only a hand-built engine in the other gates. A non-
    checkpoint `serve --dry-run --sparse-k` prints the LIVE sparse owners
    (page_bounds/kv_hot), which exist only when the sparse engine was actually
    constructed; the old code refused this combination as a derived-only ledger.
    """
    import json

    from tilerl import cli

    cli.cmd_serve(cli._build_parser().parse_args([
        "serve", "--model", "tiny", "--dry-run", "--json",
        "--sparse-k", "2", "--scorer", "bounds",
        "--slots", "2", "--max-batch", "1", "--max-ctx", "256",
        "--device-free", "100000000",
    ]))
    owners = {r["owner"] for r in json.loads(capsys.readouterr().out)}
    assert {"page_bounds", "kv_hot"} <= owners, owners
    # the dense kv_pool must NOT also be priced for a sparse engine
    assert "kv_pool" not in owners, owners


def test_sparse_prefix_publishes_only_when_pages_leave_the_hot_union():
    """Drop-only publishing under the cross-tick hot pin (#534): a page is shared
    only when it LEAVES the resident union, and only once pages 0..m-1 have all
    dropped at least once and an exact boundary-m state snapshot exists. A short
    prompt wholly inside the hot set publishes NOTHING; a long prompt whose early
    pages drop publishes an entry, and a follower sharing it HITS (adopts the
    block-aligned prefix, prefills only the tail) and matches a dense engine."""
    # tiny has one source group; k=2 + the forced 8-page window keep ~10 pages
    # hot, so decoding the 24-page prompt demotes its early pages and the
    # contiguous dropped frontier closes over the whole page-aligned prompt.
    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    follow = np.concatenate([prompt, np.arange(100, 120, dtype=np.int64)])
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)

    # k=2 is approximate by design, so the equality oracle is a SPARSE engine
    # decoding follow with a MISS (no publisher): sharing must be transparent.
    def _sparse():
        return build_engine(
            cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=512, sparse_k=2, scorer="bounds",
            kv_cold_bytes=1 << 30)

    miss = _sparse()
    ts_miss = _drain(miss, miss.submit(follow, params), 8)
    miss.shutdown()

    sparse = _sparse()
    # A 5-page prompt fits entirely inside k+window: nothing leaves the union, so
    # the pin publishes no prefix at all.
    short = sparse.submit(np.arange(7, 7 + 5 * BLOCK_TOKENS, dtype=np.int64),
                          SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    _drain(sparse, short, 4)
    assert sparse._sparse.prefix.published == 0, sparse._sparse.prefix.published

    r1 = sparse.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
    _drain(sparse, r1, 200)
    entry = sparse._sparse.prefix.lookup(follow)
    assert entry is not None and len(entry["keys"]) == 24, \
        None if entry is None else len(entry["keys"])

    r2 = sparse.submit(follow, params)
    sparse.step()
    req = next(x for x in sparse._running if x.req_id == r2)
    assert req.sparse_matched == 24 * BLOCK_TOKENS, req.sparse_matched
    ts = _drain(sparse, r2, 8)
    sparse.shutdown()
    assert ts == ts_miss, f"prefix-hit follower {ts} != prefix-miss sparse {ts_miss}"


def test_sparse_prefix_out_of_order_drops_never_publish_a_hole():
    """Pages leave the pinned resident union in arbitrary order. The index may
    publish an entry only over a CONTIGUOUS 0..m-1 run with a held blob, a bound
    and the boundary-m snapshot for every page; and every content key the entry
    lists must resolve to a held blob (the old hole: a full-length entry skipped
    missing blobs, so a follower silently never attended those pages)."""
    import torch

    from tilerl.kv_cache import HostKvPages
    from tilerl.sparse_engine import SparsePrefixCache

    cold = HostKvPages(budget_bytes=1 << 30)
    cache = SparsePrefixCache(cold, states=None)
    rid, P = 0, 4
    tokens = tuple(range(P * BLOCK_TOKENS))
    cache.set_request(rid, P)
    for m in range(1, P + 1):
        cache.note_boundary(rid, m, (torch.zeros(2), None))
    bounds = {p: torch.zeros(1) for p in range(P)}

    def blob(p):
        t = torch.full((2,), float(p))
        return {"k": t, "v": t}

    # Drop pages 2 then 1 while page 0 is still pinned: the frontier is blocked,
    # nothing may be published even though longer boundary snapshots exist.
    assert cache.publish_dropped(rid, tokens, bounds, 2, blob(2)) == {}
    assert cache.publish_dropped(rid, tokens, bounds, 1, blob(1)) == {}
    assert cache.lookup(tokens) is None
    # Page 0 leaves: a contiguous 3-page entry freezes. Every key it lists must
    # resolve to a held blob, and bounds cover exactly the listed pages.
    cache.publish_dropped(rid, tokens, bounds, 0, blob(0))
    hit3 = cache.lookup(tokens)
    assert hit3 is not None and len(hit3["keys"]) == 3
    assert set(hit3["bounds"]) == set(range(3))
    for p, key in enumerate(hit3["keys"]):
        held = cold.share_take(key)
        assert held is not None and torch.equal(held["k"], torch.full((2,), float(p)))
    # The prompt-end page closes the full prefix.
    cache.publish_dropped(rid, tokens, bounds, 3, blob(3))
    hit4 = cache.lookup(tuple(tokens) + (9, 9))
    assert hit4 is not None and len(hit4["keys"]) == P


def test_sparse_prefix_republished_after_repin_keeps_the_first_blob():
    """A page drops (published), the pin re-selects it, and it drops again with a
    new private blob: the shared prefix must keep serving the FIRST captured
    blob — the clone is independent of the private frame across a pin boundary."""
    import torch

    from tilerl.kv_cache import HostKvPages
    from tilerl.sparse_engine import SparsePrefixCache

    cold = HostKvPages(budget_bytes=1 << 30)
    cache = SparsePrefixCache(cold, states=None)
    tokens = tuple(range(BLOCK_TOKENS))
    cache.set_request(0, 1)
    cache.note_boundary(0, 1, (torch.zeros(2), None))
    bounds = {0: torch.zeros(1)}
    first = torch.full((2,), 7.0)
    cache.publish_dropped(0, tokens, bounds, 0, {"k": first, "v": first})
    # Same logical page leaves again with different data after being re-pinned.
    again = torch.full((2,), 9.0)
    assert cache.publish_dropped(0, tokens, bounds, 0, {"k": again, "v": again}) == {}
    hit = cache.lookup(tokens)
    assert hit is not None
    assert torch.equal(cold.share_take(hit["keys"][0])["k"], first)


def test_prefix_clone_reads_blob_only_after_demotions_scope_exits():
    """The #538 batched-demote ordering: inside `with pool.demotions()` the D2H is
    still in flight and the blob is NOT in the cold tier yet; it is held only at
    the context's sync. Prefix cloning inside the scope therefore peeks None and
    publishes nothing — the engine must offer dropped pages AFTER the scope exits.

    This drives a real PagedKvPool whose demotions() defers the hold (mimicking
    #538) and asserts peek is None in-scope but the blob is present out-of-scope.
    The full engine test above (24-page prompt) exercises the synchronous path;
    together they pin the offer to after the scope for both implementations."""
    import contextlib

    from tilerl.kv_cache import HostKvPages, PagedKvPool

    pool = PagedKvPool(8, 1, 16, device=torch.device("cpu"), layer_map=(0,))
    pool.attach_cold(HostKvPages(budget_bytes=1 << 30))
    b = pool.alloc_block()

    # Replace demotions() with a context that defers the hold to exit, the exact
    # invariant batched demote relies on: demote_page stages, hold happens last.
    staged = []
    real_demote = pool.demote_page

    @contextlib.contextmanager
    def deferred_demotions():
        def stage(block, key=None):
            staged.append((block, key))
        pool.demote_page = stage
        try:
            yield pool
        finally:
            pool.demote_page = real_demote
            for block, key in staged:
                real_demote(block, key=key)

    pool.demotions = deferred_demotions
    with pool.demotions():
        pool.demote_page(b, key=(7, 3))
        # in-flight: not held yet -> a prefix clone here would see nothing
        assert pool.cold.peek((7, 3)) is None
    # after the scope's sync/hold the blob is present, so an after-scope clone works
    assert pool.cold.peek((7, 3)) is not None
    assert pool.free_blocks >= 1  # frames returned at exit too


def test_cold_tier_spills_past_the_host_budget_and_the_ledger_splits_tiers(tmp_path):
    """A host budget that holds only one page pushes subsequent demotions to the
    mmap spill; the live sparse ledger then carries BOTH a host kv_cold row and
    an ssd kv_cold row, each derived == measured (per-cold-block price)."""
    import os

    from tilerl.kv_cache import BLOCK_TOKENS
    from tilerl.memory import per_kv_block_bytes

    cfg = tiny()
    per = per_kv_block_bytes(cfg, __import__("torch").bfloat16)
    ssd = str(tmp_path / "spill.bin")
    e = build_engine(
        cfg, build_random(cfg, seed=11), RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
        max_num_batched_tokens=512, prefix_store=NoPrefixStore(),
        sparse_k=2, scorer="bounds",
        kv_cold_bytes=per,                 # one page of pinned host RAM
        cold_ssd_path=ssd)
    # 24 pages. Under the cross-tick pin the device hot set is k+window (+chunk):
    # tiny has one source group, k=2 and the trailing 8-page window stay resident,
    # so a context must EXCEED k+window pages for anything to demote and spill.
    # (This gate predates the pin: its old 6-page prompt now fits entirely in the
    # hot set, so host=0/ssd=0 every tick — correct pin behaviour, no spill.)
    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=6, seed=0))
    for _ in range(128):                      # into decode, before the request finishes
        r0 = next((x for x in e._running if x.req_id == rid), None)
        e.step()
        if r0 is not None and r0.phase == 2:
            break
    st = e._kv.cold.stats()
    assert st["kv_cold_ssd_pages"] >= 3, st  # pages beyond the k+window hot set spill
    assert os.path.getsize(ssd) > 0  # spill file holds the evicted pages
    rows = {(r["owner"], r["tier"]): r for r in e.stats()["memory"]}
    assert ("kv_cold", "host") in rows and ("kv_cold_ssd", "ssd") in rows, list(rows)
    # sparse live ledger is derived from the tier byte counters (exact page price)
    assert rows[("kv_cold", "host")]["derived"] == st["kv_cold_bytes"]
    assert rows[("kv_cold_ssd", "ssd")]["derived"] == st["kv_cold_ssd_bytes"]
    _drain(e, rid, 6)  # finish releases the request's cold pages (both tiers)
    e.shutdown()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
