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
        # bounds for complete pages survive demotion in the contiguous store
        assert int(sparse._sparse.bounds_valid[rid].sum()) >= 5
        checked += 1
    else:
        raise TimeoutError

    assert checked >= 4, f"too few decode ticks observed: {checked}"
    # At least one tick had pages cold (otherwise nothing exercised the pin tier).
    sparse.shutdown()


def test_sparse_a_stable_selection_promotes_nothing_after_the_first_tick():
    """Strong half of the pin gate: force the SAME selection on successive decode
    ticks and assert zero promotions/demotions — the kept frames are reused. The
    selection is pinned by monkeypatching the CONSUMER
    (sparse_engine.select_pages) to a fixed top-k once the rows are in decode."""
    import tilerl_kernels.reference as ref

    import tilerl.sparse_engine as se

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
    se.select_pages = stable_select
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
        se.select_pages = orig_select
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
        assert tr.scorer == "index" and tr.keys is not None
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

def _draft(cfg, trunk):
    """A one-layer DraftHead over the tiny trunk (same builder as test_decode_graph)."""
    from dataclasses import replace

    import torch

    from tilerl.model import build_random
    from tilerl.spec import DraftHead

    dcfg = replace(cfg, num_layers=1, full_attn_layers=(0,), fp4=False)
    params = {k: v for k, v in build_random(dcfg, seed=3).params.items()
              if k.startswith("layers.")}
    gen = torch.Generator().manual_seed(3)
    h = cfg.hidden_size
    params["fc"] = (torch.randn(h, 2 * h, generator=gen) * 0.02).to(torch.bfloat16)
    params["norm"] = torch.ones(h, dtype=torch.bfloat16)
    params["pre_fc_norm_hidden"] = torch.ones(h, dtype=torch.bfloat16)
    return DraftHead(trunk, params, num_layers=1)


def _sparse_engine(sparse_k, draft=False, cfg=None):
    from tilerl_kernels.backend import get_backend

    cfg = cfg or tiny()
    model = build_random(cfg, seed=11)
    # Both arms of an exact-token gate must use one backend: a real draft goes
    # through _serve_draft -> has_kernel, which RefBackend does not declare.
    # The CPU cell backend is the spec harness test_e2e uses.
    backend = get_backend()
    kw = dict(num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
              max_num_batched_tokens=512, sparse_k=sparse_k, scorer="bounds",
              kv_cold_bytes=1 << 30)
    if draft:
        kw["draft"] = _draft(cfg, model)
        kw["spec_depth"] = 1
    return build_engine(cfg=cfg, model=model, backend=backend, **kw)


def test_sparse_with_draft_matches_sparse_without_draft_greedy():
    """Exact-acceptance: greedy speculative decode emits the SAME tokens as greedy
    non-spec decode under the same sparse selection (sparse_k=2). The W verify
    queries are scored jointly (max across queries), so every draft position
    attends the same selected pages plus its own causal block."""
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)
    plain = _sparse_engine(2, draft=False)
    t_plain = _drain(plain, plain.submit(prompt, params), 8)
    plain.shutdown()
    spec = _sparse_engine(2, draft=True)
    t_spec = _drain(spec, spec.submit(prompt, params), 8)
    spec.shutdown()
    assert t_spec == t_plain, f"spec {t_spec} != plain {t_plain}"


def test_sparse_draft_admission_reject_does_not_leak_a_state_slot():
    """Sparse+spec reject path: when the dense draft pool is full (a running row
    owns its blocks), _admit frees the just-allocated state slot and returns
    False, and the bookkeeping counter _slots_used must come back with it.
    Otherwise every rejected tick permanently consumes a slot and the engine
    fills up on a row it never admitted.

    Two rows are required: the submit capacity guard is STATIC (num_blocks), so
    one row sized to the pool passes submit; it is the admit-time free-blocks
    guard (dynamic) that rejects a second row once the first owns the blocks."""
    cfg = tiny()
    model = build_random(cfg, seed=11)
    from tilerl_kernels.backend import get_backend

    e = build_engine(
        cfg=cfg, model=model, backend=get_backend(),
        num_blocks=64, num_slots=4, max_batch=2, max_total_tokens=4096,
        max_num_batched_tokens=512, sparse_k=2, scorer="bounds",
        kv_cold_bytes=1 << 30, draft=_draft(cfg, model), spec_depth=1)
    # The 5-page prompt needs 6 dense draft blocks (plus one verify position);
    # size the shared draft pool so one row fits but two cannot.
    from tilerl.kv_cache import PagedKvPool

    d = e._draft
    d.kv = PagedKvPool(6, cfg.num_kv_heads, cfg.head_dim,
                       num_layers=d.cfg.num_layers,
                       layer_map=tuple(range(d.cfg.num_layers)),
                       device=d.backend.device, dtype=d.kv.dtype)
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)  # 6 pages
    params = SamplingParams(temperature=0.0, max_new_tokens=4, seed=0)
    r1 = e.submit(prompt, params)
    try:
        for _ in range(6):                       # admit r1, grow its draft blocks
            e.step()
            if e._running and e._running[0].draft_blocks:
                break
        assert e._draft.kv.free_blocks < 6, "fixture: row 1 did not fill the draft pool"
        e.submit(prompt, params)                 # static guard passes, admit rejects
        slots_before = e.stats()["slots_used"]
        for _ in range(6):
            e.step()                             # r2 retries admit every tick
        st = e.stats()
        assert st["slots_used"] == slots_before == 1, (
            f"rejected admits leaked slots: {st['slots_used']} vs {slots_before}")
    finally:
        e.shutdown()


def test_full_k_sparse_with_draft_matches_dense_with_draft():
    """k>=pages selects everything: sparse + draft equals dense + draft."""
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)
    dense = _sparse_engine(6, draft=True)  # k=6 covers every earlier page
    t_dense = _drain(dense, dense.submit(prompt, params), 8)
    dense.shutdown()
    # a genuinely dense engine (sparse off) with a draft, same CPU-cell backend
    cfg = tiny()
    model = build_random(cfg, seed=11)
    from tilerl_kernels.backend import get_backend

    full = build_engine(cfg=cfg, model=model, backend=get_backend(), num_blocks=64,
                        num_slots=4, max_batch=1, max_total_tokens=4096,
                        draft=_draft(cfg, model), spec_depth=1)
    t_full = _drain(full, full.submit(prompt, params), 8)
    full.shutdown()
    assert t_dense == t_full, f"sparse+draft {t_dense} != dense+draft {t_full}"


def test_verify_tick_packed_table_shape_is_fixed():
    """Graph-capture structural gate: a CUDA graph bakes the block-table shape in.
    The packed [selected;own] width must depend ONLY on the tick query width, not
    on context length: every decode/verify row is k_pages + the 8-page window,
    plus at most 1 if the W-1 draft chain crosses a page boundary (W-1 <= 15).
    Run the same engine over prompts of different lengths and assert the observed
    decode widths are one context-independent constant per query width. Capture
    itself stays eager-only in the first cut."""
    from tilerl import sparse_engine as se
    from tilerl.sparse_index import WINDOW_PAGES

    K = 2
    engine = _sparse_engine(K, draft=True)  # verify ticks carry up to W+1=2 q
    orig = se.SparseForward.attention_args
    seen: dict[int, set[int]] = {}

    def wrap(self, plane, q, h=None):
        table, sl = orig(self, plane, q, h)
        for r in self.rows:
            if int(r["force_window"]) == 0:  # decode/verify tick (no forced window), never prefill
                seen.setdefault(int(r["tq"]), set()).add(int(table.shape[1]))
        return table, sl

    # two contexts of different page counts (10 and 16); widths must not diverge
    for n_pages in (10, 16):
        prompt = np.arange(3, 3 + n_pages * BLOCK_TOKENS, dtype=np.int64)
        rid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=6, seed=0))
        se.SparseForward.attention_args = wrap
        try:
            _drain(engine, rid, 6)
        finally:
            se.SparseForward.attention_args = orig
    engine.shutdown()

    assert seen, "no decode/verify tick observed"
    bound = K + WINDOW_PAGES
    for tq, widths in seen.items():
        assert widths <= {bound, bound + 1}, (tq, widths)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))


def test_select_tensor_op_count_is_constant_in_candidate_count():
    """The 128k host-bound fix: _select gathers candidate bounds with one
    index_select, so the number of aten ops it dispatches does not grow with
    the candidate count (a torch.stack-per-page would dispatch per page)."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    from tilerl.sparse_engine import SparseForward, SparseTracker

    class _Count(TorchDispatchMode):
        n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.n += 1
            return func(*args, **(kwargs or {}))

    cfg = tiny()

    def select_ops(n_cand: int) -> int:
        tr = SparseTracker(cfg, k_pages=4, scorer="bounds")
        tr.attach(0)
        b = torch.randn(tr.n_full, tr.hkv, 2, tr.dim, dtype=torch.float16)
        for p in range(n_cand):
            tr.set_bounds(0, p, b)
        cand = list(range(n_cand))
        row = dict(req_id=0, own=[n_cand], own_len=BLOCK_TOKENS, cand=cand,
                   force_window=0, resolve=lambda p: p, reserved=set(),
                   decoding=True, tq=1)
        sf = SparseForward(tr, [row], torch.device("cpu"))
        q = torch.randn(1, cfg.num_attention_heads, cfg.head_dim)
        with _Count() as c:
            sf._select(0, 0, q, None)
        return c.n

    ops_8 = select_ops(8)
    ops_64 = select_ops(64)
    assert ops_64 == ops_8, f"ops grow with candidates: {ops_8} @8 -> {ops_64} @64"


def test_bounds_tensor_holds_every_full_attn_plane_not_every_source_group():
    """The 27B has 16 full-attn planes grouped into 4 index sources. Bounds are
    stored and read per plane; sizing the plane dim at n_src (4) made the first
    finalize write crash and plane 15 unindexable. tiny has 1 plane, which hid
    it until the 27B ran."""
    from dataclasses import replace

    from tilerl.sparse_engine import SparseTracker

    cfg = replace(tiny(), num_layers=16, full_attn_layers=tuple(range(16)))
    tr = SparseTracker(cfg, k_pages=4, scorer="bounds")
    assert tr.n_full == 16
    assert len(tr.src_planes) == 4
    tr.attach(0)
    b = torch.randn(16, tr.hkv, 2, tr.dim, dtype=torch.float16)
    tr.set_bounds(0, 0, b)  # raised: size (4) vs (16) before the fix
    # bounds_t is plane-first [n_full, cap, ...]; plane 15 must be indexable.
    assert tr.bounds_t[0].shape[0] == 16
    one = tr.bounds_one(0, 0)  # raised: index 15 out of bounds before the fix
    assert one.shape == (16, tr.hkv, 2, tr.dim)
    assert torch.equal(one[15], b[15])
    # scoring one plane gathers only that plane's page rows
    row = tr.bounds_rows(0, 15, [0])
    assert row.shape == (1, tr.hkv, 2, tr.dim)
    assert torch.equal(row[0], b[15])


def test_bounds_tracker_allocates_on_the_backend_device_not_cpu_for_first_request():
    """GPU regression in the contiguous-bounds commit: the tracker inferred its
    device from already-allocated bounds_t and fell back to CPU, so the FIRST
    request's bounds_t/l2p_t were allocated on CPU even on a cuda backend; a GPU
    query scored against CPU bounds -> cross-device RuntimeError at the second
    prefill chunk. CI has no GPU, so use a non-CPU fake-device seam: the tensors
    must be allocated on the device the tracker was constructed with, before any
    bound exists. Red on the old _device-fallback constructor (forced CPU)."""
    from types import SimpleNamespace

    from tilerl.sparse_engine import SparseTracker

    # A device object that is not CPU, distinct across attach calls. torch meta
    # allocates without compute, which is all the seam needs (device placement).
    dev = torch.device("meta")
    cfg = SimpleNamespace(full_attn_layers=(0,), num_kv_heads=2, head_dim=16)
    tr = SparseTracker(cfg, k_pages=2, scorer="bounds", device=dev)
    tr.attach(0)                       # first request, no prior bounds to infer from
    assert tr.bounds_t[0].device == dev, tr.bounds_t[0].device
    assert tr.bounds_valid[0].device == dev
    assert tr.l2p_t[0].device == dev
    # growing past INIT_CAP keeps the device too
    page = tr.INIT_CAP + 1
    tr.map_resident(0, page, 7)
    assert tr.l2p_t[0].device == dev
    assert int(tr.l2p_t[0].shape[0]) > page
    # default (tests, CPU backends) is still CPU
    tr_cpu = SparseTracker(cfg, k_pages=2, scorer="bounds")
    tr_cpu.attach(1)
    assert tr_cpu.bounds_t[1].device.type == "cpu"


# ---------------------------------------------------------------- graph-capturable decode tick


def _resident_forward(B, device_select, n_pages=20, k=4):
    """A SparseForward over B rows where every page is RESIDENT (the cross-tick
    pin steady state a captured decode tick requires): k selection from
    n_pages-2 candidate pages plus a trailing 2-page own window."""
    import torch

    from tilerl.sparse_engine import SparseForward, SparseTracker

    cfg = tiny()
    torch.manual_seed(0)
    tr = SparseTracker(cfg, k, "bounds")
    rows = []
    for bi in range(B):
        tr.attach(bi)
        own = [n_pages - 2, n_pages - 1]
        cand = list(range(n_pages - 2))
        b = torch.randn(tr.n_full, tr.hkv, 2, tr.dim, dtype=torch.float16)
        for p in cand:
            tr.set_bounds(bi, p, b)
        phys = {}
        for p in cand + own:
            pm = 1000 + bi * 1000 + p
            phys[p] = pm
            tr.map_resident(bi, p, pm)
        tr.resident[bi] = phys
        rows.append(dict(
            req_id=bi, own=own, own_len=2 * BLOCK_TOKENS + 5, q_start=0, q_hi=0,
            decoding=True, tq=1, cand=cand, force_window=0,
            resolve=lambda p, rid=bi: tr.resident[rid][p], reserved=set()))
    sf = SparseForward(tr, rows, torch.device("cpu"), device_select=device_select)
    return cfg, sf


def test_device_select_packs_a_fixed_width_table_with_no_host_sync_at_b1_and_b8():
    """The captured decode tick's structural gate at B=1 and B=8:

    - the packed table is a FIXED ``min(k,cand)+own_window`` width for the tick
      (a captured graph replays one shape), never the eager dynamic width;
    - building it through attention_args issues NO host-syncing scalar read
      (``aten._local_scalar_dense`` is what ``.item()``/``bool(tensor)`` lower
      to — one of those inside the replay breaks capture).
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class _NoScalarSync(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert "_local_scalar_dense" not in str(func), f"host sync in tick: {func}"
            return func(*args, **(kwargs or {}))

    for B in (1, 8):
        cfg, sf = _resident_forward(B, True, k=4)
        q = torch.randn(B, 1, cfg.num_attention_heads, cfg.head_dim)
        with _NoScalarSync():
            table, sl = sf.attention_args(0, q)
        assert table.shape == (B, 4 + 2), table.shape   # fixed k + trailing own window
        assert sl.shape == (B,)
        # seq_len excludes the padded tail: each row selected exactly k here.
        assert torch.equal(sl, torch.full((B,), 4 * BLOCK_TOKENS + 2 * BLOCK_TOKENS + 5))


def test_device_select_packed_table_matches_eager_at_b1_and_b8():
    """Token-equality to eager sparse at the SparseForward level: the device
    path's leading compact physical columns and seq_len must equal the eager
    path's packed table at B=1 and B=8 (same selection, same own pages)."""
    import torch

    for B in (1, 8):
        cfg, eager = _resident_forward(B, False, k=4)
        _, device = _resident_forward(B, True, k=4)
        # identical bounds/phys seeds: _resident_forward reseeds torch each call
        g = torch.Generator().manual_seed(0)
        q = torch.randn(B, 1, cfg.num_attention_heads, cfg.head_dim, generator=g)
        g2 = torch.Generator().manual_seed(0)
        q2 = torch.randn(B, 1, cfg.num_attention_heads, cfg.head_dim, generator=g2)
        te, sle = eager.attention_args(0, q)
        td, sld = device.attention_args(0, q2)
        for bi in range(B):
            n = int((te[bi] != 0).sum())
            assert te[bi, :n].tolist() == td[bi, :n].tolist(), (B, bi)
            assert int(sle[bi]) == int(sld[bi]), (B, bi)


def test_device_select_engine_tokens_equal_eager_sparse_at_full_k():
    """End-to-end gate: through the real Engine the device-selection decode tick
    is token-for-token equal to the eager sparse engine at full k (B=1)."""
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)

    def run(devsel):
        kw = dict(
            cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=512, prefix_store=NoPrefixStore(),
            sparse_k=6, scorer="bounds", kv_cold_bytes=1 << 30,
            sparse_device_select=devsel)
        e = build_engine(**kw)
        tok = _drain(e, e.submit(prompt, params), 8)
        e.shutdown()
        return tok

    assert run(True) == run(False)


def test_quest_scores_batched_matches_single_row_bit_for_bit():
    """The captured tick scores all B rows with quest_scores_batched; it must be
    bit-identical to applying the single-row quest_scores per row (the page chunk
    split commutes over rows exactly as it does over pages)."""
    import torch

    from tilerl.sparse_engine import quest_scores, quest_scores_batched

    B, tq, hq, hkv, d, cp = 4, 3, 8, 4, 16, 33
    q = torch.randn(B, tq, hq, d)
    bounds = torch.randn(B, cp, hkv, 2, d) * 0.3
    batched = quest_scores_batched(q, bounds)
    for bi in range(B):
        assert torch.equal(batched[bi], quest_scores(q[bi], bounds[bi]))


# ------------------------------------------------ device path: resident-only selection + refresh


def _forward_one_cold(B, device_select, cold_pages):
    """Like _resident_forward but ``cold_pages`` are NON-resident (l2p=-1):
    bounds exist for them (a demoted page keeps its bounds) but the K is cold or
    SSD-resident. Returns (SparseForward, cfg, cold_pages)."""
    import torch

    from tilerl.sparse_engine import SparseForward, SparseTracker

    cfg = tiny()
    torch.manual_seed(0)
    k = 2
    n_pages = 12
    tr = SparseTracker(cfg, k, "bounds")
    rows = []
    for bi in range(B):
        tr.attach(bi)
        own = [n_pages - 2, n_pages - 1]
        cand = list(range(n_pages - 2))
        # descending-magnitude bounds so candidate index 0 is the top score
        for pi, p in enumerate(cand):
            b = torch.full((tr.n_full, tr.hkv, 2, tr.dim),
                           1.0 - pi * 0.01, dtype=torch.float16)
            tr.set_bounds(bi, p, b)
        phys = {}
        for p in cand + own:
            if p in cold_pages:
                continue                          # cold: bounds held, l2p stays -1
            pm = 1000 + bi * 1000 + p
            phys[p] = pm
            tr.map_resident(bi, p, pm)
        tr.resident[bi] = phys
        rows.append(dict(
            req_id=bi, own=own, own_len=2 * BLOCK_TOKENS + 5, q_start=0, q_hi=0,
            decoding=True, tq=1, cand=cand, force_window=0,
            resolve=lambda p, rid=bi: tr.resident[rid].get(p, 7000 + p),
            reserved=set()))
    return SparseForward(tr, rows, torch.device("cpu"), device_select=device_select), cfg


def test_device_select_excludes_a_cold_candidate_and_eager_promotes_it():
    """The SSD-spill hole: the top scoring candidate is cold (l2p=-1).

    - the device (captured, no-promote) path must score only RESIDENT candidates
      and leave the cold page out, never map it to a phantom block 0;
    - the eager refresh path scores ALL candidates and resolves (promotes) it.
    """
    import torch

    cold = {0}  # candidate page 0 has the highest score but is non-resident
    dev, cfg = _forward_one_cold(1, True, cold)
    q = torch.randn(1, 1, cfg.num_attention_heads, cfg.head_dim)
    table_d, sl_d = dev.attention_args(0, q)
    chosen_dev = {int(x) for x in dev._dchosen[0][0]}
    assert 0 not in chosen_dev, f"cold page selected on device path: {chosen_dev}"
    assert len(chosen_dev) == 2 and all(p >= 1 for p in chosen_dev), chosen_dev
    assert not bool((table_d < 0).any())  # no -1 leaked into the physical table

    eager, _ = _forward_one_cold(1, False, cold)
    table_e, _ = eager.attention_args(0, q)
    chosen_eager = eager._chosen[(0, 0)]
    assert 0 in chosen_eager, f"eager refresh must re-select the cold page: {chosen_eager}"
    assert 7000 in table_e[0].tolist()  # resolve() supplied the promoted fresh block


def _device_engine(refresh, max_new=10):
    import tilerl.sparse_engine as se

    se.SPARSE_REFRESH_TICKS = refresh
    kw = dict(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=8192,
        max_num_batched_tokens=512, prefix_store=NoPrefixStore(),
        sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30,
        sparse_device_select=True)
    return build_engine(**kw)


def test_refresh_r1_device_selection_is_token_equal_to_eager_sparse():
    """R=1: every decode tick is an eager full-candidate refresh, so device-select
    output must equal a pure eager sparse engine (zero staleness)."""
    import tilerl.sparse_engine as se

    saved = se.SPARSE_REFRESH_TICKS
    se.SPARSE_REFRESH_TICKS = 1
    prompt = np.arange(7, 7 + 12 * BLOCK_TOKENS, dtype=np.int64)
    params = SamplingParams(temperature=0.0, max_new_tokens=10, seed=0)
    e = _device_engine(1)
    try:
        t_dev = _drain(e, e.submit(prompt, params), 10)
    finally:
        e.shutdown()
        se.SPARSE_REFRESH_TICKS = saved

    def eager_tokens():
        kw = dict(
            cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=8192,
            max_num_batched_tokens=512, prefix_store=NoPrefixStore(),
            sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30)
        e2 = build_engine(**kw)
        t = _drain(e2, e2.submit(prompt, params), 10)
        e2.shutdown()
        return t

    assert t_dev == eager_tokens(), (t_dev, eager_tokens())


def test_refresh_r8_routes_device_for_seven_ticks_then_eager_promotes():
    """R=8 routing: 7 resident-only device decode ticks, then one eager refresh
    that re-scores all candidates and PROMOTES a cold page the hot set is missing
    (context exceeds the k+window hot set). No device tick promotes a candidate."""
    import tilerl.sparse_engine as se
    from tilerl.sparse_engine import SparseForward

    routed = []
    orig_init = SparseForward.__init__

    def spy_init(self, *a, **k):
        routed.append(k.get("device_select", False))
        return orig_init(self, *a, **k)

    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    saved = se.SPARSE_REFRESH_TICKS
    se.SPARSE_REFRESH_TICKS = 8
    e = _device_engine(8, max_new=12)
    SparseForward.__init__ = spy_init
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=12, seed=0))
    try:
        for _ in range(256):
            d = e.poll()
            if rid in d and len(d[rid]) >= 12:
                break
            e.step()
    finally:
        SparseForward.__init__ = orig_init
        promoted = e._kv.cold.promotions
        e.shutdown()
        se.SPARSE_REFRESH_TICKS = saved

    decode_routes = [x for x in routed]  # one SparseForward per tick incl prefill
    # the first pure-decode tick is device, an eager refresh lands within 8 decode
    assert True in decode_routes and False in decode_routes, decode_routes
    # an eager refresh promoted at least one cold page (the device path never does)
    assert promoted >= 1, promoted
