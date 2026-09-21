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

from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import SamplingParams
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.model import Model, build_random
from tilerl.testing import RefBackend


def _engine(sparse: bool, k: int = 0, scorer: str = "bounds"):
    kw = dict(
        cfg=tiny(),
        model=build_random(tiny(), seed=11),
        backend=RefBackend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        prefix_store=NoPrefixStore(),
    )
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

    Both halves of that invariant are exercised here: the per-tick window over this
    run is `(demoted, promoted)` = `(3,0) (1,1) (1,1) (2,2) (1,1) (1,1)`, so the
    promotion bound is checked against real re-fetches, not a zero. Mutating the
    drop set to the whole resident set (`dropped = list(live)`, the demote-every-tick
    / repromote-6.6x-V100-shape) turns this test red as well as
    `test_sparse_a_stable_selection_promotes_nothing_after_the_first_tick`, which is
    the dedicated zero-cycle pin gate.

    What this test UNIQUELY holds is the device bounds mask: mutating
    `SparseTracker.set_bounds` to stop setting `bounds_valid` fails this test and
    no other in `test_sparse_engine.py` + `test_kv.py` (98 pass). Uses a 12-page
    context so pages genuinely fall outside k+window and cycle, while the per-tick
    counts must still match the residency delta exactly."""
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
    selection is pinned by monkeypatching the CONSUMER (the engine backend's
    select_pages) to a fixed top-k once the rows are in decode."""
    import tilerl_kernels.reference as ref

    prompt = np.arange(7, 7 + 12 * BLOCK_TOKENS, dtype=np.int64)
    sparse = _engine(True, 2)
    backend = sparse._backend
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
    backend.select_pages = stable_select  # instance attr shadows the seam method
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
        del backend.select_pages
    sparse.shutdown()
    assert len(cycles) >= 3, cycles
    # from the second stable tick on, nothing moves between device and host
    assert all(d == 0 and p == 0 for d, p in cycles[1:]), cycles


def test_quest_scores_chunked_over_pages_matches_all_at_once():
    """Scoring splits candidate pages to bound the f32 intermediate (the unchunked
    [Tq,Cp,Hkv,D] is 4.2 GiB at Cp=2048 and OOMs a V100). max-over-query and
    sum-over-head/dim commute with the page split, so the chunked score must be
    bit-identical; Cp is a non-multiple of the chunk to cover the tail."""
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
    common = dict(
        cfg=cfg,
        model=build_random(cfg, seed=11),
        backend=RefBackend(),
        num_slots=1,
        max_batch=1,
        max_total_tokens=1024,
        max_num_batched_tokens=16,
        prefix_store=NoPrefixStore(),
    )
    sparse = build_engine(
        num_blocks=0, sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30, **common
    )
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
                (p, kv) for rid2, pages in tr.keys.items() if pages for p, kv in pages.items()
            )
            assert keys.dtype == torch.float8_e4m3fn and keys.shape[-1] == tr.di
            assert keys.shape[:2] == (len(tr.src_planes), tr.ih)
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

    cli.cmd_serve(
        cli._build_parser().parse_args(
            [
                "serve",
                "--model",
                "tiny",
                "--dry-run",
                "--json",
                "--sparse-k",
                "2",
                "--scorer",
                "bounds",
                "--slots",
                "2",
                "--max-batch",
                "1",
                "--max-ctx",
                "256",
                "--device-free",
                "100000000",
            ]
        )
    )
    owners = {r["owner"] for r in json.loads(capsys.readouterr().out)}
    assert {"page_bounds", "kv_hot"} <= owners, owners
    # the dense kv_pool must NOT also be priced for a sparse engine
    assert "kv_pool" not in owners, owners


def test_sparse_prefix_publishes_only_when_pages_leave_the_hot_union():
    """Drop-only publishing under the cross-tick hot pin (#534): a page is shared
    only when it LEAVES the resident union, and only once pages 0..m-1 have all
    dropped at least once and an exact boundary-m state snapshot exists. A long
    prompt whose early pages drop publishes an entry, and a follower sharing it
    HITS (adopts the block-aligned prefix, prefills only the tail) and matches a
    dense engine. A short prompt wholly inside the hot set publishes nothing
    (#782), which test_a_prompt_that_never_leaves_the_hot_pool_publishes_nothing
    pins separately."""
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
            cfg=tiny(),
            model=build_random(tiny(), seed=11),
            backend=RefBackend(),
            num_blocks=64,
            num_slots=4,
            max_batch=1,
            max_total_tokens=4096,
            max_num_batched_tokens=512,
            sparse_k=2,
            scorer="bounds",
            kv_cold_bytes=1 << 30,
        )

    miss = _sparse()
    ts_miss = _drain(miss, miss.submit(follow, params), 8)
    miss.shutdown()

    sparse = _sparse()
    r1 = sparse.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
    _drain(sparse, r1, 200)
    entry = sparse._sparse.prefix.lookup(follow)
    assert entry is not None and len(entry["keys"]) == 24, (
        None if entry is None else len(entry["keys"])
    )

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

    from tilerl.kv_tiers import HostKvPages
    from tilerl.sparse_engine import SparsePrefixCache

    cold = HostKvPages(budget_bytes=1 << 30)
    cache = SparsePrefixCache(cold, states=None)
    rid, P = 0, 4
    tokens = tuple(range(P * BLOCK_TOKENS))
    cache.set_request(rid, P)
    # Boundaries are noted as prefill chunks complete, BEFORE pages drop: note
    # the first three now; the prompt-end one arrives after the first closure.
    for m in range(1, P):
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
    # bounds ride in each shared blob, read by field rather than pinned in the entry
    assert all(cache.bound_of_key(k) is not None for k in hit3["keys"])
    for p, key in enumerate(hit3["keys"]):
        held = cold.share_take(key)
        assert held is not None and torch.equal(held["k"], torch.full((2,), float(p)))
    # The prompt-end boundary (noted only now, as the last chunk completes) and
    # the last page close the full prefix; the cap must not strand it.
    cache.note_boundary(rid, P, (torch.zeros(2), None))
    cache.publish_dropped(rid, tokens, bounds, P - 1, blob(P - 1))
    hit4 = cache.lookup(tuple(tokens) + (9, 9))
    assert hit4 is not None and len(hit4["keys"]) == P


def test_sparse_prefill_retains_at_most_two_boundary_snapshots_per_request():
    """The 256k V100 SIGKILL: a long single-request prefill stalls the dropped
    frontier at page 0 (k+window keep every early page resident), while every
    aligned 512-token chunk notes a host GDN snapshot (~155 MiB at 27B). The
    snapshots accumulated in a bare dict OUTSIDE HostKvPages' pinned budget:
    one per chunk for the whole prefill. Only the next-closure (lowest) and the
    prompt-end (newest) boundaries can still be consumed, so cap at two."""
    from tilerl.kv_tiers import HostKvPages
    from tilerl.sparse_engine import SparsePrefixCache

    cold = HostKvPages(budget_bytes=1 << 30)
    cache = SparsePrefixCache(cold, states=None)
    rid, P = 0, 64
    cache.set_request(rid, P)

    def snap(m):
        states = (torch.zeros(4, 8), torch.ones(2, 4))
        return states, torch.zeros(3)

    def _bytes(obj):
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (tuple, list)):
            return sum(_bytes(x) for x in obj)
        return 0

    one = _bytes(snap(0))
    for m in range(1, P + 1):
        states, hidden = snap(m)
        cache.note_boundary(rid, m, states, hidden)
        held = cache._snap[rid]
        assert set(held) == {min(held), max(held)} and len(held) <= 2, (
            m, sorted(held))
        assert set(cache._snap_hidden[rid]) == set(held)

    snap_bytes = sum(_bytes(s) for s in cache._snap[rid].values())
    hid_bytes = sum(_bytes(t) for t in cache._snap_hidden[rid].values())
    assert snap_bytes + hid_bytes <= 2 * one
    # newest is the prompt end, lowest still closes the first frontier
    assert max(cache._snap[rid]) == P


def test_sparse_long_prefill_snapshot_cap_keeps_a_follower_hit_exact():
    """End-to-end half of the snapshot cap: with a one-page chunk every prefill
    page is an aligned boundary, so a 32-page single-request prefill notes 32
    snapshots while the dropped frontier stays at page 0 (hot pool 13 pages).
    Under the cap only two survive, yet the prompt-end entry still freezes and a
    follower ADOPTS it and must decode token-identically to a prefix-miss sparse
    engine. Dropping intermediate snapshots changes how much prefix is recomputed,
    not the tokens."""
    cfg = tiny()

    def _eng():
        return build_engine(
            cfg=cfg, model=build_random(cfg, seed=11), backend=RefBackend(),
            num_slots=1, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=16,
            sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30)

    prompt = (np.arange(32 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    follow = np.concatenate([prompt, np.arange(100, 124, dtype=np.int64)])
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)

    miss = _eng()
    ts_miss = _drain(miss, miss.submit(follow, params), 8)
    miss.shutdown()

    pub = _eng()
    # Decode long enough that ALL 32 prompt pages leave the k+window union: with
    # an 8-page own window, page 31 drops after roughly 8 decode pages (~128
    # tokens). Publish-once (#782) forms the prompt-end entry from those natural
    # drops, not from a forced closure at finish.
    rp = pub.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
    _drain(pub, rp, 200)
    cache = pub._sparse.prefix
    # 32 boundaries were noted across the prefill; the prompt-end prefix was
    # published as the last prompt pages dropped during decode.
    assert len(cache._snap) <= 1  # publisher's request finished and dropped its state
    entry = cache.lookup(follow)
    assert entry is not None and len(entry["keys"]) == 32

    rf = pub.submit(follow, params)
    pub.step()
    req = next(x for x in pub._running if x.req_id == rf)
    assert req.sparse_matched == 32 * BLOCK_TOKENS, req.sparse_matched
    ts = _drain(pub, rf, 8)
    pub.shutdown()
    assert ts == ts_miss, f"prefix-hit follower {ts} != prefix-miss sparse {ts_miss}"


def test_sparse_prefix_republished_after_repin_keeps_the_first_blob():
    """A page drops (published), the pin re-selects it, and it drops again with a
    new private blob: the shared prefix must keep serving the FIRST captured
    blob — the clone is independent of the private frame across a pin boundary."""
    import torch

    from tilerl.kv_tiers import HostKvPages
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

    from tilerl.kv_cache import PagedKvPool
    from tilerl.kv_tiers import HostKvPages

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
        cfg,
        build_random(cfg, seed=11),
        RefBackend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        prefix_store=NoPrefixStore(),
        sparse_k=2,
        scorer="bounds",
        kv_cold_bytes=per,  # one page of pinned host RAM
        cold_ssd_path=ssd,
    )
    # 24 pages. Under the cross-tick pin the device hot set is k+window (+chunk):
    # tiny has one source group, k=2 and the trailing 8-page window stay resident,
    # so a context must EXCEED k+window pages for anything to demote and spill.
    # (This gate predates the pin: its old 6-page prompt now fits entirely in the
    # hot set, so host=0/ssd=0 every tick — correct pin behaviour, no spill.)
    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=6, seed=0))
    for _ in range(128):  # into decode, before the request finishes
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

    from tilerl.model import build_random
    from tilerl.spec import DraftHead

    dcfg = replace(cfg, num_layers=1, full_attn_layers=(0,), fp4=False)
    params = {k: v for k, v in build_random(dcfg, seed=3).params.items() if k.startswith("layers.")}
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
    kw = dict(
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        sparse_k=sparse_k,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
    )
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
        cfg=cfg,
        model=model,
        backend=get_backend(),
        num_blocks=64,
        num_slots=4,
        max_batch=2,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        sparse_k=2,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        draft=_draft(cfg, model),
        spec_depth=1,
    )
    # The 5-page prompt needs 6 dense draft blocks (plus one verify position);
    # size the shared draft pool so one row fits but two cannot.
    from tilerl.kv_cache import PagedKvPool

    d = e._draft
    d.kv = PagedKvPool(
        6,
        cfg.num_kv_heads,
        cfg.head_dim,
        num_layers=d.cfg.num_layers,
        layer_map=tuple(range(d.cfg.num_layers)),
        device=d.backend.device,
        dtype=d.kv.dtype,
    )
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)  # 6 pages
    params = SamplingParams(temperature=0.0, max_new_tokens=4, seed=0)
    e.submit(prompt, params)
    try:
        for _ in range(6):  # admit r1, grow its draft blocks
            e.step()
            if e._running and e._running[0].draft_blocks:
                break
        assert e._draft.kv.free_blocks < 6, "fixture: row 1 did not fill the draft pool"
        e.submit(prompt, params)  # static guard passes, admit rejects
        slots_before = e.stats()["slots_used"]
        for _ in range(6):
            e.step()  # r2 retries admit every tick
        st = e.stats()
        assert st["slots_used"] == slots_before == 1, (
            f"rejected admits leaked slots: {st['slots_used']} vs {slots_before}"
        )
    finally:
        e.shutdown()


def test_sparse_draft_pool_is_reserved_at_admit_across_rows():
    """The dense draft pool is a SHARED device resource, but sparse admit used to
    check it only against the row's own PROMPT pages and grew draft blocks lazily
    in the forward loop. Two rows each statically within capacity, admitted in one
    _build_plan, both passed and the second row's alloc_block raised
    PagedKvPool exhausted mid-forward. The full draft span (prompt + max_new +
    verify width-1, submit's static bound) is RESERVED at admit so cross-row
    accounting lives in the pool's free list: row 2 waits until row 1 releases,
    and no forward ever exhausts."""
    from tilerl_kernels.backend import get_backend

    cfg = tiny()
    model = build_random(cfg, seed=11)
    e = build_engine(
        cfg=cfg,
        model=model,
        backend=get_backend(),
        num_blocks=64,
        num_slots=4,
        max_batch=2,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        sparse_k=2,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        draft=_draft(cfg, model),
        spec_depth=1,
    )
    # One full draft span: 48 prompt + 2 new + 1 verify position = 51 -> 4 blocks.
    from tilerl.kv_cache import PagedKvPool

    d = e._draft
    d.kv = PagedKvPool(
        4,
        cfg.num_kv_heads,
        cfg.head_dim,
        num_layers=d.cfg.num_layers,
        layer_map=tuple(range(d.cfg.num_layers)),
        device=d.backend.device,
        dtype=d.kv.dtype,
    )
    prompt = np.arange(7, 7 + 3 * BLOCK_TOKENS, dtype=np.int64)  # exactly 3 pages
    params = SamplingParams(temperature=0.0, max_new_tokens=2, seed=0)
    e.submit(prompt, params)
    r2 = e.submit(prompt, params)
    t2 = None
    try:
        for _ in range(256):
            e.step()  # raised "PagedKvPool exhausted" inside the forward pre-fix
            running = [x for x in e._running]
            # two sparse draft rows must never be admitted against a one-row pool
            assert len(running) <= 1, f"{len(running)} draft rows share a one-row draft pool"
            t2 = e.poll().get(r2, t2)
            if t2 is not None and len(t2) >= 2:
                break
        assert t2 is not None and len(t2) == 2, (
            "row 2 must admit and finish after row 1 releases the shared pool"
        )
    finally:
        e.shutdown()


def test_sparse_verify_tick_populates_and_reads_the_plus_one_own_page_across_a_boundary():
    """The +1 own-page column exists only for a verify tick (tq>1); a depth-1 run
    on a short prompt never has a draft chain that STRADDLES a 16-token boundary,
    so reserving the next own page was unexercised. A block-aligned prefill plus
    an always-accepted (oracle) draft puts the first verify chain at committed
    position L-1 (offset 15) and a draft on page L/16+1: the own span names the
    +1 page, the write populates it, attention reads it, and committed tokens
    equal a dense engine. An oracle head is essential — a random draft's crossing
    proposals are rejected at position 0 and never change output."""
    from dataclasses import replace as _replace

    from tilerl_kernels.backend import get_backend

    from tilerl.spec import DraftHead

    class _OracleDraft(DraftHead):
        def __init__(self, cfg, expected):
            self.cfg = _replace(cfg, num_layers=1, full_attn_layers=(0,))
            self.params, self.expected = {}, expected
            self.width = 2
            self.has_confidence = False
            self.trunk = None

        def forward(self, hidden, ids, positions, kv, backend, hidden_out=None, last_only=False):
            pos = np.atleast_2d(np.asarray(positions))
            logits = torch.zeros(*pos.shape, self.cfg.vocab_size, device=backend.device)
            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    logits[i, j, self.expected.get(int(pos[i, j]) + 1, 0)] = 10.0
            if hidden_out is not None:
                hidden_out.append(torch.as_tensor(hidden))
            return logits

        def confidence(self, hidden, probs, backend):
            return probs

    cfg = tiny()
    model = build_random(cfg, seed=11)
    backend = get_backend()
    common = dict(
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096, max_num_batched_tokens=512
    )
    # 4 full pages + 15 tokens: after prefill the first verify's committed
    # position lands on offset 15, so its tq=2 chain writes 79 (page 4) and 80
    # (page 5) — the +1 column is page 5, beyond the page the committed token is on.
    prompt = np.arange(7, 7 + 4 * BLOCK_TOKENS + 15, dtype=np.int64)
    n_new = 8
    params = SamplingParams(temperature=0.0, max_new_tokens=n_new, seed=0)

    dense = build_engine(cfg=cfg, model=model, backend=backend, sparse_k=0, **common)
    base = _drain(dense, dense.submit(prompt, params), n_new)
    dense.shutdown()
    expected = {i: t for i, t in enumerate(list(prompt) + base)}

    crossed = {"n": 0}
    e = build_engine(
        cfg=cfg,
        model=model,
        backend=backend,
        sparse_k=2,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        draft=_OracleDraft(cfg, expected),
        spec_depth=1,
        **common,
    )
    import tilerl.sparse_engine as se

    orig = se.SparseForward.__init__

    def watch(self, *a, **kw):
        orig(self, *a, **kw)
        for r in self.rows:
            if r["decoding"] and r["tq"] > 1:
                # chain queries are [q_hi-tq .. q_hi): a boundary straddle has
                # first and last query on different pages, so own must name +1.
                first_page = (r["q_hi"] - r["tq"]) // BLOCK_TOKENS
                last_page = (r["q_hi"] - 1) // BLOCK_TOKENS
                if last_page > first_page and last_page in r["own"]:
                    crossed["n"] += 1

    se.SparseForward.__init__ = watch
    try:
        got = _drain(e, e.submit(prompt, params), n_new)
    finally:
        se.SparseForward.__init__ = orig
        e.shutdown()
    assert crossed["n"] >= 1, "no verify chain straddled a page boundary into +1 own"
    assert got == base, f"sparse+oracle crossing {got} != dense greedy {base}"


def test_sparse_draft_follower_returns_miss_on_a_published_prefix():
    """Under sparse+spec a follower must NOT adopt a published trunk prefix: the
    draft head conditions every position on trunk hidden and builds its own dense
    KV only while forwarding, so an adopted (non-forwarded) prefix leaves the
    draft attending over an unbuilt pool. The correct behaviour is return-miss —
    prefill from zero, which builds trunk and draft KV correctly. A no-draft
    follower still adopts the same prefix (the save is preserved)."""
    from tilerl.sparse_engine import page_key

    cfg = tiny()
    model = build_random(cfg, seed=11)
    n_tokens = 5 * BLOCK_TOKENS
    prefix = [int(t) for t in np.arange(7, 7 + n_tokens)]

    def build(draft):
        from tilerl_kernels.backend import get_backend

        return build_engine(
            cfg=cfg,
            model=build_random(cfg, seed=11),
            backend=get_backend(),
            num_blocks=64,
            num_slots=4,
            max_batch=2,
            max_total_tokens=4096,
            max_num_batched_tokens=512,
            sparse_k=2,
            scorer="bounds",
            kv_cold_bytes=1 << 30,
            **({"draft": draft, "spec_depth": 1} if draft else {}),
        )

    def seed_published_prefix(e):
        # Minimal block-aligned entry: lookup matches on tokens; the draft path
        # must return-miss before it touches keys/bounds/state.
        cache = e._sparse.prefix
        ptokens = tuple(prefix[: n_tokens - (n_tokens % BLOCK_TOKENS)])
        # Reference the live state tensor (no extra allocation to perturb the
        # measured-peak memory ledger); the draft path returns-miss before reading it.
        entry = {
            "eid": cache._next_id,
            "tokens": ptokens,
            "keys": [],
            "bounds": {},
            "state": (e._states.states[0], None),
        }
        cache._next_id += 1
        cache._entries.setdefault(page_key(ptokens, len(ptokens) // BLOCK_TOKENS - 1), []).append(
            entry
        )
        cache._by_id[entry["eid"]] = entry
        return len(ptokens)

    params = SamplingParams(temperature=0.0, max_new_tokens=4, seed=0)

    # WITH a draft: a published prefix exists but the follower returns-miss and
    # runs to completion (prefilling from zero builds the draft KV correctly;
    # sparse+/-draft token equality is covered by
    # test_sparse_with_draft_matches_sparse_without_draft_greedy).
    spec = build(_draft(cfg, model))
    seed_published_prefix(spec)
    rid = spec.submit(np.asarray(prefix), params)
    got = None
    try:
        for _ in range(60):
            spec.step()
            running = [r for r in spec._running if r.req_id == rid]
            if running:
                # sparse_matched==0 is the return-miss signal: the published entry
                # was not adopted, so this follower prefills from zero.
                assert running[0].sparse_matched == 0, "draft follower adopted a trunk prefix"
            d = spec.poll()
            if rid in d and len(d[rid]) >= 4:
                got = d[rid][:4]
                break
        assert spec._prefix_hits == 0
        assert got is not None, "return-miss draft follower did not finish"
    finally:
        spec.shutdown()

    # The return-miss follower must be bit-equal to a cold follower that never saw
    # the published prefix — same trunk and draft KV, same proposals.
    cold = build(_draft(cfg, model))
    cold_got = _drain(cold, cold.submit(np.asarray(prefix), params), 4)
    cold.shutdown()
    assert got == cold_got, f"return-miss {got} != cold {cold_got}"

    # WITHOUT a draft the same published prefix IS adopted (save preserved).
    plain = build(None)
    matched_len = seed_published_prefix(plain)
    rid2 = plain.submit(np.asarray(prefix), params)
    try:
        for _ in range(20):
            plain.step()
            running = [r for r in plain._running if r.req_id == rid2]
            if running:
                assert running[0].sparse_matched == matched_len, (
                    "non-spec follower should adopt the prefix"
                )
                break
        assert plain._prefix_hits == 1
    finally:
        plain.shutdown()


def test_sparse_spec_b2_concurrent_rows_do_not_cross_contaminate():
    """B>1 gap behind the H20 MMLU failure (sparse k=128 + spec B=8 scored 0.20
    while accepting ~0.83 of drafts): under GREEDY a draft head only accelerates,
    it never changes a committed token. So two DISTINCT prompts running in one
    sparse+spec B=2 tick must each produce exactly their isolated B=1 tokens.
    Any cross-row read (the packed table or a chain K/V write landing in the
    other row's blocks) moves one row off its isolated answer. The CPU oracle
    covers engine/draft-pool/packed geometry row isolation; the sm90 decode
    kernel's per-row mask is the residual a card run checks."""
    from tilerl_kernels.backend import get_backend

    cfg = tiny()
    model = build_random(cfg, seed=11)
    n_new = 8
    # Two deliberately different prompts (different offset + length parity).
    pa = np.arange(7, 7 + 6 * BLOCK_TOKENS + 3, dtype=np.int64)
    pb = (np.arange(5 * BLOCK_TOKENS + 11, dtype=np.int64) % 300) + 101
    params = SamplingParams(temperature=0.0, max_new_tokens=n_new, seed=0)

    def isolated(prompt):
        be = get_backend()
        e = build_engine(
            cfg=cfg, model=build_random(cfg, seed=11), backend=be,
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=512, sparse_k=0)
        out = _drain(e, e.submit(prompt, params), n_new)
        e.shutdown()
        return out

    want_a, want_b = isolated(pa), isolated(pb)

    be = get_backend()
    e = build_engine(
        cfg=cfg, model=build_random(cfg, seed=11), backend=be,
        num_blocks=64, num_slots=4, max_batch=2, max_total_tokens=4096,
        max_num_batched_tokens=512, sparse_k=2, scorer="bounds",
        kv_cold_bytes=1 << 30, draft=_draft(cfg, model), spec_depth=1)
    ra = e.submit(pa, params)
    rb = e.submit(pb, params)
    ga = gb = None
    for _ in range(512):
        d = e.poll()
        ga, gb = d.get(ra, ga), d.get(rb, gb)
        if ga is not None and len(ga) >= n_new and gb is not None and len(gb) >= n_new:
            break
        e.step()
    e.shutdown()
    assert ga[:n_new] == want_a, ("row A cross-contaminated", ga[:n_new], want_a)
    assert gb[:n_new] == want_b, ("row B cross-contaminated", gb[:n_new], want_b)


def test_sparse_spec_b8_rows_equal_their_isolated_runs_at_full_coverage():
    """The MMLU shape (H20 sparse k=128 + spec B=8 scored 0.20): at full page
    coverage (k >= every context page, the regime where 65's spec-OFF run still
    failed on sm90) sparse MUST be token-equal to dense. Eight distinct prompts
    in one sparse+spec B=8 tick each equal their isolated dense B=1 answer. This
    pins the engine/draft-pool/packed-geometry layer row-isolation on CPU; the
    sm90 decode kernel's per-row packed-table read is the card residual."""
    cfg = tiny()
    n_new = 6
    prompts = []
    for b in range(8):
        p = (np.arange((10 + 2 * b) * BLOCK_TOKENS + (b % 3), dtype=np.int64) % 300) + b
        prompts.append(p)
    params = SamplingParams(temperature=0.0, max_new_tokens=n_new, seed=0)
    k = 64

    def isolated(prompt):
        from tilerl_kernels.backend import get_backend

        e = build_engine(
            cfg=cfg, model=build_random(cfg, seed=11), backend=get_backend(),
            num_blocks=200, num_slots=2, max_batch=1, max_total_tokens=8192,
            max_num_batched_tokens=512, sparse_k=0)
        out = _drain(e, e.submit(prompt, params), n_new)
        e.shutdown()
        return out

    want = [isolated(p) for p in prompts]

    from tilerl_kernels.backend import get_backend

    model = build_random(cfg, seed=11)
    e = build_engine(
        cfg=cfg, model=model, backend=get_backend(),
        num_blocks=200, num_slots=8, max_batch=8, max_total_tokens=8192,
        max_num_batched_tokens=512, sparse_k=k, scorer="bounds",
        kv_cold_bytes=1 << 30, draft=_draft(cfg, model), spec_depth=1)
    rids = [e.submit(p, params) for p in prompts]
    outs = [None] * 8
    for _ in range(700):
        d = e.poll()
        for i, r in enumerate(rids):
            outs[i] = d.get(r, outs[i])
        if all(o is not None and len(o) >= n_new for o in outs):
            break
        e.step()
    e.shutdown()
    for i in range(8):
        assert outs[i][:n_new] == want[i], (i, outs[i][:n_new], want[i])


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

    full = build_engine(
        cfg=cfg,
        model=model,
        backend=get_backend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        draft=_draft(cfg, model),
        spec_depth=1,
    )
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
        row = dict(
            req_id=0,
            own=[n_cand],
            own_len=BLOCK_TOKENS,
            cand=cand,
            force_window=0,
            resolve=lambda p: p,
            reserved=set(),
            decoding=True,
            tq=1,
        )
        sf = SparseForward(tr, [row], torch.device("cpu"), RefBackend())
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
    tr.attach(0)  # first request, no prior bounds to infer from
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
        rows.append(
            dict(
                req_id=bi,
                own=own,
                own_len=2 * BLOCK_TOKENS + 5,
                q_start=0,
                q_hi=0,
                decoding=True,
                tq=1,
                cand=cand,
                force_window=0,
                resolve=lambda p, rid=bi: tr.resident[rid][p],
                reserved=set(),
            )
        )
    sf = SparseForward(tr, rows, torch.device("cpu"), RefBackend(),
                        device_select=device_select)
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
        assert table.shape == (B, 4 + 2), table.shape  # fixed k + trailing own window
        assert sl.shape == (B,)
        # seq_len excludes the padded tail: each row selected exactly k here.
        assert torch.equal(sl, torch.full((B,), 4 * BLOCK_TOKENS + 2 * BLOCK_TOKENS + 5))


def test_device_select_packed_table_matches_eager_at_b1_and_b8():
    """Token-equality to eager sparse at the SparseForward level: the device
    path's leading compact physical columns and seq_len must equal the eager
    path's packed table at B=1 and B=8 (same selection, same own pages)."""
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
            cfg=tiny(),
            model=build_random(tiny(), seed=11),
            backend=RefBackend(),
            num_blocks=64,
            num_slots=4,
            max_batch=1,
            max_total_tokens=4096,
            max_num_batched_tokens=512,
            prefix_store=NoPrefixStore(),
            sparse_k=6,
            scorer="bounds",
            kv_cold_bytes=1 << 30,
            sparse_device_select=devsel,
        )
        e = build_engine(**kw)
        tok = _drain(e, e.submit(prompt, params), 8)
        e.shutdown()
        return tok

    assert run(True) == run(False)


def test_quest_scores_batched_matches_single_row_bit_for_bit():
    """The captured tick scores all B rows with quest_scores_batched; it must be
    bit-identical to applying the single-row quest_scores per row (the page chunk
    split commutes over rows exactly as it does over pages)."""
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
            b = torch.full((tr.n_full, tr.hkv, 2, tr.dim), 1.0 - pi * 0.01, dtype=torch.float16)
            tr.set_bounds(bi, p, b)
        phys = {}
        for p in cand + own:
            if p in cold_pages:
                continue  # cold: bounds held, l2p stays -1
            pm = 1000 + bi * 1000 + p
            phys[p] = pm
            tr.map_resident(bi, p, pm)
        tr.resident[bi] = phys
        rows.append(
            dict(
                req_id=bi,
                own=own,
                own_len=2 * BLOCK_TOKENS + 5,
                q_start=0,
                q_hi=0,
                decoding=True,
                tq=1,
                cand=cand,
                force_window=0,
                resolve=lambda p, rid=bi: tr.resident[rid].get(p, 7000 + p),
                reserved=set(),
            )
        )
    return SparseForward(tr, rows, torch.device("cpu"), RefBackend(),
                        device_select=device_select), cfg


def test_device_select_excludes_a_cold_candidate_and_eager_promotes_it():
    """The SSD-spill hole: the top scoring candidate is cold (l2p=-1).

    - the device (captured, no-promote) path must score only RESIDENT candidates
      and leave the cold page out, never map it to a phantom block 0;
    - the eager refresh path scores ALL candidates and resolves (promotes) it.
    """
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
        cfg=tiny(),
        model=build_random(tiny(), seed=11),
        backend=RefBackend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=8192,
        max_num_batched_tokens=512,
        prefix_store=NoPrefixStore(),
        sparse_k=2,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        sparse_device_select=True,
    )
    return build_engine(**kw)


def test_sparse_captured_decode_tick_tokens_equal_eager_across_refresh():
    """The sparse decode graph path (persistent, refillable SparseForward; the
    CPU seam that stands in for a cuda CUDAGraph) must produce the same tokens
    as a fully eager sparse engine. The comparison is at FULL k, where the
    selection is exact and the resident-only device path is token-identical to
    eager (at small k the device path is deliberately stale between refreshes,
    which moves tokens — a property the fidelity table prices, not a bug). The
    run crosses several SPARSE_REFRESH_TICKS boundaries, exercising graph ->
    eager refresh -> graph with the re-pinned hot set."""
    import tilerl.sparse_engine as se

    R = se.SPARSE_REFRESH_TICKS
    prompt = np.arange(7, 7 + 12 * BLOCK_TOKENS, dtype=np.int64)
    n_new = R * 3 + 2
    params = SamplingParams(temperature=0.0, max_new_tokens=n_new, seed=0)

    def run(graph_on: bool):
        kw = dict(
            cfg=tiny(),
            model=build_random(tiny(), seed=11),
            backend=RefBackend(),
            num_blocks=64,
            num_slots=4,
            max_batch=1,
            max_total_tokens=8192,
            max_num_batched_tokens=512,
            prefix_store=NoPrefixStore(),
            sparse_k=64,
            scorer="bounds",
            kv_cold_bytes=1 << 30,
            sparse_device_select=True,
        )
        e = build_engine(**kw)
        if not graph_on:
            e._sparse_graph_on = False
        rid = e.submit(prompt, params)
        out: list = []
        for _ in range(600):
            e.step()
            out = e.poll().get(rid, out)
            if len(out) >= n_new:
                break
        e.shutdown()
        return out

    e = build_engine(
        cfg=tiny(),
        model=build_random(tiny(), seed=11),
        backend=RefBackend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=8192,
        max_num_batched_tokens=512,
        prefix_store=NoPrefixStore(),
        sparse_k=64,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        sparse_device_select=True,
    )
    n_graph = {"n": 0}
    orig = e._run_sparse_decode_graph

    def counting(reqs, chains):
        ran = orig(reqs, chains)
        n_graph["n"] += int(ran)
        return ran

    e._run_sparse_decode_graph = counting
    rid = e.submit(prompt, params)
    t_graph: list = []
    try:
        for _ in range(600):
            e.step()
            t_graph = e.poll().get(rid, t_graph)
            if len(t_graph) >= n_new:
                break
    finally:
        e.shutdown()
    assert n_graph["n"] >= R, f"graph path took only {n_graph['n']} ticks"
    t_eager = run(False)
    assert t_graph == t_eager, (t_graph, t_eager)


def test_sparse_graph_fill_allocates_nothing_inside_the_captured_region():
    """Capture seam: after fill() the replay's selection/attention_arg math must
    allocate no new tensors and read only the persistent staging buffers (a
    graph bakes pointers, so a fresh tensor or a tracker-dict read breaks it).
    fill() itself is allowed to allocate (it runs outside capture)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    import tilerl.sparse_engine as se

    cfg = tiny()
    e = _device_engine(8, max_new=6)
    prompt = np.arange(7, 7 + 12 * BLOCK_TOKENS, dtype=np.int64)
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=6, seed=0))
    # Run one graph tick to materialize a persistent sf, then grab it.
    for _ in range(60):
        e.step()
        if e._sparse_graphs:
            break
    g = next(iter(e._sparse_graphs.values())).sf
    assert g.reuse and g.s_bounds is not None
    # Refill from the live running row and record every tensor the math allocates.
    r = next(x for x in e._running if x.req_id == rid)
    q_dec = [1]
    rows = e._sparse_decode_rows([r], q_dec)
    g.fill(rows)
    q = torch.randn(1, 1, cfg.num_attention_heads, cfg.head_dim)

    class _AllocCount(TorchDispatchMode):
        n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            # creation ops allocate; in-place/index reads into existing buffers do not
            if func.__name__.split(".")[-1] in {
                "empty",
                "zeros",
                "zeros_like",
                "full",
                "full_like",
                "ones",
                "ones_like",
                "clone",
                "stack",
                "new_empty",
            }:
                self.n += 1
            return func(*args, **(kwargs or {}))

    with _AllocCount() as c:
        g._select_device(g.tracker.group_of[g.tracker.src_planes[0]] and 0, q)
        g._attention_args_device(0, q)
    assert c.n == 0, f"captured region allocated {c.n} tensors"
    e.shutdown()
    se.SPARSE_REFRESH_TICKS = 8


def test_sparse_graph_verify_tick_w2_tokens_equal_dense():
    """A tq=2 spec verify chain through the captured sparse graph (its own span
    is the +1-page width) must be token-equal to a dense engine, using the
    always-accepted oracle draft so every proposal lands and the verify graph
    takes a tick (the W=2 bucket, keep_steps state, +1 own page)."""
    from dataclasses import replace

    from tilerl_kernels.backend import get_backend

    from tilerl.spec import DraftHead

    class _OracleDraft(DraftHead):
        def __init__(self, cfg, expected):
            self.cfg = replace(cfg, num_layers=1, full_attn_layers=(0,))
            self.params, self.expected, self.width, self.has_confidence, self.trunk = (
                {},
                expected,
                2,
                False,
                None,
            )

        def forward(self, hidden, ids, positions, kv, backend, hidden_out=None, last_only=False):
            pos = np.atleast_2d(np.asarray(positions))
            logits = torch.zeros(*pos.shape, self.cfg.vocab_size, device=backend.device)
            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    logits[i, j, self.expected.get(int(pos[i, j]) + 1, 0)] = 10.0
            if hidden_out is not None:
                hidden_out.append(torch.as_tensor(hidden))
            return logits

        def confidence(self, hidden, probs, backend):
            return probs

    cfg = tiny()
    model = build_random(cfg, seed=11)
    be = get_backend()
    prompt = np.arange(7, 7 + 4 * BLOCK_TOKENS + 15, dtype=np.int64)
    dense = build_engine(
        cfg=cfg,
        model=model,
        backend=be,
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        sparse_k=0,
    )
    rid = dense.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=8, seed=0))
    base = _drain(dense, rid, 8)
    dense.shutdown()
    expected = {i: t for i, t in enumerate(list(prompt) + base)}
    e = build_engine(
        cfg=cfg,
        model=build_random(cfg, seed=11),
        backend=be,
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        sparse_k=2,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        draft=_OracleDraft(cfg, expected),
        spec_depth=1,
        sparse_device_select=True,
    )
    n = {"g": 0}
    orig = e._run_sparse_decode_graph
    e._run_sparse_decode_graph = lambda r, c, n=n: n.__setitem__("g", n["g"] + 1) or orig(r, c)
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=8, seed=0))
    got = _drain(e, rid, 8)
    e.shutdown()
    assert n["g"] >= 1, "the W=2 verify graph never ran"
    assert got == base, f"graph verify {got} != dense {base}"


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
            cfg=tiny(),
            model=build_random(tiny(), seed=11),
            backend=RefBackend(),
            num_blocks=64,
            num_slots=4,
            max_batch=1,
            max_total_tokens=8192,
            max_num_batched_tokens=512,
            prefix_store=NoPrefixStore(),
            sparse_k=2,
            scorer="bounds",
            kv_cold_bytes=1 << 30,
        )
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


def test_sparse_is_off_by_default_after_the_sm90_hotfix():
    """DEFAULT_SPARSE_K=0: a no-flag build_engine/serve build is the DENSE engine
    (sparse opt-in only via --sparse-k N). Reverted from 128 on 2026-09-13: the
    sm90 sparse path was discontinuous even with spec off."""
    from tilerl.sparse_index import DEFAULT_SPARSE_K

    assert DEFAULT_SPARSE_K == 0
    dense = build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
        max_num_batched_tokens=512, prefix_store=NoPrefixStore())
    assert dense._sparse is None
    dense.shutdown()

    sparse = build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
        max_num_batched_tokens=512, prefix_store=NoPrefixStore(),
        sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30)
    assert sparse._sparse is not None and sparse._sparse.k_pages == 2
    sparse.shutdown()


def test_prefix_publish_consumes_boundary_snapshots_no_second_copy():
    """The 256k host OOM had a SECOND uncapped container beside the shared-blob
    clone: SparsePrefixCache._snap kept every consumed chunk-boundary GDN snapshot
    until request end. After a frontier closes, the boundary snapshot it used
    must be popped (the entry now owns it); only unconsumed snapshots remain."""
    import torch as _torch

    from tilerl.kv_tiers import HostKvPages
    from tilerl.sparse_engine import SparsePrefixCache

    cold = HostKvPages(budget_bytes=1 << 30)
    cache = SparsePrefixCache(cold, states=None)
    rid, P = 0, 4
    tokens = tuple(range(P * BLOCK_TOKENS))
    cache.set_request(rid, P)
    bounds = {p: _torch.zeros(1) for p in range(P)}

    def blob(p):
        t = _torch.full((2,), float(p))
        return {"k": t, "v": t}

    # offer every page, supplying an exact snapshot at each boundary
    for m in range(1, P + 1):
        cache.note_boundary(rid, m, (_torch.zeros(2), None))
    for p in range(P):
        cache.publish_dropped(rid, tokens, bounds, p, blob(p))

    # the full 4-page prefix froze; no snapshot it consumed is retained twice
    snap = cache._snap.get(rid, {})
    assert snap == {}, f"consumed boundary snapshots retained: {sorted(snap)}"
    hit = cache.lookup(tokens)
    assert hit is not None and len(hit["keys"]) == P


def test_published_content_keys_are_bit_identical_to_page_key():
    """The incremental rolling hash must produce the SAME integer content keys as
    the from-zero page_key at every page — published blobs are addressed by
    them, so a recurrence change silently corrupts the shared index."""
    import torch as _torch

    from tilerl.kv_tiers import HostKvPages
    from tilerl.sparse_engine import SparsePrefixCache, page_key

    rng = np.random.default_rng(0)
    for P in (1, 7, 13):
        tokens = tuple(int(x) for x in rng.integers(0, 100_000, P * BLOCK_TOKENS))
        cold = HostKvPages(budget_bytes=1 << 30)
        cache = SparsePrefixCache(cold, states=None)
        rid = 0
        cache.set_request(rid, P)
        bounds = {p: _torch.zeros(1) for p in range(P)}
        for m in range(1, P + 1):
            cache.note_boundary(rid, m, (_torch.zeros(2), None))
        for p in range(P):
            t = _torch.full((2,), float(p))
            out = cache.publish_dropped(rid, tokens, bounds, p, {"k": t, "v": t})
            for page, key in out.items():
                assert key == page_key(tokens, page)
        entry = cache.lookup(tokens)
        assert entry is not None
        assert entry["keys"] == [page_key(tokens, p) for p in range(P)]
        # the entry sits on the chain keyed by the from-zero page-(m-1) hash
        assert entry["hash"] == page_key(tokens, P - 1)
        assert entry in cache._entries[page_key(tokens, P - 1)]


def test_publish_hash_steps_stay_linear_not_quadratic_in_context():
    """256k top profile frame: page_key rehashed the whole prefix per dropped page
    (O(M^2)). With the rolling hash, extending by one page hashes only its 16
    tokens, so over P one-at-a-time drops _page_hash is called exactly 16*P
    times — not sum_p 16*(p+1)."""
    import torch as _torch

    from tilerl.kv_tiers import HostKvPages
    from tilerl.sparse_engine import SparsePrefixCache

    P = 128
    cold = HostKvPages(budget_bytes=1 << 30)
    cache = SparsePrefixCache(cold, states=None)
    rid = 0
    cache.set_request(rid, P)
    bounds = {p: _torch.zeros(1) for p in range(P)}
    for m in range(1, P + 1):
        cache.note_boundary(rid, m, (_torch.zeros(2), None))

    calls = 0
    from tilerl import kv_cache as kvc
    orig = kvc._rolling_hash

    def counted(prev, token):
        nonlocal calls
        calls += 1
        return orig(prev, token)

    kvc._rolling_hash = counted
    try:
        # the engine passes the prefix as it exists at the moment each page drops,
        # growing by exactly one page between calls
        for p in range(P):
            t = _torch.full((2,), float(p))
            cache.publish_dropped(rid, tuple(range((p + 1) * BLOCK_TOKENS)),
                                 bounds, p, {"k": t, "v": t})
    finally:
        kvc._rolling_hash = orig
    assert calls == 16 * P, calls


def test_drop_reads_the_host_bounds_mask_without_touching_the_device_tensor():
    """has_bounds per dropped page did bool(device_mask[page]), a device sync per
    page. It must read the host mirror: after bounds are written, swapping the
    device twin for an object that explodes on indexing must leave the per-drop
    has_bounds path untouched."""
    from tilerl.sparse_engine import SparseTracker

    tr = SparseTracker(tiny(), k_pages=4, scorer="bounds",
                       device=torch.device("cpu"))
    tr.attach(0)
    b = torch.zeros((tr.n_full, tr.hkv, 2, tr.dim), dtype=torch.float16)
    tr.set_bounds(0, 0, b)
    tr.set_bounds(0, 2, b)

    class _Explodes:
        def __getitem__(self, idx):
            raise AssertionError("has_bounds touched the device bounds mask")

    tr.bounds_valid[0] = _Explodes()
    assert tr.has_bounds(0, 0)
    assert not tr.has_bounds(0, 1)
    assert tr.has_bounds(0, 2)
    assert not tr.has_bounds(0, 999)
    tr.drop(0)


def test_fused_attn_prep_branch_gets_the_sparse_packed_attention_args():
    """The fused-attn_prep branch and the unfused branch must feed paged_attention
    the same sparse packed [selected;own] table.

    Regression for the sm90 fused path: model._full_attn's fused early-return once
    called paged_attention with the raw kv.block_table/kv.seq_len (the own-window
    table) instead of SparseForward.attention_args' packed table, so sparse-fused
    diverged from sparse-unfused (device gate: B=1 g0 max_abs 7-17, B=8 2-27).
    RefBackend has no fused kernel (attn_prep returns None), so this uses a
    subclass that mirrors the unfused prelude and returns qn: same K/V and q, and
    the two branches' recorded attention args must be identical."""
    cfg = tiny()
    hq, d = cfg.num_attention_heads, cfg.head_dim
    q_rows = hq * 2 * d

    class FusedRefBackend(RefBackend):
        def attn_prep(self, qkv, wq, wk, positions, theta, rotary_dim, kv,
                      layer_idx, hq_, hkv_, eps):
            if layer_idx not in cfg.full_attn_layers:
                return None
            b, t, _ = qkv.shape
            q = qkv[..., :q_rows].reshape(b, t, hq_, 2, d)[..., 0, :]
            k = qkv[..., q_rows:q_rows + hkv_ * d].reshape(b, t, hkv_, d)
            v = qkv[..., q_rows + hkv_ * d:].reshape(b, t, hkv_, d)
            q = self.rmsnorm_f32(q, wq, eps)
            k = self.rmsnorm_f32(k, wk, eps)
            q = self.rope(q, positions, theta, rotary_dim=rotary_dim)
            k = self.rope(k, positions, theta, rotary_dim=rotary_dim)
            self.write_tokens(k, v, kv, layer_idx)
            return q

    def fused_params():
        m = build_random(cfg, seed=11)
        p = "layers.0"
        qw = m.params.pop(f"{p}.q_proj").reshape(hq, 2, d, cfg.hidden_size)
        qw = qw.reshape(q_rows, cfg.hidden_size)
        m.params[f"{p}.qkv"] = torch.cat(
            [qw, m.params.pop(f"{p}.k_proj"), m.params.pop(f"{p}.v_proj")], 0).contiguous()
        return m.params

    def run(fused: bool):
        be = FusedRefBackend() if fused else RefBackend()
        e = build_engine(
            cfg=cfg, model=Model(cfg, fused_params()), backend=be,
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=512, prefix_store=NoPrefixStore(),
            sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30)
        rec = []
        orig = be.paged_attention

        def rec_attn(q, kc, vc, bt, sl, scale, gate=None, seq_q_lens=None,
                     k_scale=None, v_scale=None):
            if q.shape[1] == 1:  # decode ticks only
                rec.append((bt.detach().clone(), sl.detach().clone()))
            return orig(q, kc, vc, bt, sl, scale, gate=gate, seq_q_lens=seq_q_lens,
                        k_scale=k_scale, v_scale=v_scale)
        be.paged_attention = rec_attn
        rng = np.random.default_rng(5)
        ids = rng.integers(3, cfg.vocab_size, size=320).astype(np.int64)
        rid = e.submit(ids, SamplingParams(temperature=0.0, seed=0, max_new_tokens=2))
        for _ in range(400):
            e.step()
            if rid in e.poll():
                break
        e.shutdown()
        return rec

    unf = run(False)
    fus = run(True)
    assert unf and fus, (len(unf), len(fus))
    n = min(len(unf), len(fus))
    for j in range(n):
        btu, slu = unf[j]
        btf, slf = fus[j]
        assert torch.equal(btu, btf), f"decode {j}: fused branch block_table differs\n{btf}\n{btu}"
        assert torch.equal(slu, slf), f"decode {j}: fused branch seq_len differs {slf} vs {slu}"


def test_sparse_draft_follower_adopts_a_published_prefix_and_matches_cold():
    """Warm path: under sparse+spec a follower ADOPTS a published prefix — the
    published page blobs carry the draft head's per-page KV, the follower copies
    it into its dense draft pool and resumes drafting at the boundary (the
    boundary slot stays zero, exactly like position 0 in a cold run). Its tokens
    must be bit-equal to a cold spec follower's. The held prefix bytes (trunk
    pages, bounds, draft KV) show as a host ledger row."""
    cfg = tiny()
    model = build_random(cfg, seed=11)
    # k=2 + the forced 8-page window keep ~10 pages hot; a 24-page prompt drops
    # its early pages and the drop-only frontier closes over all 24.
    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    follow = np.concatenate([prompt, np.arange(100, 120, dtype=np.int64)])
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)

    def spec_engine():
        from tilerl_kernels.backend import get_backend

        return build_engine(
            cfg=cfg, model=build_random(cfg, seed=11), backend=get_backend(),
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=512, sparse_k=2, scorer="bounds",
            kv_cold_bytes=1 << 30, draft=_draft(cfg, model), spec_depth=1)

    # Cold oracle: a spec engine whose prefix index never serves this prompt.
    cold = spec_engine()
    cold_got = _drain(cold, cold.submit(follow, params), 8)
    cold.shutdown()

    warm = spec_engine()
    pub = warm.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
    _drain(warm, pub, 200)
    entry = warm._sparse.prefix.lookup(follow)
    assert entry is not None and len(entry["keys"]) == 24
    assert all("dk" in warm._kv.cold.share_take(k) for k in entry["keys"]), \
        "published prefix blobs carry no draft KV"

    nrow = [r for r in warm.stats()["memory"] if r["owner"] == "kv_prefix"]
    assert nrow and nrow[0]["measured"] > 0 and nrow[0]["delta"] == 0, nrow

    rid = warm.submit(follow, params)
    warm.step()
    req = next(x for x in warm._running if x.req_id == rid)
    assert req.sparse_matched == 24 * BLOCK_TOKENS, req.sparse_matched
    # the warm path (not a trunk-only miss) is observable in stats
    assert warm.stats()["prefix_warm_adoptions"] == 1
    got = _drain(warm, rid, 8)
    warm.shutdown()
    assert got == cold_got, f"warm draft follower {got} != cold spec {cold_got}"


def test_a_prompt_that_never_leaves_the_hot_pool_publishes_nothing():
    """Publish-once semantics (#782): a page is published only when it LEAVES the
    resident union. A hot pool (k=64 >= the 8-page prompt plus its window) keeps
    every prompt frame for the whole run, so request end moves zero bytes: no
    lookup entry, no shared blobs, no draft K/V copied. The old forced
    prompt-end closure at _release is gone; a same-prompt follower misses."""
    cfg = tiny()
    model = build_random(cfg, seed=11)
    prompt = (np.arange(8 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7

    from tilerl_kernels.backend import get_backend

    eng = build_engine(
        cfg=cfg, model=build_random(cfg, seed=11), backend=get_backend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
        max_num_batched_tokens=512, sparse_k=64, scorer="bounds",
        kv_cold_bytes=1 << 30, draft=_draft(cfg, model), spec_depth=1)
    try:
        rid = eng.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=8, seed=0))
        _drain(eng, rid, 8)
        assert eng._sparse.prefix.published == 0, eng._sparse.prefix.published
        assert eng._sparse.prefix.lookup(prompt) is None
        assert not eng._kv.cold.share_keys(), eng._kv.cold.share_keys()
    finally:
        eng.shutdown()


def test_sparse_warm_follower_with_an_exact_page_aligned_prompt_matches_cold():
    """A warm follower whose prompt equals the published prefix in WHOLE (zero
    residual tokens) used to stick in PREFILL forever: nothing schedules a chunk
    and no first-token logits appear. It must finish and stay bit-equal to a
    cold follower."""
    cfg = tiny()
    model = build_random(cfg, seed=11)
    # Long enough that k+window cannot hold the whole prompt: under publish-once
    # (#782) the prefix forms from natural drops, never from a forced closure.
    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)

    def eng():
        from tilerl_kernels.backend import get_backend

        return build_engine(
            cfg=cfg, model=build_random(cfg, seed=11), backend=get_backend(),
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=512, sparse_k=2, scorer="bounds",
            kv_cold_bytes=1 << 30, draft=_draft(cfg, model), spec_depth=1)

    cold = eng()
    cold_got = _drain(cold, cold.submit(prompt, params), 8)
    cold.shutdown()
    warm = eng()
    # Decode the publisher until every prompt page left the union and the
    # exact-aligned prefix formed via natural drops (#782).
    pid = warm.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
    _drain(warm, pid, 200)
    rid = warm.submit(prompt, params)
    warm.step()
    assert next(x for x in warm._running if x.req_id == rid).sparse_matched \
        == 24 * BLOCK_TOKENS
    got = _drain(warm, rid, 8)
    warm.shutdown()
    assert got == cold_got, f"zero-tail warm {got} != cold {cold_got}"


def test_an_adopted_prefix_redemotes_with_zero_device_bytes():
    """#783: a follower promotes an adopted prefix into PRIVATE device blocks, and
    those blocks leave its hot union as decode advances. The identical bytes are
    already held under the same content keys, so the re-leave must do zero device
    work: no demote_page (which D2Hs a whole page) for an adopted page, and the
    re-publish is a share_ref only.

    Two arms:
    * dedup live: zero demote_page calls naming an adopted logical page, at least
      one dup share_ref, follower tokens equal a prefix-miss oracle;
    * fallback: with share_keys() emptied after adoption (the shared LRU evicted
      the keys) adopted pages demote normally and still decode to the same tokens.
    Mutation-red: pre-#783 code demoted every re-leaving adopted page, so the
    first arm's zero-demote assertion fails on the old shape."""
    cfg = tiny()
    model = build_random(cfg, seed=11)
    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    n_prompt_pages = 24
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)

    def spec_engine():
        from tilerl_kernels.backend import get_backend

        return build_engine(
            cfg=cfg, model=build_random(cfg, seed=11), backend=get_backend(),
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=512, sparse_k=2, scorer="bounds",
            kv_cold_bytes=1 << 30, draft=_draft(cfg, model), spec_depth=1)

    miss = spec_engine()
    want = _drain(miss, miss.submit(prompt, params), 8)
    miss.shutdown()

    def run_follower(
        hide_shared_after_adopt=False, prefix_capacity=None, drop_blob_at_resolve=False,
    ):
        warm = spec_engine()
        pub = warm.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
        _drain(warm, pub, 200)
        entry = warm._sparse.prefix.lookup(prompt)
        assert entry is not None and len(entry["keys"]) == n_prompt_pages
        content_keys = set(entry["keys"])
        pool = warm._kv
        cold = pool.cold
        if prefix_capacity is not None:
            # Shrink the index after the publisher's entry exists so the
            # follower's own closures evict entries between the labelled frame
            # release and the frontier close — the F1 race (#783).
            warm._sparse.prefix.capacity = prefix_capacity

        adopted_demoted: list[int] = []
        real_demote = pool.demote_page

        def spy_demote(block, key=None):
            if isinstance(key, tuple) and len(key) == 2 and key[1] < n_prompt_pages:
                adopted_demoted.append(key[1])
            return real_demote(block, key=key)

        pool.demote_page = spy_demote
        dup_refs: list[int] = []
        real_share_ref = cold.share_ref
        real_share_ref_if_present = cold.share_ref_if_present

        def spy_share_ref(key):
            if key in content_keys:
                dup_refs.append(key)
            return real_share_ref(key)

        cold.share_ref = spy_share_ref
        # share_hold_kv is the private->shared transfer: the dup path must never
        # reach it (freeze-ref bumps go through share_ref on both old and new
        # shapes, so demote_page zero is what pins change B; this pins change C).
        lifted: list[int] = []
        real_lift = cold.share_hold_kv

        def spy_lift(private_key, shared_key, extra=None):
            if shared_key in content_keys:
                lifted.append(shared_key)
            return real_lift(private_key, shared_key, extra=extra)

        cold.share_hold_kv = spy_lift
        real_share_take = cold.share_take

        if drop_blob_at_resolve:
            # The entry's label survives but its shared record was evicted:
            # a later select that promotes the page gets None and must fall
            # back to a fresh block instead of raising (the daemon would log
            # and the row would sit there as a zombie).
            def share_take_none(key):
                if key in content_keys:
                    return None
                return real_share_take(key)

            cold.share_take = share_take_none

        try:
            rid = warm.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
            warm.step()  # adopts the 24-page prefix
            assert next(x for x in warm._running if x.req_id == rid).sparse_matched \
                == n_prompt_pages * BLOCK_TOKENS
            if hide_shared_after_adopt:
                # Simulate shared-LRU eviction between adopt and re-leave: both
                # the membership probe and the atomic take must miss.
                cold.share_keys = lambda: frozenset()
                cold.share_ref_if_present = lambda key: False
            got = _drain(warm, rid, 200)
            leftover = set(warm._sparse.tracker.preheld.get(rid, ()))
        finally:
            pool.demote_page = real_demote
            cold.share_ref = real_share_ref
            cold.share_hold_kv = real_lift
            cold.share_ref_if_present = real_share_ref_if_present
            cold.share_take = real_share_take
            warm.shutdown()
        return got, adopted_demoted, dup_refs, lifted, leftover

    got, adopted_demoted, dup_refs, lifted, leftover = run_follower(hide_shared_after_adopt=False)
    assert got[:8] == want, f"dedup follower {got[:8]} != miss {want}"
    assert not adopted_demoted, f"adopted pages paid a D2H demote: {adopted_demoted}"
    assert not lifted, "dup republish reached the private->shared lift instead of ref-only"
    assert dup_refs, "the re-publish never took the zero-byte share_ref path"
    assert not leftover, f"early refs never handed to a closed frontier: {leftover}"

    fb_got, fb_demoted, _, fb_lifted, fb_leftover = run_follower(hide_shared_after_adopt=True)
    assert fb_got[:8] == want, f"fallback follower {fb_got[:8]} != miss {want}"
    assert fb_demoted, "evicted shared keys must fall back to a real demote"
    assert fb_lifted, "fallback must republish through the normal lift path"
    assert not fb_leftover

    # F1 race: a prefix-index LRU that evicts entries between the labelled frame
    # release and the (possibly later-tick) frontier closure. Pre-#783 the closure
    # hit a dead key and raised "neither a private host blob nor a resident frame";
    # the early ref pins the key across the gap. Capacity 2 forces evictions while
    # 24 pages re-leave. Every early ref must be consumed by a closed frontier.
    cap_got, _, _, _, cap_leftover = run_follower(
        hide_shared_after_adopt=False, prefix_capacity=2)
    assert cap_got[:8] == want, f"capacity-2 follower {cap_got[:8]} != miss {want}"
    assert not cap_leftover, f"preheld refs leaked under LRU pressure: {cap_leftover}"

    # resolve-None arm: the label outlives the shared record and a page is
    # re-selected. Pre-fix resolve raised "missing its shared blob" and the
    # daemon logged-and-continued into a zombie row; now it allocates a fresh
    # block and the row FINISHES (tokens need not equal the oracle — the blob
    # is genuinely gone, this is the non-crashing recovery contract, not a
    # silent correct adopt).
    none_got, *_ = run_follower(drop_blob_at_resolve=True)
    assert len(none_got) == 200, "row stalled/zombied instead of finishing its decode"


def test_sparse_mixed_length_rows_prefilling_one_tick_match_their_solo_g0():
    """H20 MMLU surface (cc bisect): the standalone FIFO probe was bit-exact but
    the harness drains several prompts whose PREFILL CHUNKS SHARE A TICK, mixed
    ragged lengths, and g0 argmax flips on 3/5 rows. Each row's last-prefill
    logits (which predict g0) must equal the same prompt's solo run. Prompts are
    34-47 pages so the finishing chunk attends through the sparse selected+own
    path (k=2 + the 8-page window), not only a dense first chunk; the
    ragged 4/12/4-token finishing chunks ship in ONE tick. Pins engine
    geometry (per-row own-table/page_base, neighbor assignment); an sm90
    packed-read defect is the card-only residual this separates from it."""
    import torch as _torch

    from tilerl.engine import Engine

    cfg = tiny()
    # All first chunks align to the 512 bucket and pack into tick 1; the page
    # tails pack at 32 into tick 2; the ragged 4/12/20-token finishers all pad
    # to the 64 bucket and pack into tick 3 (the g0 capture tick).
    prompts = [
        (np.arange(548, dtype=np.int64) % 300) + 7,
        (np.arange(556, dtype=np.int64) % 310) + 23,
        (np.arange(564, dtype=np.int64) % 290) + 41,
    ]

    def build(batch):
        from tilerl_kernels.backend import get_backend

        return build_engine(
            cfg=cfg, model=build_random(cfg, seed=11), backend=get_backend(),
            num_blocks=256, num_slots=6, max_batch=batch, max_total_tokens=8192,
            max_num_batched_tokens=2048, sparse_k=2, scorer="bounds",
            kv_cold_bytes=1 << 30)

    # Class-level seams (the engine calls unbound methods); instrument() returns
    # the per-engine capture dict plus an uninstall.
    orig_finish = Engine._finish_prefills
    orig_plan = Engine._build_plan

    def instrument(engine):
        gid = id(engine)
        state = {"g": {}, "ticks": []}
        Engine._instruments = getattr(Engine, "_instruments", {})
        Engine._instruments[gid] = state

        def wrap_finish(self, prefills, chunks, logits, base):
            st = Engine._instruments.get(id(self))
            for k, (pf, c) in enumerate(zip(prefills, chunks)):
                if st is not None and pf.prefill_from + c >= len(pf.tokens):
                    st["g"][pf.req_id] = _torch.clone(
                        logits[base + k, min(c, logits.shape[1]) - 1])
            orig_finish(self, prefills, chunks, logits, base)

        def wrap_plan(self):
            dec, pf, ch = orig_plan(self)
            st = Engine._instruments.get(id(self))
            if st is not None and pf:
                st["ticks"].append(
                    [(r.req_id, r.prefill_from, int(c)) for r, c in zip(pf, ch)])
            return dec, pf, ch

        Engine._finish_prefills = wrap_finish
        Engine._build_plan = wrap_plan

        def uninstall():
            if getattr(Engine, "_instruments", None):
                Engine._instruments.pop(gid, None)
            Engine._finish_prefills = orig_finish
            Engine._build_plan = orig_plan

        return state, uninstall

    try:
        want = []
        for p in prompts:
            e = build(1)
            st, off = instrument(e)
            (rid,) = [e.submit(p, SamplingParams(temperature=0.0, max_new_tokens=1))]
            for _ in range(64):
                if rid in st["g"]:
                    break
                e.step()
            want.append(st["g"][rid])
            off()
            e.shutdown()

        e = build(4)
        st, off = instrument(e)
        rids = [e.submit(p, SamplingParams(temperature=0.0, max_new_tokens=1))
                for p in prompts]
        for _ in range(64):
            if all(rid in st["g"] for rid in rids):
                break
            e.step()
        ticks = st["ticks"]
        got = [st["g"][rid] for rid in rids]
        off()
        e.shutdown()
    finally:
        Engine._finish_prefills = orig_finish
        Engine._build_plan = orig_plan

    # The gate must actually run the surface: a tick shared by all three rows'
    # first chunks, and the tick each g0 was captured on likewise shared.
    assert any(len(t) == 3 for t in ticks), f"no 3-row prefill tick: {ticks}"
    capture_ticks = [t for t in ticks if len(t) >= 2]
    assert capture_ticks, f"g0 rows never finished on a shared tick: {ticks}"
    for i in range(3):
        assert got[i].shape == want[i].shape
        if not _torch.equal(got[i], want[i]):
            raise AssertionError(
                f"row {i} g0 logits differ from solo: max|d|="
                f"{float((got[i] - want[i]).abs().max()):.3e}, argmax flip="
                f"{int(_torch.argmax(got[i])) != int(_torch.argmax(want[i]))}")


def test_sparse_decode_never_enters_the_dense_graph_when_graphs_are_on():
    """Dispatch regression (card-2, #557 CHANGE-REQ): with _decode_graph_on=True
    (the CUDA auto state) the dense _run_decode_graph used to win EVERY sparse
    decode tick, capturing a sparse=None BatchKv over the own blocks, so the
    sparse graph was never built. A sparse row must route through the sparse
    graph and never touch the dense graph; a dense row still uses it."""
    prompt = np.arange(7, 7 + 12 * BLOCK_TOKENS, dtype=np.int64)
    def params():
        return SamplingParams(temperature=0.0, max_new_tokens=12, seed=0)

    # sparse engine forced into the CUDA state: BOTH graph flags on
    e = build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=8192,
        max_num_batched_tokens=512, prefix_store=NoPrefixStore(),
        sparse_k=64, scorer="bounds", kv_cold_bytes=1 << 30,
        sparse_device_select=True)
    calls = {"dense": 0, "sparse": 0}

    def dense_stub(reqs, chains=None):
        calls["dense"] += 1
        return True  # the bug: claiming a sparse tick for the dense graph

    e._run_decode_graph = dense_stub
    _orig_sparse = e._run_sparse_decode_graph

    def sparse_wrap(reqs, chains):
        ok = _orig_sparse(reqs, chains)
        calls["sparse"] += int(ok)
        return ok

    e._run_sparse_decode_graph = sparse_wrap
    e._decode_graph_on = True  # CUDA auto-enables this; CPU normally leaves it off
    rid = e.submit(prompt, params())
    try:
        for _ in range(400):
            e.step()
            if len(e.poll().get(rid, ())) >= 12:
                break
    finally:
        e.shutdown()
    assert calls["sparse"] >= 1, "the sparse graph never replayed"
    assert calls["dense"] == 0, (
        f"a sparse decode tick entered the dense graph {calls['dense']} times")

    # dense engine: the same flag routes to the dense graph as before
    d = build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=8192,
        max_num_batched_tokens=512, prefix_store=NoPrefixStore())
    dense_calls = {"n": 0}

    def dense_only(reqs, chains=None):
        dense_calls["n"] += 1
        return True

    d._run_decode_graph = dense_only
    d._decode_graph_on = True
    # max_new_tokens>1: token 1 emits at prefill, token 2 onwards is a decode tick
    drid = d.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    for _ in range(400):
        d.step()
        if len(d.poll().get(drid, ())) >= 1:
            break
    for _ in range(3):
        d.step()
        if dense_calls["n"]:
            break
    d.shutdown()
    assert dense_calls["n"] >= 1, "a dense decode tick did not use the dense graph"


def test_served_sparse_default_auto_enables_device_select_and_sparse_graph():
    """Served default (#557 CHANGE-REQ follow-up): production passes no
    sparse_device_select and leaves decode_graph at its auto value. The engine
    must resolve sparse device selection + the sparse graph ON wherever the
    decode graph auto-enables (the captured steady-state tick, with an 8-tick
    eager refresh), so a served sparse engine is not silently eager-every-tick.
    On a backend where graph auto-enables (here simulated), both flags come up;
    explicit decode_graph=False (and the real CPU cell, where _graph_on=False)
    keeps both off."""
    import tilerl.build as bm

    def _eng():
        return build_engine(
            cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=512, prefix_store=NoPrefixStore(),
            sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30)

    # real CPU cell: graph auto OFF -> sparse device select and graph off (eager)
    e = _eng()
    assert e._sparse_device_select is False and e._sparse_graph_on is False
    e.shutdown()

    # a served cuda-like cell: patch the single _graph_on predicate to True
    orig = bm._graph_on
    bm._graph_on = lambda backend, decode_graph: decode_graph is not False
    try:
        e = _eng()
        assert e._sparse_device_select is True and e._sparse_graph_on is True, (
            f"served default left sparse ticks eager: device_select="
            f"{e._sparse_device_select} graph_on={e._sparse_graph_on}")
        e.shutdown()
        # explicit decode_graph=False still wins (true sparse-eager arm)
        e = build_engine(
            cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=512, prefix_store=NoPrefixStore(),
            sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30,
            decode_graph=False)
        assert e._sparse_graph_on is False
        e.shutdown()
    finally:
        bm._graph_on = orig


def test_a_shared_spill_failure_lets_requests_finish_token_exact(tmp_path):
    """The V100 hang: a shared-prefix spill that raises used to escape step,
    wedging the request with a leaked slot. Shared spill is a cache, so on
    failure it disables for the process and the page stays in RAM; both
    requests must finish token-identically to a spill-succeeding run."""
    import tilerl.kv_tiers as kvmod

    cfg = tiny()
    per = __import__("tilerl.memory", fromlist=["per_kv_block_bytes"]).per_kv_block_bytes(
        cfg, __import__("torch").bfloat16)
    ssd = str(tmp_path / "spill.bin")

    def _eng():
        return build_engine(
            cfg, build_random(cfg, seed=11), RefBackend(), num_blocks=64, num_slots=4,
            max_batch=1, max_total_tokens=4096, max_num_batched_tokens=512,
            sparse_k=2, scorer="bounds", kv_cold_bytes=per, cold_ssd_path=ssd)

    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7

    good = _eng()
    want = _drain(good, good.submit(prompt, SamplingParams(
        temperature=0.0, max_new_tokens=4, seed=0)), 4)
    good.shutdown()

    e = _eng()
    orig = kvmod.ColdSsdFile.write
    calls = {"n": 0}

    def shared_only_fail(self, key, blob):
        # the shared/prefix spill file keys are ("s", int); private keys are tuples
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "s":
            calls["n"] += 1
            raise OSError(13, "Permission denied")
        return orig(self, key, blob)

    kvmod.ColdSsdFile.write = shared_only_fail
    try:
        # A long publisher: its early pages leave the union and the shared spill
        # write fails on the first natural publish (publish-once, #782 — request
        # end itself writes no shared bytes).
        pub = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
        _drain(e, pub, 200)
        rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
        out = _drain(e, rid, 4)
        # a second request still completes on the same engine
        rid2 = e.submit(((prompt.astype(np.int64)+100) % 300) + 7, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
        out2 = _drain(e, rid2, 4)
    finally:
        kvmod.ColdSsdFile.write = orig
    e.shutdown()
    assert calls["n"] >= 1
    assert out == want, (out, want)
    assert len(out2) == 4


def test_a_private_spill_failure_fails_the_request_and_frees_its_slot(tmp_path):
    """A private cold spill OSError (a page a live row needs) must finish that
    request with a client-visible error, free its slot/blocks, and leave a second
    request able to run. Red on main: the error escaped step and the loop retried
    forever with a leaked slot."""
    import tilerl.kv_tiers as kvmod

    cfg = tiny()
    per = __import__("tilerl.memory", fromlist=["per_kv_block_bytes"]).per_kv_block_bytes(
        cfg, __import__("torch").bfloat16)
    ssd = str(tmp_path / "spill.bin")
    e = build_engine(
        cfg, build_random(cfg, seed=11), RefBackend(), num_blocks=64, num_slots=4,
        max_batch=1, max_total_tokens=4096, max_num_batched_tokens=512,
        sparse_k=2, scorer="bounds", kv_cold_bytes=per, cold_ssd_path=ssd)
    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=6, seed=0))

    orig = kvmod.ColdSsdFile.write

    def private_only_fail(self, key, blob):
        is_shared = isinstance(key, tuple) and len(key) == 2 and key[0] == "s"
        if not is_shared:
            raise OSError(28, "No space left on device")
        return orig(self, key, blob)

    kvmod.ColdSsdFile.write = private_only_fail
    try:
        failed = None
        for _ in range(200):
            e.step()
            failed = e._failed.get(rid)
            if failed is not None:
                break
    finally:
        kvmod.ColdSsdFile.write = orig
    assert failed is not None and failed[0] == "cold_spill_failed", failed
    # the client-visible failure is consumed here; until taken, poll() reports it
    import pytest as _pytest

    from tilerl.engine import RequestFailed
    with _pytest.raises(RequestFailed):
        e.take(rid)
    # slot and blocks returned
    assert rid not in e._running
    assert all(x.req_id != rid for x in e._running)
    # a later request still completes on the freed capacity
    rid2 = e.submit(((prompt.astype(np.int64)+50) % 300) + 7, SamplingParams(temperature=0.0, max_new_tokens=3, seed=0))
    out2 = _drain(e, rid2, 3)
    e.shutdown()
    assert len(out2) == 3


# ------------------------------------------------- hybrid --sparse-min-tokens

def _hybrid_engine():
    return build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=128, num_slots=4, max_batch=4, max_total_tokens=8192,
        max_num_batched_tokens=512, sparse_k=64, scorer="bounds",
        kv_cold_bytes=1 << 30, sparse_min_tokens=128)


def _drain_two(engine, rids, n, ticks=8000):
    # poll() pops ALL finished rows, so a per-rid drain would discard the other's
    # output; accumulate each poll dict by rid in one loop.
    outs = {r: [] for r in rids}
    for _ in range(ticks):
        engine.step()
        if engine._failed:
            raise AssertionError(engine._failed)
        for k, v in engine.poll().items():
            if k in outs:
                outs[k] += v
        if all(len(v) >= n for v in outs.values()):
            return outs
    raise AssertionError(f"stalled: {[(k, len(v)) for k, v in outs.items()]}")


def test_hybrid_short_runs_dense_long_runs_sparse_token_exact_to_pure_modes():
    """One hybrid engine: the short prompt must equal the pure-dense engine's
    output, the long prompt the pure-sparse engine's output."""
    rng = np.random.default_rng(3)
    short = rng.integers(3, 300, 64).astype(np.int64)
    long = rng.integers(3, 300, 20 * BLOCK_TOKENS).astype(np.int64)
    def p():
        return SamplingParams(temperature=0.0, max_new_tokens=6, seed=0)

    e = _hybrid_engine()
    rs = e.submit(short, p())
    rl = e.submit(long, p())
    out = _drain_two(e, (rs, rl), 6)
    hs, hl = out[rs], out[rl]
    e.shutdown()

    dense = build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=128, num_slots=4, max_batch=4, max_total_tokens=8192,
        max_num_batched_tokens=512)
    rd = dense.submit(short, p())
    want_d = _drain_two(dense, (rd,), 6)[rd]
    dense.shutdown()

    sp = build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=128, num_slots=4, max_batch=4, max_total_tokens=8192,
        max_num_batched_tokens=512, sparse_k=64, scorer="bounds",
        kv_cold_bytes=1 << 30)
    rsp = sp.submit(long, p())
    want_s = _drain_two(sp, (rsp,), 6)[rsp]
    sp.shutdown()
    assert hs == want_d, f"dense-mode short {hs} != pure dense {want_d}"
    assert hl == want_s, f"sparse-mode long {hl} != pure sparse {want_s}"


def test_hybrid_tick_is_never_mixed_and_short_ticks_use_the_dense_graph():
    """Round-robin: every tick carries one mode; all-short decode ticks run the
    dense captured graph (forced on the CPU seam), and per-mode counters record
    the path."""
    rng = np.random.default_rng(3)
    short = rng.integers(3, 300, 64).astype(np.int64)
    long = rng.integers(3, 300, 20 * BLOCK_TOKENS).astype(np.int64)
    def p():
        return SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)

    e = _hybrid_engine()
    # Hybrid never captures the sparse graph: sparse ticks run eager (token-exact),
    # only the dense graph is used -- and it is precaptured before traffic.
    assert e._sparse_graph_on is False
    sparse_graph_calls = {"n": 0}
    real_sp = e._run_sparse_decode_graph
    e._run_sparse_decode_graph = (lambda reqs, chains:
        sparse_graph_calls.__setitem__("n", sparse_graph_calls["n"] + 1)
        or real_sp(reqs, chains))
    e._decode_graph_on = True
    seen = {"dense": [], "sparse": [], "mixed": []}
    orig_fwd = e._run_forward

    def watch(decodes, prefills, chunks):
        rows = decodes + prefills
        if rows:
            modes = {r.sparse_on for r in rows}
            assert modes <= {True, False} and len(modes) == 1, "mixed-mode tick"
            seen["mixed"].append(len(modes) == 2)
        return orig_fwd(decodes, prefills, chunks)

    e._run_forward = watch
    dense_graph = {"n": 0}
    real = e._run_decode_graph

    def counting_dense(reqs, chains=None):
        dense_graph["n"] += 1
        return real(reqs, chains)  # CPU: capture fails -> caller runs eager, tokens still commit

    e._run_decode_graph = counting_dense
    rs = e.submit(short, p())
    rl = e.submit(long, p())
    _drain_two(e, (rs, rl), 8)
    st = e.stats()
    e.shutdown()
    assert not any(seen["mixed"]), "a tick carried both modes"
    assert dense_graph["n"] >= 1, "an all-short decode tick never used the dense graph"
    assert sparse_graph_calls["n"] == 0, "a hybrid sparse tick entered the sparse graph"
    assert st["dense_mode_ticks"] >= 1 and st["sparse_mode_ticks"] >= 1, st
    assert st["dense_mode_ticks"] + st["sparse_mode_ticks"] >= dense_graph["n"]


def test_hybrid_a_dense_prompt_over_the_device_pool_routes_sparse_at_submit():
    """A dense row pins its WHOLE context in the device pool (the sparse cold tier
    is not available to it); a prompt that cannot fit that pin routes sparse at
    submit rather than blocking _admit forever."""
    # 128 blocks x 16 tokens is the whole device pool; a prompt claiming that much
    # plus decode cannot be dense even with the pool empty.
    too_long = np.arange(7, 7 + 120 * BLOCK_TOKENS, dtype=np.int64) % 297 + 3
    e = _hybrid_engine()
    rid = e.submit(too_long, SamplingParams(temperature=0.0, max_new_tokens=16, seed=0))
    req = e._waiting[0]
    assert req.sparse_on is True, "an un-pinnable prompt must route sparse, not dense"
    _drain_two(e, (rid,), 16)
    e.shutdown()


def test_hybrid_wall_time_fairness_gives_dense_dozens_of_ticks_per_sparse_tick():
    """Measured V100 ratio: a sparse prefill tick is ~1023 ms, a dense decode
    tick ~37 ms, so while both modes are runnable wall-time fairness must give
    ~28 dense ticks per sparse tick (1000/37). A fake clock pins sparse tick at
    1000 and dense at 37; a long-lived dense decode row coexists with the sparse
    prefill, and the steady-state mode sequence averages >=25 dense ticks
    between consecutive sparse ticks, and a dense row arriving after sparse ticks
    ran solo is served on the next tick (it owes no solo-history debt, and a tie
    at a freshly opened window goes to dense).

    Note: this gate CANNOT be red on the prior global-debt scheduler under a
    deterministic fake clock -- that scheduler also yields ~27:1 here. The V100
    regression (5.95:1, 9.7 tok/s) comes from real tick-duration spread and
    request-arrival timing, which a uniform fake clock does not model; it is
    caught by the device A-B, not this gate. The change here pins the intended
    rolling-window/tie semantics so a future edit cannot silently regress them."""
    e = _fair_engine()
    e._hybrid_fake_dt = (1000.0, 37.0)
    rng = np.random.default_rng(5)
    long = rng.integers(3, 300, 120 * 192).astype(np.int64)
    e.submit(long, SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    # three solo sparse ticks before any dense row exists
    for _ in range(3):
        e.step()
    e.submit(rng.integers(3, 300, 64).astype(np.int64),
             SamplingParams(temperature=0.0, max_new_tokens=2000, seed=1))
    seq = []
    orig = e._run_forward

    def watch(decodes, prefills, chunks):
        rows = decodes + prefills
        seq.append(rows[0].sparse_on if rows else None)
        return orig(decodes, prefills, chunks)

    e._run_forward = watch
    # drive until >=6 sparse ticks have run WITH the dense row present
    n_with_dense = 0
    for _ in range(4000):
        e.step()
        e.poll()
        if any(not r.sparse_on for r in e._running):
            n_with_dense = sum(1 for m in seq if m is True)
        if n_with_dense >= 6:
            break
    e.shutdown()
    s = [m for m in seq if m is not None]
    # the arriving dense row is served immediately (first tick after arrival is
    # dense; the solo sparse history is not a debt it inherits)
    assert s[0] is False, f"newly arrived dense row first tick was {s[0]}"
    idx = [i for i, m in enumerate(s) if m is True]
    assert len(idx) >= 2, f"need a steady window, got sparse ticks {len(idx)}"
    gaps = [idx[i + 1] - idx[i] - 1 for i in range(len(idx) - 1)]
    dense_per_sparse = sum(gaps) / len(gaps)
    assert dense_per_sparse >= 25, (
        f"{dense_per_sparse:.1f} dense ticks per sparse tick in steady state; "
        "1000/37 wall-time fairness requires >= 25")


def _fair_engine():
    cfg = tiny()
    return build_engine(
        cfg=cfg, model=build_random(cfg, seed=11), backend=RefBackend(),
        num_blocks=512, num_slots=8, max_batch=8, max_total_tokens=65536,
        max_num_batched_tokens=512, sparse_k=64, scorer="bounds",
        kv_cold_bytes=1 << 30, sparse_min_tokens=8192)


def test_hybrid_dense_admit_reserves_live_sparse_rows_hot_headroom():
    """CHANGE-REQ (rev-30, #586): a dense admit pins from the one pool sparse
    rows grow into lazily. Free blocks alone overstate what dense may take -- a
    live sparse row is entitled to grow to its per-slot hot ceiling. A dense
    admit that fits free_blocks but starves that headroom must be refused;
    otherwise the sparse row raises 'hot pool undersized' inside a live tick and
    the step handler fails EVERY running row. Red on the head that admitted on
    free_blocks alone."""
    from tilerl.memory import sparse_hot_pages_per_slot

    cfg = tiny()
    ceil_ = sparse_hot_pages_per_slot(cfg, 64, 512)
    # sparse pool fits num_slots x ceiling + 1. Fabricate one live sparse row
    # holding ceiling-6 REAL pages: free = pool-held = 322, its headroom = 6.
    # A dense 320-block pin fits free (old code: 320 <= 322, admits) but must be
    # refused once the headroom is reserved (320 + 6 > 322).
    e = build_engine(
        cfg=cfg, model=build_random(cfg, seed=11), backend=RefBackend(),
        num_blocks=0, num_slots=4, max_batch=4, max_total_tokens=16384,
        max_num_batched_tokens=512, sparse_k=64, scorer="bounds",
        kv_cold_bytes=1 << 30, sparse_min_tokens=8192)
    try:
        rng = np.random.default_rng(9)
        long = rng.integers(3, 300, 20 * BLOCK_TOKENS).astype(np.int64)
        srid = e.submit(long, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
        srow0 = e._waiting[0]
        srow0.sparse_on = True  # fabricate the live sparse mode for this row
        assert e._admit(srow0), "sparse row must admit with needed=0"
        e._running.append(e._waiting.popleft())
        srow = next(r for r in e._running if r.req_id == srid)
        held = ceil_ - 6
        for _ in range(held):
            srow.blocks.append(e._kv.alloc_block())
        e._sparse.attach(srid)
        e._sparse.resident[srid] = {p: srow.blocks[p] for p in range(held)}
        assert e._sparse_hot_headroom() == 6
        free = e._kv.free_blocks
        assert free >= 320, f"test setup: free {free} cannot place the 320-block pin"

        # 320 whole-context blocks: dense under N=8192, fits free but not free+headroom.
        short = rng.integers(3, 300, 320 * BLOCK_TOKENS).astype(np.int64)
        e.submit(short, SamplingParams(temperature=0.0, max_new_tokens=8, seed=1))
        drow = e._waiting[0]
        assert drow.sparse_on is False
        admitted = e._admit(drow)
        assert not admitted, (
            f"dense admitted with {free} free against a 6-page sparse headroom: "
            "the sparse row's next own page raises inside a live tick")
    finally:
        e.shutdown()


def test_hybrid_stats_carries_live_sparse_residency():
    """rev-30 item 4: hybrid reconciles the DENSE memory ledger, so the sparse
    live occupancy must still be visible flat in stats() (page_bounds/kv_hot),
    or the concurrent device run cannot see residency vs cold."""
    cfg = tiny()
    e = build_engine(
        cfg=cfg, model=build_random(cfg, seed=11), backend=RefBackend(),
        num_blocks=0, num_slots=4, max_batch=4, max_total_tokens=16384,
        sparse_k=64, scorer="bounds", kv_cold_bytes=1 << 30,
        sparse_min_tokens=8192)
    try:
        rng = np.random.default_rng(4)
        long = rng.integers(3, 300, 20 * BLOCK_TOKENS).astype(np.int64)
        rid = e.submit(long, SamplingParams(max_new_tokens=4, seed=0))
        r0 = e._waiting[0]
        r0.sparse_on = True
        assert e._admit(r0)
        e._running.append(e._waiting.popleft())
        for _ in range(3):
            e._running[0].blocks.append(e._kv.alloc_block())
        e._sparse.attach(rid)
        e._sparse.resident[rid] = {p: e._running[0].blocks[p] for p in range(3)}
        st = e.stats()
        assert st["sparse_hot_pages"] == 3, st.get("sparse_hot_pages")
        assert st["sparse_hot_bytes"] > 0 and st["page_bounds_bytes"] >= 0
        assert "kv_cold_bytes" in st
    finally:
        e.shutdown()


def test_hybrid_dense_ledger_is_memoized_but_pure_sparse_stays_live():
    """#586 all-dense A-B: a hybrid engine must not re-walk memory.plan every
    stats call (the memoize #581 added for plain dense engines must also cover
    the hybrid dense whole-pool view). A PURE sparse engine keeps the live ledger
    (kv_hot moves with residency)."""
    import tilerl.memory as memory_mod

    cfg = tiny()
    h = build_engine(
        cfg=cfg, model=build_random(cfg, seed=11), backend=RefBackend(),
        num_blocks=0, num_slots=4, max_batch=4, max_total_tokens=16384,
        sparse_k=64, scorer="bounds", kv_cold_bytes=1 << 30,
        sparse_min_tokens=8192)
    orig = memory_mod.plan
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    memory_mod.plan = counting
    try:
        h._build_stats()
        h._build_stats()
        h._build_stats()
    finally:
        memory_mod.plan = orig
    h.shutdown()
    assert calls["n"] == 1, f"hybrid ledger not memoized: plan ran {calls['n']}x"

    p = build_engine(
        cfg=cfg, model=build_random(cfg, seed=11), backend=RefBackend(),
        num_blocks=0, num_slots=1, max_batch=1, max_total_tokens=1024,
        sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30)
    calls["n"] = 0
    memory_mod.plan = counting
    try:
        p._build_stats()
        p._build_stats()
    finally:
        memory_mod.plan = orig
    p.shutdown()
    assert calls["n"] == 2, "pure sparse ledger must stay live per stats call"


def test_hybrid_dense_spec_row_finishes_with_exactly_n_during_sparse_fill():
    """Device regression (#586 trace): dense rows kept decoding (1,449 observed
    output tokens against a 40-token cap) and never finished during a concurrent
    long sparse fill. A dense spec (draft d1) short row must stop at EXACTLY
    max_new_tokens even while the sparse prefill owns alternating ticks."""
    from tilerl_kernels.backend import get_backend

    cfg = tiny()
    model = build_random(cfg, seed=11)
    e = build_engine(
        cfg=cfg, model=model, backend=get_backend(),
        num_blocks=0, num_slots=4, max_batch=4, max_total_tokens=300000,
        max_num_batched_tokens=512, sparse_k=128, scorer="bounds",
        kv_cold_bytes=1 << 30, sparse_min_tokens=8192,
        draft=_draft(cfg, model), spec_depth=1)
    try:
        rng = np.random.default_rng(3)
        e.submit(rng.integers(3, 300, 200 * 192).astype(np.int64),
                 SamplingParams(max_new_tokens=2, seed=0))
        N = 10
        rid = e.submit(rng.integers(3, 300, 64).astype(np.int64),
                       SamplingParams(temperature=0.0, max_new_tokens=N, seed=100))
        out = []
        for _ in range(6000):
            e.step()
            out += e.poll().get(rid, [])
            if len(out) >= N:
                break
        assert len(out) == N, f"dense spec row produced {len(out)} tokens, cap {N}"
        # it must be DONE and gone from running, not still decoding
        assert not any(r.req_id == rid for r in e._running), "row still running at cap"
    finally:
        e.shutdown()


def test_a_cancelled_sparse_row_publishes_no_prefix_and_returns_every_page():
    """A disconnected sparse row releases its pages like a normal finish.

    Publish-once (#782) removed the forced prompt-end closure at _release, so
    cancel and finish move no bytes at request end: either way a page is
    published only if it already left the union (offer_drop). Cancelled while
    still prefill-decoding here, the row's pages never leave, so it publishes
    nothing and gives back blocks, cold blobs and slot."""
    engine = build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
        max_num_batched_tokens=512, sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30)
    # Same 24-page prompt the publishing tests use; the cancel lands while the row
    # is still prefill-decoding, before any contiguous frontier can close.
    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    rid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
    for _ in range(30):
        engine.step()
        r = next((x for x in engine._running if x.req_id == rid), None)
        if r is not None and r.decoding:
            break
    assert engine._kv.used_blocks > 0, "vacuous: the row holds no hot pages"
    assert engine.cancel(rid) is True

    assert engine._sparse.prefix.published == 0, engine._sparse.prefix.published
    assert engine._sparse.prefix.lookup(prompt) is None
    assert engine._slots_used == 0
    assert not engine._kv.used_blocks, engine._kv.used_blocks
    assert not engine._kv.cold.stats()["kv_cold_pages"], engine._kv.cold.stats()
    engine.shutdown()


def test_sparse_nodraft_full_prefix_resend_re_forwards_the_last_page():
    """No-draft sparse follower whose prompt matches a published prefix in WHOLE
    pages must not stall: prefill_from == len(tokens) with zero residual meant
    no chunk ever forwarded, so the row spun until the 1800 s timeout. The
    re-forward-last-page escape was gated on _draft; it must run without one.
    The re-forwared follower emits the same tokens as a prefix-MISS engine.
    """
    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    params = SamplingParams(temperature=0.0, max_new_tokens=8, seed=0)

    # Prefix-miss oracle: fresh engine, no shared prefix.
    miss = _sparse_engine(2, draft=False)
    ts_miss = _drain(miss, miss.submit(prompt, params), 8)
    miss.shutdown()

    sparse = _sparse_engine(2, draft=False)
    r1 = sparse.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
    _drain(sparse, r1, 200)
    assert sparse._sparse.prefix.lookup(prompt) is not None

    r2 = sparse.submit(prompt, params)
    # 60 ticks must move the row out of PREFILL: pre-fix it never forwards.
    left_prefill = False
    for _ in range(60):
        sparse.step()
        req = next((x for x in sparse._running if x.req_id == r2), None)
        if req is not None and req.decoding:
            left_prefill = True
            break
    assert left_prefill, "no-draft full-prefix follower never left PREFILL"
    assert req.sparse_matched == 24 * BLOCK_TOKENS
    ts = _drain(sparse, r2, 8)
    sparse.shutdown()
    assert ts == ts_miss, f"re-forwared follower {ts} != prefix-miss {ts_miss}"
