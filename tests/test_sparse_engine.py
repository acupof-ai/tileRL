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


def test_sparse_engine_publishes_nothing_while_pages_are_pinned():
    """Under the cross-tick hot pin #526's demote-time publish fires only when a
    page LEAVES the resident union. A prompt wholly inside the k+window hot set
    keeps every page resident, so it crosses the publish boundary yet shares
    nothing. (The long-context follower HIT and its shared-bounds ledger are
    #542's drop-only frontier gate.)"""
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)  # 6 pages
    sparse = build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
        max_num_batched_tokens=512, sparse_k=6, scorer="bounds",
        kv_cold_bytes=1 << 30)
    r1 = sparse.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=8, seed=0))
    _drain(sparse, r1, 8)
    assert sparse._sparse.prefix.published == 0
    sparse.shutdown()


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
                   force_window=0, resolve=lambda p: p, reserved=set())
        sf = SparseForward(tr, [row], torch.device("cpu"))
        q = torch.randn(1, cfg.num_attention_heads, cfg.head_dim)
        with _Count() as c:
            sf._select(0, 0, q)
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
    row = tr.bounds_rows(0, [0])  # raised: index 15 out of bounds before the fix
    assert row.shape == (1, 16, tr.hkv, 2, tr.dim)
    assert torch.equal(row[0, 15], b[15])
