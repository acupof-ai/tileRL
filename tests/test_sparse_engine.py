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

from tilerl.config import tiny
from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.model import build_random
from tilerl.testing import RefBackend


def _engine(sparse: bool, k: int = 0):
    kw = dict(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
        max_num_batched_tokens=512, prefix_store=NoPrefixStore())
    if sparse:
        kw.update(sparse_k=k, scorer="bounds", kv_cold_bytes=1 << 30)
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


def test_sparse_demotes_and_promotes_pages_every_tick():
    """k smaller than the context: pages move to the host and back, bounds stay
    resident, and the run still produces finite output."""
    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)  # 6 pages
    sparse = _engine(True, 2)
    rid = sparse.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))

    saw_demote = saw_promote = saw_cold = False
    for _ in range(128):
        done = sparse.poll()
        if rid in done and len(done[rid]) >= 4:
            break
        r0 = next((x for x in sparse._running if x.req_id == rid), None)
        sparse.step()
        cold = sparse._kv.cold
        saw_demote |= cold.demotions > 0
        saw_promote |= cold.promotions > 0
        saw_cold |= cold.bytes_held > 0
        if r0 is not None and r0.phase == 2:
            # decode tick: bounds for every complete page survive demotion
            assert len(sparse._sparse.bounds[rid]) >= 5
    else:
        raise TimeoutError

    tok = done[rid][:4]
    sparse.shutdown()
    assert saw_demote and saw_promote and saw_cold, (
        f"tiering did not cycle: demote={saw_demote} promote={saw_promote} cold={saw_cold}")
    assert len(tok) == 4 and all(isinstance(t, int) for t in tok)


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


def test_sparse_engine_runs_with_the_default_prefix_store():
    """Regression (5f CHANGE-REQ): build_engine's DEFAULT store is the real
    PrefixStore, which publishes req.blocks[:N] at prefill boundaries. Sparse
    demotes pages out of req.blocks, so without forcing NoPrefixStore a
    >=64-token (4-page, first publish boundary) request died
    ``64 tokens need 4 blocks, got 0``. The default store must be coerced and a
    long request run through prefill AND decode."""
    from tilerl.kv_cache import PrefixStore  # noqa: F401  (documents the default)

    # No prefix_store arg: build_engine would normally build PrefixStore.
    engine = build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
        max_num_batched_tokens=512, sparse_k=2, scorer="bounds",
        kv_cold_bytes=1 << 30)
    # First publish boundary is at 64 tokens; use a longer prompt to cross it.
    prompt = np.arange(3, 3 + 5 * BLOCK_TOKENS, dtype=np.int64)  # 80 tokens, 5 pages
    rid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    tok = _drain(engine, rid, 4)
    engine.shutdown()
    assert len(tok) == 4 and all(isinstance(t, int) for t in tok)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
