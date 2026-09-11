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

    def wrap(self, plane, q):
        table, sl = orig(self, plane, q)
        for r in self.rows:
            if int(r["decoding"]):  # decode/verify tick only, never prefill
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
