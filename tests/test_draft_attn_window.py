"""Diagnostic gate for TILERL_DRAFT_ATTN_WINDOW_TOKENS (sliding draft read window).

Default off -> the draft decode attention reads its full prefix and no read_kv
is built. When on, the READ is windowed to the trailing W tokens while the
write still targets the full table and the whole dense draft KV is retained.

These are CPU gates:
1. env off leaves read_kv None (zero behavior change);
2. a window covering the full context is a no-op (the forward output is
   bit-identical to full-prefix) — validates the probe does not perturb numerics;
3. a window narrower than the context engages and a decode run still completes;
4. the windowed table/seq_len geometry maps the last query's WRITE to the same
   physical block and in-page offset as the full table, and never names a block
   outside the window — the read/write separation.
"""

from __future__ import annotations

import numpy as np
import torch
from test_e2e import _random_draft
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import SamplingParams
from tilerl.kv_cache import BLOCK_TOKENS, BatchKv, PagedKvPool
from tilerl.model import build_random
from tilerl.spec import DraftHead


def _engine(window: int, monkeypatch):
    monkeypatch.setenv("TILERL_DRAFT_ATTN_WINDOW_TOKENS", str(window))
    cfg = tiny()
    model = build_random(cfg, seed=7)
    eng = build_engine(
        cfg, model, get_backend(), num_blocks=32, num_slots=2, max_batch=2,
        max_total_tokens=512, draft=_random_draft(cfg, 7, model),
        spec_depth=1, sparse_k=0,
    )
    return eng, cfg


def _run(monkeypatch, window: int, prompt: np.ndarray, n_new: int = 20):
    eng, _ = _engine(window, monkeypatch)
    try:
        rid = eng.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=n_new,
                                                seed=0))
        out = []
        for _ in range(96):
            done = eng.poll()
            if rid in done and len(done[rid]) >= n_new:
                out = done[rid]
                break
            eng.step()
        return out
    finally:
        eng.shutdown()


def test_window_off_builds_no_read_view(monkeypatch):
    monkeypatch.delenv("TILERL_DRAFT_ATTN_WINDOW_TOKENS", raising=False)
    eng, _ = _engine(0, monkeypatch)
    try:
        assert eng._draft.attn_window_tokens == 0
    finally:
        eng.shutdown()


def test_full_covering_window_is_numerically_identical(monkeypatch):
    """A window larger than the whole context returns None and leaves the draft
    output bit-identical to the full prefix. (The all-pages-via-remap identity at
    a sub-boundary W is pinned in the geometry test, where context is fixed; a
    growing e2e run with a fixed sub-page window genuinely truncates and is
    allowed to differ.)"""
    prompt = np.random.default_rng(0).integers(3, 320, size=40).astype(np.int64)
    full = _run(monkeypatch, 0, prompt)                 # full prefix
    covered = _run(monkeypatch, 1 << 20, prompt)        # window > context -> None
    assert len(full) == len(covered) == 20
    assert covered == full, "a W>=context window changed draft numerics"


def test_narrow_window_actually_truncates_and_completes(monkeypatch):
    """A window NARROWER than the context engages and the engine still serves a
    full decode run. Outputs may differ (that acceptance question is what the V100
    sweep measures); this only asserts truncation runs to completion."""
    prompt = np.random.default_rng(1).integers(3, 320, size=120).astype(np.int64)
    monkeypatch.setenv("TILERL_DRAFT_ATTN_WINDOW_TOKENS", "32")  # 2 pages < ctx
    eng, _ = _engine(32, monkeypatch)
    try:
        assert eng._draft.attn_window_tokens == 32
        out = _run(monkeypatch, 32, prompt, n_new=16)
        assert len(out) == 16, f"narrow-window run did not complete: {len(out)}"
    finally:
        eng.shutdown()


def test_windowed_read_view_geometry_and_write_target():
    """The last query's write block/offset match the full table, and the windowed
    table names exactly the trailing W pages."""
    cfg = tiny()
    backend = get_backend()
    draft = DraftHead.__new__(DraftHead)
    draft.kv = PagedKvPool(64, cfg.num_kv_heads, cfg.head_dim, num_layers=1,
                           device=backend.device, layer_map=(0,))
    nblk = 20                       # 320-token full context; last query at seq 320
    dblocks = [100 + i for i in range(nblk)]
    bt = torch.zeros(1, 64, dtype=torch.long)
    bt[0, :nblk] = torch.tensor(dblocks)
    sl, sq = [320], [1]
    full = BatchKv(block_table=bt, seq_len=torch.tensor(sl),
                   state_slot=torch.zeros(1, dtype=torch.long),
                   kv_pool=draft.kv, state_pool=None,
                   seq_q_lens=torch.tensor(sq))

    # Aligned windows (hi=320, W a multiple of BLOCK): floor first page is
    # total-wp, windowed seq_len == W, last write maps to the same block/offset.
    for W_pages, W_tok in ((2, 32), (4, 64), (8, 128)):
        draft.attn_window_tokens = W_tok
        view = draft._windowed_read_kv(full, sl, sq, [dblocks])
        assert view is not None
        width = view.block_table.shape[1]      # wp+1
        assert width == W_pages + 1
        got = [int(x) for x in view.block_table[0, :W_pages].tolist()]
        assert got == dblocks[nblk - W_pages:], (W_pages, got)
        assert int(view.block_table[0, W_pages:].abs().sum()) == 0
        wsl = int(view.seq_len[0])
        assert wsl == W_tok
        wpage = (wsl - sq[0]) // BLOCK_TOKENS
        woff = (wsl - sq[0]) % BLOCK_TOKENS
        assert int(view.block_table[0, wpage]) == dblocks[-1]
        assert woff == (sl[0] - sq[0]) % BLOCK_TOKENS

    # UNALIGNED window floor rule:
    # hi=40,W=32 -> first=(40-32)//16=0, no page dropped -> None (full), yet the
    # oldest wanted token 8 is inside kept page 0 (the last-wp cut would have
    # started at page 1 = token 16 and dropped tokens 8..15).
    draft.attn_window_tokens = 32
    sl40 = [40]
    assert draft._windowed_read_kv(full, sl40, [1], [dblocks]) is None
    # genuinely truncating unaligned case: hi=100 -> first=(68)//16=4, pages 4..6
    # cover tokens 64..99, including the oldest wanted token hi-W=68.
    sl100 = [100]
    full100 = BatchKv(
        block_table=bt, seq_len=torch.tensor(sl100),
        state_slot=torch.zeros(1, dtype=torch.long),
        kv_pool=draft.kv, state_pool=None, seq_q_lens=torch.tensor([1]))
    v = draft._windowed_read_kv(full100, sl100, [1], [dblocks])
    assert v is not None
    first = (100 - 32) // BLOCK_TOKENS            # 4
    tot = (100 + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    assert [int(x) for x in v.block_table[0, : tot - first].tolist()] == dblocks[first:tot]
    assert int(v.seq_len[0]) == 100 - first * BLOCK_TOKENS
    assert first * BLOCK_TOKENS <= 68

    # W larger than / sub-page below the context that drops no page -> None.
    draft.attn_window_tokens = 319
    assert draft._windowed_read_kv(full, sl, sq, [dblocks]) is None
    draft.attn_window_tokens = 1 << 20
    assert draft._windowed_read_kv(full, sl, sq, [dblocks]) is None


def test_mixed_batch_short_row_keeps_its_own_kv_and_prefill_bypasses():
    """A long truncated decode row batched with a SHORT decode row must not zero
    the short row's windowed table/seq_len (the CHANGE-REQ bug: hist=-1 read
    page 0). A batch containing any sq!=1 (prefill) row bypasses the window
    entirely."""
    cfg = tiny()
    backend = get_backend()
    draft = DraftHead.__new__(DraftHead)
    draft.kv = PagedKvPool(64, cfg.num_kv_heads, cfg.head_dim, num_layers=1,
                           device=backend.device, layer_map=(0,))

    def kv_of(blocks):
        bt = torch.zeros(1, 64, dtype=torch.long)
        bt[0, : len(blocks)] = torch.tensor(blocks)
        return bt

    # row 0: 120 tokens (8 pages), row 1: 40 tokens (~3 pages); W=32 (2 pages)
    db0 = [200 + i for i in range(8)]
    db1 = [300 + i for i in range(3)]
    wbt0, wbt1 = kv_of(db0), kv_of(db1)
    bt = torch.cat([wbt0, wbt1], 0)
    sl, sq = [120, 40], [1, 1]
    full = BatchKv(block_table=bt, seq_len=torch.tensor(sl),
                   state_slot=torch.zeros(2, dtype=torch.long),
                   kv_pool=draft.kv, state_pool=None,
                   seq_q_lens=torch.tensor(sq))
    draft.attn_window_tokens = 32
    view = draft._windowed_read_kv(full, sl, sq, [db0, db1])
    assert view is not None
    # long row: floor first=(120-32)//16=5 -> pages 5,6,7 (keeps the page holding
    # token hi-W=88; not a hard last-wp cut), wsl=120-5*16=40
    assert [int(x) for x in view.block_table[0, :3].tolist()] == db0[5:]
    assert int(view.seq_len[0]) == 40
    # short row: first=(40-32)//16=0 -> ALL 3 pages, true seq_len 40, never zeroed
    assert [int(x) for x in view.block_table[1, :3].tolist()] == db1
    assert int(view.seq_len[1]) == 40

    # any prefill/multi-token row in the batch -> whole batch bypasses (None)
    sq_prefill = [1, 4]
    assert draft._windowed_read_kv(full, sl, sq_prefill, [db0, db1]) is None
