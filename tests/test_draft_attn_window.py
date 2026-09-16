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


def test_draft_window_cli_flag_precedence(monkeypatch):
    """The serve flag --draft-attn-window-tokens reaches the loaded draft head and
    overrides env/module default; omitting it leaves the head's own resolution
    (default 0 / env). Wiring + default-behavior gate; the frozen CLI surface is
    pinned separately in test_docs_links."""
    import inspect

    from tilerl import cli
    from tilerl.spec import DRAFT_ATTN_WINDOW_TOKENS_DEFAULT

    parser = cli._build_parser()
    omitted = parser.parse_args(["serve"])
    explicit = parser.parse_args(["serve", "--draft-attn-window-tokens", "2048"])
    assert omitted.draft_attn_window_tokens is None
    assert explicit.draft_attn_window_tokens == 2048
    # cmd_serve applies an explicit flag to the loaded draft and keeps None as
    # "don't override"; a negative value is rejected rather than read as a window.
    assert "draft.attn_window_tokens = args.draft_attn_window_tokens" in inspect.getsource(
        cli.cmd_serve)
    assert DRAFT_ATTN_WINDOW_TOKENS_DEFAULT == 0

    # Head resolution independent of the CLI: unset -> 0; env -> env; the CLI
    # override is a plain assignment on the already-loaded head.
    monkeypatch.delenv("TILERL_DRAFT_ATTN_WINDOW_TOKENS", raising=False)
    eng0, _ = _engine(0, monkeypatch)
    try:
        assert eng0._draft.attn_window_tokens == 0
    finally:
        eng0.shutdown()
    eng_env, _ = _engine(4096, monkeypatch)
    try:
        assert eng_env._draft.attn_window_tokens == 4096
        eng_env._draft.attn_window_tokens = 2048
        assert eng_env._draft.attn_window_tokens == 2048
    finally:
        eng_env.shutdown()


def test_real_accept_tick_sq2_window_engages_in_engine(monkeypatch):
    """Engine-level gate for the device-inert root cause. A random draft rarely
    matches the trunk, so force deterministic ACCEPTANCE: the draft's greedy and
    the trunk's verify sampler both return one constant token (temperature 0).
    After the first commit every verify tick then re-drafts q = n_ok + 1 = 2
    positions, which the old sq==1 guard silently bypassed. Drive real
    eng.step() decode ticks past the window and assert an engaged window on a
    genuine sq=2 row whose floor truncates (first>0)."""
    import torch as _t

    prompt = np.random.default_rng(2).integers(3, 320, size=120).astype(np.int64)
    eng, _ = _engine(32, monkeypatch)             # W=32 tokens (2 pages), context 120
    tok7 = 7
    try:
        backend = eng._backend
        orig_greedy, orig_sample = backend.greedy, backend.sample_batch

        def const_greedy(logits):
            t, p = orig_greedy(logits)
            return _t.full_like(t, tok7), p

        def const_sample(logits, *a, **k):
            t, lp = orig_sample(logits, *a, **k)
            return _t.full_like(t, tok7), lp

        backend.greedy = const_greedy
        backend.sample_batch = const_sample

        rid = eng.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=24,
                                                seed=0))
        engaged = []
        for _ in range(120):
            done = eng.poll()
            if rid in done and len(done[rid]) >= 24:
                break
            eng.step()
            st = eng._draft.read_window_stats()
            if st is not None:
                engaged.append(st)
        # At least one engaged view is exactly the post-accept shape: a single
        # decode row with q=2 whose floor truncates (>=1 prefix page dropped).
        assert any(
            st["sq"] == [2] and st["first"] == [f] and f > 0 and st["pages"][0]
            for st in engaged for f in [st["first"][0]]
        ), f"no engaged sq=2 truncating window across ticks: {engaged[:4]}"
    finally:
        backend.greedy = orig_greedy
        backend.sample_batch = orig_sample
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
    draft.width = 3
    nblk = 20                       # 320-token full context; last query at seq 320
    dblocks = [100 + i for i in range(nblk)]
    bt = torch.zeros(1, 64, dtype=torch.long)
    bt[0, :nblk] = torch.tensor(dblocks)
    sl, sq = [320], [1]
    full = BatchKv(block_table=bt, seq_len=torch.tensor(sl),
                   state_slot=torch.zeros(1, dtype=torch.long),
                   kv_pool=draft.kv, state_pool=None,
                   seq_q_lens=torch.tensor(sq))

    # Aligned windows (length=320, W a multiple of BLOCK): floor first page is
    # total-wp, windowed seq_len == W, last write maps to the same block/offset.
    # The windowed descriptor keeps the FULL table width nb=64 (the kernel's
    # compiled-in Mb -> no recompile); kept pages are packed from column 0 and the
    # remaining columns are zero.
    nb = bt.shape[1]
    for W_pages, W_tok in ((2, 32), (4, 64), (8, 128)):
        draft.attn_window_tokens = W_tok
        view = draft._windowed_read_kv(full, sl, sq, [dblocks])
        assert view is not None
        assert view.block_table.shape[1] == nb          # same Mb, packed from col 0
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
    # length=40,W=32 -> first=(40-32)//16=0, no page dropped -> None (full), yet the
    # oldest wanted token 8 is inside kept page 0 (the last-wp cut would have
    # started at page 1 = token 16 and dropped tokens 8..15).
    draft.attn_window_tokens = 32
    sl40 = [40]
    assert draft._windowed_read_kv(full, sl40, [1], [dblocks]) is None
    # genuinely truncating unaligned case: length=100 -> first=(68)//16=4, pages
    # 4..6 cover tokens 64..99, including the oldest wanted token length-W=68.
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
    page 0). Prefill/catch-up rows are flagged per row (``decode=False``), not by
    sq: any one of them bypasses the window for the whole batch."""
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
    draft.width = 2
    view = draft._windowed_read_kv(full, sl, sq, [db0, db1])
    assert view is not None
    # long row: floor first=(120-32)//16=5 -> pages 5,6,7 (keeps the page holding
    # token hi-W=88; not a hard last-wp cut), wsl=120-5*16=40
    assert [int(x) for x in view.block_table[0, :3].tolist()] == db0[5:]
    assert int(view.seq_len[0]) == 40
    # short row: first=(40-32)//16=0 -> ALL 3 pages, true seq_len 40, never zeroed
    assert [int(x) for x in view.block_table[1, :3].tolist()] == db1
    assert int(view.seq_len[1]) == 40

    # any PREFILL/catch-up row in the batch (decode=False) -> whole batch bypasses,
    # regardless of that row's sq.
    assert draft._windowed_read_kv(
        full, sl, [1, 4], [db0, db1], decode=[True, False]) is None
    # ... a decode verify TAIL (q <= draft.width) windows regardless of q: with
    # width=2, q=2 alongside q=1 engages (sq alone no longer bails).
    assert draft._windowed_read_kv(
        full, sl, [1, 2], [db0, db1], decode=[True, True]) is not None
    # ... but a decode-phase CATCH-UP row (q > width, the hidden-gap case) stays
    # full-prefix even though its phase is decode.
    assert draft._windowed_read_kv(
        full, sl, [1, 4], [db0, db1], decode=[True, True]) is None


# ---------------------------------------------------------------------------
# Verify-tick gates (#668 follow-up).
#
# Root cause being pinned: a served spec verify tick re-drafts the bonus token
# plus the new chain in ONE forward, so its row spans q = n_ok + 1 positions:
# ACCEPT (the common case) -> q = 2 at spec_depth=1; reject -> q = 1. The #668
# guard ``any(q != 1 for q in sq)`` therefore returned None on almost every real
# tick, so the window never engaged on device and the W sweep was bit-identical.
# The fix windows DECODE rows per row regardless of q (the verify tail reads the
# same trailing history; uniform renumbering preserves the multi-query causal
# mask), and bails the whole batch only when a row is still PREFILLING (signalled
# per row via decode=False, not by sq — a verify tail can itself carry q>1).
# ---------------------------------------------------------------------------


def _bare_draft():
    cfg = tiny()
    backend = get_backend()
    draft = DraftHead.__new__(DraftHead)
    draft.kv = PagedKvPool(64, cfg.num_kv_heads, cfg.head_dim, num_layers=1,
                           device=backend.device, layer_map=(0,))
    draft.width = 2  # spec_depth=1: one committed token + one draft
    return draft


def _full_view(draft, blocks_per_row, sl, sq):
    tables = []
    for blocks in blocks_per_row:
        bt = torch.zeros(1, 64, dtype=torch.long)
        bt[0, : len(blocks)] = torch.tensor(blocks)
        tables.append(bt)
    bt = tables[0] if len(tables) == 1 else torch.cat(tables, 0)
    return BatchKv(block_table=bt, seq_len=torch.tensor(sl),
                   state_slot=torch.zeros(len(sl), dtype=torch.long),
                   kv_pool=draft.kv, state_pool=None,
                   seq_q_lens=torch.tensor(sq))


def test_verify_tick_accepted_row_sq2_window_engages():
    """An accepted verify tick drafts q=2 tail positions. The window MUST engage:
    today the sq==1 guard returns None (the device-inert bug).

    hi=120 (seq_len 121), lo=119, q=2, W=32 -> floor first=(120-32)//16=5,
    pages 5..7 cover global tokens 80..120, windowed seq_len=121-80=41 so
    hist'=41-2=39 == lo-80 (causal history length preserved under renumber)."""
    draft = _bare_draft()
    db = [100 + i for i in range(8)]          # 128 logical slots; hi=120 -> 8 pages
    sl, sq = [121], [2]
    full = _full_view(draft, [db], sl, sq)
    draft.attn_window_tokens = 32
    view = draft._windowed_read_kv(full, sl, sq, [db])
    assert view is not None, "accept tick (sq=2) must window, not bypass"
    assert [int(x) for x in view.block_table[0, :3].tolist()] == db[5:8]
    assert int(view.block_table[0, 3:].abs().sum()) == 0
    assert int(view.seq_len[0]) == 41
    # last query (global hi=120) maps to the SAME physical page/in-page offset
    wsl = int(view.seq_len[0])
    assert int(view.block_table[0, (wsl - 1) // BLOCK_TOKENS]) == db[-1]
    assert (wsl - 1) % BLOCK_TOKENS == (sl[0] - 1) % BLOCK_TOKENS
    # renumbered history equals the true tail-query history length
    assert int(wsl - sq[0]) == 121 - sq[0] - 5 * BLOCK_TOKENS


def test_mixed_verify_batch_sq2_and_sq1_each_windowed_or_full():
    """One ACCEPT row (sq=2, long) batched with one REJECT row (sq=1, short) in
    the same forward (the real multi-row serving shape, missing from #668): the
    long row truncates, the short row keeps ALL its own pages/seq_len and is
    never zero-filled to page 0."""
    draft = _bare_draft()
    db0 = [200 + i for i in range(8)]          # long row, hi=120 seq 121
    db1 = [300 + i for i in range(3)]          # short reject row, seq 40
    sl, sq = [121, 40], [2, 1]
    full = _full_view(draft, [db0, db1], sl, sq)
    draft.attn_window_tokens = 32
    view = draft._windowed_read_kv(full, sl, sq, [db0, db1])
    assert view is not None
    # accept row: floor first=5 -> pages 5,6,7 and wsl=41
    assert [int(x) for x in view.block_table[0, :3].tolist()] == db0[5:]
    assert int(view.seq_len[0]) == 41
    # reject row: hi=40<=W-ish -> first=0, keeps ALL 3 pages, true seq_len
    assert [int(x) for x in view.block_table[1, :3].tolist()] == db1
    assert int(view.seq_len[1]) == 40
    assert int(view.block_table[1, 3:].abs().sum()) == 0


def test_per_row_floor_never_splits_across_page_boundary():
    """Unaligned hi keeps the PAGE holding the oldest wanted token hi-W for a
    verify-tail row, and does not read the page before that floor."""
    draft = _bare_draft()
    db = [100 + i for i in range(8)]
    # hi=100 (seq 101), q=2 (lo=99), W=32 -> first=(100-32)//16=4, pages 4..6
    sl, sq = [101], [2]
    full = _full_view(draft, [db], sl, sq)
    draft.attn_window_tokens = 32
    view = draft._windowed_read_kv(full, sl, sq, [db])
    assert view is not None
    first = (100 - 32) // BLOCK_TOKENS
    tot = (101 + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    assert [int(x) for x in view.block_table[0, : tot - first].tolist()] == db[first:tot]
    assert first * BLOCK_TOKENS <= 101 - sq[0]          # floor at/before oldest query
    assert int(view.seq_len[0]) == 101 - first * BLOCK_TOKENS


def test_window_shorter_than_tail_span_keeps_that_row_full():
    """If the trailing window would start AFTER a decode row's oldest tail query
    (window shorter than the verify span), that row is kept FULL rather than
    dropping its queries' own causal context (its floor is clamped to 0), so the
    whole single-row batch reads unwindowed. Not reachable at the production W
    (hundreds of tokens vs a handful of tail queries), but it is the correctness
    rule."""
    draft = _bare_draft()
    draft.width = 16
    db = [100 + i for i in range(13)]         # length=201
    sl, sq = [201], [16]                       # long tail, lo=185
    full = _full_view(draft, [db], sl, sq)
    draft.attn_window_tokens = 4              # window < tail span
    # floor would be (201-4)//16=12 -> 192 > lo 185, so it is clamped to 0 and no
    # page truncates -> None (full descriptor).
    assert draft._windowed_read_kv(full, sl, sq, [db]) is None


def test_prefill_or_catchup_row_bypasses_with_verify_tail_present():
    """A row outside the verify-tail shape in the batch keeps the conservative
    whole-batch bypass even when another row is a windowable verify tail. Both
    exclusions are covered: a still-PREFILLING row (decode=False) and a
    decode-phase CATCH-UP row (q > width)."""
    draft = _bare_draft()                        # width=2
    db0 = [200 + i for i in range(8)]
    db1 = [300 + i for i in range(8)]
    sl, sq = [121, 121], [2, 8]
    full = _full_view(draft, [db0, db1], sl, sq)
    draft.attn_window_tokens = 32
    # row1 still prefilling
    assert draft._windowed_read_kv(
        full, sl, sq, [db0, db1], decode=[True, False]) is None
    assert draft.read_window_stats() is None
    # row1 phase=decode but CATCHING UP (q=8 > width=2): also full prefix
    assert draft._windowed_read_kv(
        full, sl, sq, [db0, db1], decode=[True, True]) is None
    # control: make row1 a real verify tail (width>=8, q=8) and it windows
    draft.width = 8
    assert draft._windowed_read_kv(
        full, sl, sq, [db0, db1], decode=[True, True]) is not None


def test_read_window_stats_observable_when_engaged():
    """Self-proof observability: when the env window engages on a verify tick the
    actual per-row windowed pages/seq_len/sq are externally readable; with the
    window off it reports None. Pins the interface BEFORE the src implements it
    (red), so perf1 can visually confirm truncation before re-sweeping."""
    draft = _bare_draft()
    db = [100 + i for i in range(8)]
    sl, sq = [121], [2]
    full = _full_view(draft, [db], sl, sq)

    draft.attn_window_tokens = 0
    assert draft._windowed_read_kv(full, sl, sq, [db]) is None
    off = draft.read_window_stats()
    assert off is None

    draft.attn_window_tokens = 32
    draft._windowed_read_kv(full, sl, sq, [db])
    stats = draft.read_window_stats()
    assert stats is not None
    assert stats["window_tokens"] == 32
    assert stats["sq"] == [2]
    assert stats["seq_len"] == [41]
    assert stats["pages"][0] == db[5:8]
