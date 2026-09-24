"""The sparse window and refresh interval are settable at build, process-wide.

Both are module constants (`sparse_index.WINDOW_PAGES`, derived from
`WINDOW_TOKENS`; `sparse_engine.SPARSE_REFRESH_TICKS`) with several consumers:
the page-pool ledger in `memory`, the sparse tick's own width, and the captured
graph key. A flag that reached only some of them would size one pool and key one
graph at a different geometry than the tick it serves, which is why the override
is applied once, at `build_engine`, rather than at each read.

The gate is a non-default value arriving at every consumer, plus a control that
the same reading is RED when the value is not passed — a gate that passes both
ways is not measuring the flag.
"""

from __future__ import annotations

import pytest

from tilerl import sparse_engine, sparse_index
from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import build_random
from tilerl.testing import RefBackend

_WINDOW = 1024          # 64 pages, not the 128/8 default
_REFRESH = 32           # not the 8 default


def _build(**kw):
    return build_engine(
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
        **kw,
    )


@pytest.fixture(autouse=True)
def _restore_geometry():
    """These are process-wide module globals; leave them as found or every later
    test in the same process inherits a 1024-token window."""
    win, pages = sparse_index.WINDOW_TOKENS, sparse_index.WINDOW_PAGES
    eng_pages, refresh = sparse_engine.WINDOW_PAGES, sparse_engine.SPARSE_REFRESH_TICKS
    yield
    sparse_index.WINDOW_TOKENS, sparse_index.WINDOW_PAGES = win, pages
    sparse_engine.WINDOW_PAGES, sparse_engine.SPARSE_REFRESH_TICKS = eng_pages, refresh


def test_non_default_geometry_reaches_every_consumer():
    e = _build(sparse_window_tokens=_WINDOW, sparse_refresh_ticks=_REFRESH)
    try:
        pages = _WINDOW // 16
        # 1. the module the pool ledger and the scorers read
        assert sparse_index.WINDOW_TOKENS == _WINDOW
        assert pages == sparse_index.WINDOW_PAGES
        # 2. sparse_engine imported WINDOW_PAGES BY VALUE, so it needs its own write
        assert pages == sparse_engine.WINDOW_PAGES
        # 3. the refresh cadence the tick reads
        assert sparse_engine.SPARSE_REFRESH_TICKS == _REFRESH
        # 4. the built engine's own pool geometry: the ledger sized this pool from
        #    WINDOW_PAGES, so a pool that did not move means the override missed it.
        from tilerl.memory import sparse_pool_num_blocks

        tiny_blocks = sparse_pool_num_blocks(tiny(), 4, 2, 512)
        assert e._kv.num_blocks == tiny_blocks, (
            f"pool {e._kv.num_blocks} blocks != ledger {tiny_blocks}; the window "
            "override did not reach memory.sparse_pool_num_blocks")
    finally:
        e.shutdown()


def test_the_gate_is_red_without_the_flag():
    """Negative control on the SAME assertion as the gate above.

    Without the flag the module values stay at the default, so a gate that reads
    `== _WINDOW` here fails. This is what says the gate above is measuring the
    flag and not a constant that happens to match.
    """
    e = _build()
    try:
        assert sparse_index.WINDOW_PAGES == 8, "default window moved"
        assert sparse_engine.SPARSE_REFRESH_TICKS == 8, "default refresh moved"
        assert sparse_index.WINDOW_PAGES != _WINDOW // 16
        assert sparse_engine.SPARSE_REFRESH_TICKS != _REFRESH
    finally:
        e.shutdown()


def test_a_partial_window_is_refused():
    """A window that is not a whole number of pages would silently truncate to
    one, so it is refused rather than rounded."""
    with pytest.raises(ValueError, match="whole number"):
        _build(sparse_window_tokens=100)


def test_a_zero_or_negative_refresh_is_refused():
    with pytest.raises(ValueError, match=">= 1"):
        _build(sparse_refresh_ticks=0)


def test_default_geometry_is_unchanged_when_unset():
    """The default path must not run the override at all, so an unflagged process
    is byte-identical to the pre-flag build."""
    assert sparse_index.resolve_window_tokens() is None
    assert sparse_engine.resolve_refresh_ticks() is None
    e = _build()
    try:
        assert sparse_index.WINDOW_TOKENS == 128 and sparse_index.WINDOW_PAGES == 8
        assert sparse_engine.WINDOW_PAGES == 8
        assert sparse_engine.SPARSE_REFRESH_TICKS == 8
    finally:
        e.shutdown()


def test_env_is_the_second_source_and_the_flag_wins(monkeypatch):
    monkeypatch.setenv("TILERL_SPARSE_WINDOW_TOKENS", "1024")
    monkeypatch.setenv("TILERL_SPARSE_REFRESH_TICKS", "32")
    assert sparse_index.resolve_window_tokens() == _WINDOW
    assert sparse_engine.resolve_refresh_ticks() == _REFRESH
    # An explicit value (the serve flag) beats the env, as with the draft window.
    assert sparse_index.resolve_window_tokens(2048) == 2048
    assert sparse_engine.resolve_refresh_ticks(16) == 16
    # Empty is unset, not zero: "" must not become a 0-token window.
    monkeypatch.setenv("TILERL_SPARSE_WINDOW_TOKENS", "")
    assert sparse_index.resolve_window_tokens() is None


def test_env_reaches_the_built_engine(monkeypatch):
    """The env route is a second entry point to the same geometry, so it gets the
    same reading the flag does rather than being trusted by construction."""
    monkeypatch.setenv("TILERL_SPARSE_WINDOW_TOKENS", str(_WINDOW))
    monkeypatch.setenv("TILERL_SPARSE_REFRESH_TICKS", str(_REFRESH))
    e = _build()
    try:
        assert sparse_index.WINDOW_PAGES == _WINDOW // 16
        assert sparse_engine.WINDOW_PAGES == _WINDOW // 16
        assert sparse_engine.SPARSE_REFRESH_TICKS == _REFRESH
    finally:
        e.shutdown()
