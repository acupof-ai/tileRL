"""--max-batched-tokens must reach the engine: a flag that parses but is not
threaded into _build_engine is the --deterministic failure mode (flag in, help
in, no effect)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tilerl import cli


@pytest.fixture
def fake_cfg():
    return SimpleNamespace(max_position_embeddings=4096, name="fake")


def _captured_kw(fake_cfg, **kwargs):
    captured = {}

    def fake_build_engine(cfg, model, backend, **kw):
        captured.update(kw)
        return object()

    with patch("tilerl.engine.build_engine", fake_build_engine):
        cli._build_engine(fake_cfg, None, None, **kwargs)
    return captured


def test_max_batched_tokens_threaded(fake_cfg):
    kw = _captured_kw(fake_cfg, max_batched_tokens=2048, sparse_k=0)
    assert kw["max_num_batched_tokens"] == 2048


def test_default_leaves_engine_default(fake_cfg):
    # 0 means "do not pass it": StepLimits' own 512 stays the default.
    kw = _captured_kw(fake_cfg, sparse_k=0)
    assert "max_num_batched_tokens" not in kw


def test_parser_carries_flag():
    args = cli._build_parser().parse_args(["serve", "--max-batched-tokens", "2048"])
    assert args.max_batched_tokens == 2048
