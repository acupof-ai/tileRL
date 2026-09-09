"""The training run's gsm8k_pct emission: one arm = one validated store row."""

from __future__ import annotations

import json

import pytest

from tilerl import cli


class _Backend:
    arch = "sm90"

    class device:
        type = "cuda"
        name = "H20"


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    benchrec = cli._benchrec()
    monkeypatch.setattr(benchrec, "STORE", tmp_path / "measurements.jsonl")
    return benchrec


def test_emit_gsm8k_record(tmp_store, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    cli._emit_gsm8k_record(190, 200, 0, _Backend())
    cli._emit_gsm8k_record(196, 200, 100, _Backend())
    rows = [json.loads(l) for l in tmp_store.STORE.read_text().splitlines()]
    assert len(rows) == 2
    # steps is the population: before and after must not collapse into one row
    assert len(tmp_store.current(rows)) == 2
    by_steps = {r["shape"]["steps"]: r for r in rows}
    assert by_steps[0]["value"] == 95.0 and by_steps[100]["value"] == 98.0
    for r in rows:
        assert r["metric"] == "gsm8k_pct" and r["unit"] == "%"
        assert r["target"] == "sm90" and r["device"] == {"name": "H20", "card": 6}
        assert r["floor"]["kind"] == "measured-best"
        assert r["n"] == 200 and r["spread"] > 0
        assert len(r["commit"]) == 40 and isinstance(r["dirty"], bool)
