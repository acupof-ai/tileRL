"""The training run's eval-arm emission: one arm = two validated store rows
(gsm8k_pct and rollout_tokens; tokens/correct is a view, never stored)."""

from __future__ import annotations

import json

import pytest
import torch

from tilerl import cli


class _Backend:
    arch = "sm90"

    class device:
        type = "cuda"
        name = "H20"


class _CPUBackend:
    arch = "cpu"

    class device:
        type = "cpu"


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    benchrec = cli._benchrec()
    monkeypatch.setattr(benchrec, "STORE", tmp_path / "measurements.jsonl")
    return benchrec


def test_emit_eval_records(tmp_store, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d: "NVIDIA H20")
    lens = list(range(200))  # mean 99.5, nonzero spread
    cli._emit_eval_records(190, 200, sum(lens), lens, 0, _Backend())
    cli._emit_eval_records(196, 200, 20000, [100] * 200, 100, _Backend())
    # A rerun of the same arm: rollout_tokens has no monotonic direction, so its
    # floor is the measurement itself (reference), never the population's best.
    cli._emit_eval_records(190, 200, 22000, [110] * 200, 100, _Backend())
    rows = [json.loads(l) for l in tmp_store.STORE.read_text().splitlines()]
    assert len(rows) == 6
    by = {(r["metric"], r["shape"]["steps"], r["value"]): r for r in rows}
    assert by[("gsm8k_pct", 0, 95.0)]["spread"] > 0
    assert by[("gsm8k_pct", 100, 98.0)]["n"] == 200
    assert by[("rollout_tokens", 0, 99.5)]["spread"] > 0
    assert by[("rollout_tokens", 100, 100.0)]["spread"] == 0.0
    # steps is the population: before and after must not collapse into one row
    assert len({r["shape"]["steps"] for r in rows}) == 2
    worse = by[("rollout_tokens", 100, 110.0)]
    assert worse["floor"]["value"] == 110.0  # the measurement itself, not the prior best
    for r in rows:
        assert r["target"] == "sm90" and r["device"] == {"name": "NVIDIA H20", "card": 6}
        assert r["floor"]["kind"] == (
            "reference" if r["metric"] == "rollout_tokens" else "measured-best")
        assert r["n"] == 200 and r["spread"] >= 0
        assert len(r["commit"]) == 40 and isinstance(r["dirty"], bool)


def test_emit_eval_records_cpu(tmp_store, monkeypatch):
    """No CUDA: card is null, name falls back to the backend's target 'cpu'.

    The shape is record_common's (card always present, null when absent) -- the
    store's existing cpu rows are all {"card": null, "name": "cpu"}, and the
    grouping key reads .get("card"), so missing-key and null are one population.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cli._emit_eval_records(190, 200, 20000, [100] * 200, 0, _CPUBackend())
    rows = [json.loads(l) for l in tmp_store.STORE.read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert r["device"] == {"name": "cpu", "card": None}
