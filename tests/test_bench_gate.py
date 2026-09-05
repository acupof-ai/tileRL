"""The baseline gate's two silent failure modes: a beat that writes itself, and a
lower-is-better number fed into a higher-is-better row."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_harness as bh  # noqa: E402


def _gate(monkeypatch, tmp_path, rows):
    monkeypatch.setattr(bh, "_BASELINE", tmp_path / "baseline.json")
    (tmp_path / "baseline.json").write_text(json.dumps(rows))
    return bh.Gate("sm90")


def test_a_beat_is_proposed_not_written(monkeypatch, tmp_path):
    """`bench-baseline.json` is SOTA-only and is what the 0.97x gate compares against,
    so a run that promotes its own result has nothing left to regress against: the
    next slow run is measured against the fast one it already replaced."""
    g = _gate(monkeypatch, tmp_path, {"train/x/sm90": {"tok_s": 10.0, "commit": "a", "date": "d"}})
    assert g.check("train", "x", 20.0) == "BEAT"

    cand = tmp_path / "runs" / "r1" / "baseline-candidate.json"
    g.finish(cand)
    assert json.loads((tmp_path / "baseline.json").read_text())["train/x/sm90"]["tok_s"] == 10.0, (
        "the beat was written into the baseline; it must only be proposed")
    assert json.loads(cand.read_text())["train/x/sm90"]["tok_s"] == 20.0


def test_seconds_per_step_would_invert_the_gate(monkeypatch, tmp_path):
    """Every row is `tok_s`, higher-is-better, and all three comparisons are `>`.
    A training row therefore has to be steps/SECOND. Fed seconds/step, a run twice
    as SLOW reads as a beat -- and the table prints it as a win, which is why this
    is a test and not a comment."""
    g = _gate(monkeypatch, tmp_path, {"train/x/sm90": {"tok_s": 30.0, "commit": "a", "date": "d"}})
    slow_secs_per_step, fast_secs_per_step = 60.0, 30.0

    assert g.check("train", "x", slow_secs_per_step) == "BEAT", (
        "documents the trap: raw seconds make the slower run look like the record")
    assert g.check("train", "x", 1.0 / slow_secs_per_step) == "FAIL", (
        "as steps/s the slower run must fail")
    assert bh.Gate("sm90").check("train", "y", 1.0 / fast_secs_per_step) == "SEED"


def test_a_training_run_never_edits_the_tracked_baseline(tmp_path, monkeypatch):
    """`_timing_snapshot` runs at the end of EVERY train, the tiny smoke recipe pytest
    runs included -- so if it seeds, the suite writes rows into the tracked SOTA json.
    It did: five `train-run/tiny-*` rows landed there the first time this ran.

    The assertion is on the REAL `_BASELINE`, not a monkeypatched one: `_timing_snapshot`
    loads bench_harness through importlib, so a patched module attribute never reaches
    the copy it executes -- the first version of this test patched `bh._BASELINE`, passed
    with the guard removed, and proved nothing.
    """
    from tilerl import cli

    before = bh._BASELINE.read_text()
    monkeypatch.setattr("tilerl.ledger.runs_root", lambda: str(tmp_path))
    (tmp_path / "r1").mkdir()
    cli._timing_snapshot({
        "id": "r1", "inputs": {"model": "tiny", "algo": "grpo", "group": 6,
                               "max_new_tokens": 8},
        "metrics": {"secs_per_step_median": 0.25}})
    assert bh._BASELINE.read_text() == before, (
        "a training run edited the tracked SOTA baseline")


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
