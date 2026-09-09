"""A torn append must not take a view down: load_all skips unparseable lines
and says how many (1 = a torn tail, many = a corrupted store)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchrec  # noqa: E402


def _write(monkeypatch, tmp_path, lines):
    store = tmp_path / "measurements.jsonl"
    store.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(benchrec, "STORE", store)


def test_torn_final_line_skipped_with_count(monkeypatch, tmp_path, capsys):
    good = json.dumps({"metric": "x", "value": 1})
    _write(monkeypatch, tmp_path, [good, good, '{"metric": "x", "val'])
    assert len(benchrec.load_all()) == 2
    assert "skipped 1" in capsys.readouterr().err


def test_many_skipped_lines_say_corrupted(monkeypatch, tmp_path, capsys):
    _write(monkeypatch, tmp_path, ["garbage"] * 40)
    assert benchrec.load_all() == []
    assert "skipped 40" in capsys.readouterr().err


def test_clean_store_is_silent(monkeypatch, tmp_path, capsys):
    _write(monkeypatch, tmp_path, [json.dumps({"metric": "x"})] * 3)
    assert len(benchrec.load_all()) == 3
    assert capsys.readouterr().err == ""
