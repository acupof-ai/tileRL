"""Tests for pod_harvest_bench: the reverse-direction mirror of merge-by-id.

The fetch step (pod connection) is not tested — harvest() takes the pod
content as a string, so tests construct jsonl directly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "pod_harvest_bench", ROOT / "scripts" / "pod_harvest_bench.py"
)
pod_harvest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pod_harvest)


def _row(rid: str, metric: str = "gsm8k_pct", value: float = 91.4) -> str:
    return json.dumps({"id": rid, "metric": metric, "value": value, "unit": "%"})


def test_harvest_appends_new_and_refuses_conflict(tmp_path, capsys):
    local = tmp_path / "measurements.jsonl"
    local.write_text(_row("aaa") + "\n" + _row("bbb", value=90.0) + "\n")
    # Pod: aaa identical, bbb same id different content, ccc new
    pod = _row("aaa") + "\n" + _row("bbb", value=95.0) + "\n" + _row("ccc") + "\n"

    n = pod_harvest.harvest(local, pod)
    assert n == 1

    lines = local.read_text().splitlines()
    assert len(lines) == 3
    ids = [json.loads(l)["id"] for l in lines]
    assert ids == ["aaa", "bbb", "ccc"]
    assert json.loads(lines[1])["value"] == 90.0  # local bbb unchanged

    out = capsys.readouterr()
    assert "CONFLICT id=bbb" in out.err
    assert "1 conflict(s) refused" in out.err
    assert "append id=ccc" in out.out
    assert "read 3 pod rows, 1 new, 1 conflicts" in out.out


def test_harvest_zero_rows_still_prints(tmp_path, capsys):
    local = tmp_path / "measurements.jsonl"
    local.write_text(_row("aaa") + "\n")

    n = pod_harvest.harvest(local, _row("aaa") + "\n")
    assert n == 0
    assert "read 1 pod rows, 0 new, 0 conflicts" in capsys.readouterr().out


def test_harvest_empty_local_store(tmp_path, capsys):
    local = tmp_path / "measurements.jsonl"
    # local file doesn't exist yet

    n = pod_harvest.harvest(local, _row("aaa") + "\n" + _row("bbb") + "\n")
    assert n == 2
    assert len(local.read_text().splitlines()) == 2
    assert "read 2 pod rows, 2 new, 0 conflicts" in capsys.readouterr().out


def test_harvest_empty_pod_read(tmp_path, capsys):
    """An empty pod read (truncated fetch, wrong path, empty file) must be
    distinguishable from 'pod has rows but nothing new'."""
    local = tmp_path / "measurements.jsonl"
    local.write_text(_row("aaa") + "\n")

    assert pod_harvest.harvest(local, "") == 0
    assert "read 0 pod rows" in capsys.readouterr().out
