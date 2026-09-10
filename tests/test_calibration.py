"""Calibration ledger + roofline bound: CPU-only arithmetic and lookup gates.

The measurement itself (D2D copy / big GEMM, CUDA events) is cuda-only and is refused
off a card. What is testable here, and what the renderer depends on, is pure: the bound
is ``max(bytes/bw, flops/peak)``; floors key on the EXACT device name; and a missing or
half calibration renders pending-remote rather than a datasheet number.
"""

from __future__ import annotations

import json

import pytest

from tilerl import calibration as cal

H20 = "NVIDIA H20"
V100 = "Tesla V100-SXM2-16GB"


def _row(metric, value, unit, name, card=0):
    return {"metric": metric, "value": value, "unit": unit, "id": f"{metric}-{card}",
            "device": {"name": name, "card": card}}


def _store(tmp_path, rows):
    p = tmp_path / "measurements.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_bound_seconds_is_exact_max_of_byte_and_flop_time():
    # 100 GB at 1000 GB/s = 0.1 s; 1e13 flop at 100 TFLOP/s = 0.1 s — equal arms.
    assert cal.bound_seconds(100 * 10**9, 10**13, 1000.0, 100.0) == pytest.approx(0.1)
    # byte-bound: 200 GB -> 0.2 s dominates the 0.1 s flop time.
    assert cal.bound_seconds(200 * 10**9, 10**13, 1000.0, 100.0) == pytest.approx(0.2)
    # flop-bound: 4e13 flop -> 0.4 s dominates a 0.1 s byte time.
    assert cal.bound_seconds(100 * 10**9, 4 * 10**13, 1000.0, 100.0) == pytest.approx(0.4)


def test_latest_floor_keys_on_exact_device_name(tmp_path):
    rows = [_row(cal.BW_METRIC, 4000.0, "GB/s", H20),
            _row(cal.BW_METRIC, 900.0, "GB/s", V100, card=1)]
    p = _store(tmp_path, rows)
    loaded = cal.load_rows(p)
    assert cal.latest_floor(loaded, cal.BW_METRIC, H20)["value"] == 4000.0
    # A V100 name does not satisfy an H20 lookup — no prefix/case fallback.
    assert cal.latest_floor(loaded, cal.BW_METRIC, "nvidia h20") is None


def test_newest_row_wins_and_superseded_skipped(tmp_path):
    old = _row(cal.BW_METRIC, 3900.0, "GB/s", H20)
    old["id"] = "old"
    new = _row(cal.BW_METRIC, 4100.0, "GB/s", H20)
    new["id"] = "new"
    new["supersedes"] = "old"
    p = _store(tmp_path, [old, new])
    cur = cal.latest_floor(cal.load_rows(p), cal.BW_METRIC, H20)
    assert cur["id"] == "new" and cur["value"] == 4100.0


def test_calibration_is_none_when_a_floor_is_missing(tmp_path):
    # bandwidth present but peak absent -> half calibration must not divide anything.
    p = _store(tmp_path, [_row(cal.BW_METRIC, 4000.0, "GB/s", H20)])
    assert cal.calibration(cal.load_rows(p), H20) is None
    # and an empty store (pending-remote) likewise, not a datasheet default.
    assert cal.calibration(cal.load_rows(_store(tmp_path, [])), H20) is None


def test_calibration_returns_both_floors(tmp_path):
    p = _store(tmp_path, [_row(cal.BW_METRIC, 4000.0, "GB/s", H20),
                          _row(cal.PEAK_METRIC, 989.0, "TFLOP/s", H20)])
    got = cal.calibration(cal.load_rows(p), H20)
    assert got == {"bw_gbs": 4000.0, "peak_tflops": 989.0,
                   "bw_commit": None, "peak_commit": None}


def test_calibrate_refuses_off_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "cuda", type("C", (), {"is_available": lambda self: False})())
    # The CLI guard is the user-facing refusal; the measurement module must also have no
    # cuda device to time. Check the CLI exits with the card-bound command printed.
    import argparse

    from tilerl import cli

    args = argparse.Namespace(card=None)
    with pytest.raises(SystemExit, match="--card"):
        cli.cmd_bench_calibrate(args)
    args = argparse.Namespace(card=0)
    monkeypatch.setattr(torch, "cuda",
                        type("C", (), {"is_available": lambda self: False})())
    with pytest.raises(SystemExit, match="cuda-only"):
        cli.cmd_bench_calibrate(args)


def test_append_rows_roundtrip(tmp_path):
    p = tmp_path / "sub" / "measurements.jsonl"
    rows = [_row(cal.BW_METRIC, 4000.0, "GB/s", H20)]
    cal.append_rows(rows, p)
    got = cal.load_rows(p)
    assert len(got) == 1 and got[0]["metric"] == cal.BW_METRIC and got[0]["id"]


def test_kernels_table_pending_when_no_calibration(tmp_path, monkeypatch, capsys):
    """With no ledger row the roofline columns render pending-remote, not a datasheet
    number, even though bytes/flops always print."""
    monkeypatch.setenv("TILERL_BENCH_STORE", str(tmp_path / "none.jsonl"))
    import argparse

    from tilerl import cli

    args = argparse.Namespace(model="tiny", batches=None, context=4096, prefill=0,
                              device_name="tiny-cpu-no-cal")
    cli.cmd_bench_kernels(args)
    out = capsys.readouterr().out
    assert "pending-remote" in out
    assert "TICK TOTAL" in out
