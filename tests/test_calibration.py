"""Calibration ledger + roofline bound: CPU-only arithmetic and lookup gates.

The measurement itself (D2D copy / big GEMM, CUDA events) is cuda-only and is refused
off a card. What is testable here, and what the renderer depends on, is pure: the bound
is ``max(bytes/bw, flops/peak)``; floors key on the EXACT device name; and a missing or
half calibration renders pending-remote rather than a datasheet number.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from tilerl import calibration as cal

H20 = "NVIDIA H20"
V100 = "Tesla V100-SXM2-16GB"
#: a commit that exists in this repo so benchrec's commit-exists validator accepts it
_HEAD_SHA = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, check=True).stdout.strip()


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


def test_latest_floor_rejects_a_near_prefix_device_name(tmp_path):
    """Case-insensitivity and an exact-name typo are already covered; a PREFIX match
    is a third, distinct leak: an H200 row sharing the 'NVIDIA H20' prefix would key
    the H20 roofline to the wrong card's floor. Mutation caught: '==' replaced with a
    4-char-prefix startswith left the exact-name gate green."""
    h200 = "NVIDIA H200"
    p = _store(tmp_path, [_row(cal.BW_METRIC, 5000.0, "GB/s", h200, card=2)])
    got = cal.latest_floor(cal.load_rows(p), cal.BW_METRIC, H20)
    assert got is None, f"a prefix-near card ({h200}) must not satisfy the {H20} floor"



def test_latest_floor_keys_on_the_physical_uuid_when_uuid_rows_exist(tmp_path):
    """Two same-name rows from DIFFERENT physical cards with different values:
    without uuid the name-keyed floor is whichever appended last (the H20 card-5
    die measured 8% low). With uuid the lookup returns that card's own newest row;
    when no uuid row matches (or none passed) it falls back to the name pool."""
    ua, ub = "aaaaaaaa-0000-0000-0000-000000000000", "bbbbbbbb-0000-0000-0000-000000000000"
    ra = _row(cal.PEAK_METRIC, 136.5, "TFLOP/s", H20, card=0)
    ra["device"]["uuid"] = ua
    rb = _row(cal.PEAK_METRIC, 125.8, "TFLOP/s", H20, card=0)
    rb["device"]["uuid"] = ub
    rows = cal.load_rows(_store(tmp_path, [ra, rb]))
    assert cal.latest_floor(rows, cal.PEAK_METRIC, H20)["value"] == 125.8
    assert cal.latest_floor(rows, cal.PEAK_METRIC, H20, ua)["value"] == 136.5
    assert cal.latest_floor(rows, cal.PEAK_METRIC, H20, ub)["value"] == 125.8
    assert cal.latest_floor(rows, cal.PEAK_METRIC, H20,
                            "cccccccc-0000-0000-0000-000000000000")["value"] == 125.8
    assert cal.latest_floor(rows, cal.PEAK_METRIC, V100, ua) is None
    bw = _row(cal.BW_METRIC, 3312.0, "GB/s", H20, card=0)
    bw["device"]["uuid"] = ua
    p2 = tmp_path / "u" / "measurements.jsonl"
    p2.parent.mkdir()
    p2.write_text("".join(json.dumps(r) + "\n" for r in (ra, rb, bw)))
    got = cal.calibration(cal.load_rows(p2), H20, ua)
    assert got == {"bw_gbs": 3312.0, "peak_tflops": 136.5,
                   "peak_metric": cal.PEAK_METRIC,
                   "fp8_peak_tflops": None, "pcie_gbs": None}

def test_newest_row_wins_and_superseded_skipped(tmp_path):
    old = _row(cal.BW_METRIC, 3900.0, "GB/s", H20)
    old["id"] = "old"
    new = _row(cal.BW_METRIC, 4100.0, "GB/s", H20)
    new["id"] = "new"
    new["supersedes"] = "old"
    p = _store(tmp_path, [old, new])
    cur = cal.latest_floor(cal.load_rows(p), cal.BW_METRIC, H20)
    assert cur["id"] == "new" and cur["value"] == 4100.0


def test_a_superseded_row_is_skipped_even_when_appended_last(tmp_path):
    """Append order is not measurement time across harvests (pod_harvest merges store
    files). When the superseded row lands LAST in the file, newest-by-position returns
    the stale value; only the supersedes id set excludes it. Mutation caught with the
    rows in natural order nothing fails (newest wins anyway): emptying the supersedes
    set left newest_row_wins green. Here the stale row is appended after its replacer."""
    new = _row(cal.BW_METRIC, 4100.0, "GB/s", H20)
    new["id"] = "new"
    new["supersedes"] = "old"
    old = _row(cal.BW_METRIC, 3900.0, "GB/s", H20)
    old["id"] = "old"
    # replacer appended first, the retired row re-harvested into the file afterwards
    p = _store(tmp_path, [new, old])
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
    # bf16 pair present; fp8 peak and pcie absent -> None (both render pending)
    assert got == {"bw_gbs": 4000.0, "peak_tflops": 989.0,
                   "peak_metric": cal.PEAK_METRIC,
                   "fp8_peak_tflops": None, "pcie_gbs": None}


def test_calibration_falls_back_to_f16_peak_on_sm70(tmp_path):
    """A pre-Ampere card has an f16 peak row but no bf16 row: calibration falls
    back to it and names it in peak_metric."""
    V100 = "Tesla V100-SXM2-32GB"
    p = _store(tmp_path, [_row(cal.BW_METRIC, 778.0, "GB/s", V100),
                          _row(cal.F16_PEAK_METRIC, 88.8, "TFLOP/s", V100)])
    assert cal.calibration(cal.load_rows(p), V100) == {
        "bw_gbs": 778.0, "peak_tflops": 88.8, "peak_metric": cal.F16_PEAK_METRIC,
        "fp8_peak_tflops": None, "pcie_gbs": None}


def test_calibration_returns_fp8_peak_when_present(tmp_path):
    p = _store(tmp_path, [_row(cal.BW_METRIC, 4000.0, "GB/s", H20),
                          _row(cal.PEAK_METRIC, 989.0, "TFLOP/s", H20),
                          _row(cal.FP8_PEAK_METRIC, 1970.0, "TFLOP/s", H20)])
    assert cal.calibration(cal.load_rows(p), H20) == {
        "bw_gbs": 4000.0, "peak_tflops": 989.0, "peak_metric": cal.PEAK_METRIC,
        "fp8_peak_tflops": 1970.0, "pcie_gbs": None}


def test_row_peak_keys_by_face(tmp_path):
    """The ceiling is the MMA dtype's peak at the launch M: fp8-weight rows on e4m3
    WGMMA (M>=9) use the fp8 ceiling, on bf16 decode (M<=8) the bf16 ceiling; an
    nvfp4 prefill GEMM is w4a8 and takes fp8 too; missing fp8 floor -> None."""
    from tilerl import precision as P

    bf = {"bw_gbs": 4000.0, "peak_tflops": 100.0, "fp8_peak_tflops": 200.0}

    def row(face, name="q_proj"):
        return {"name": name, "face": face}

    assert cal.row_peak_tflops(bf, row(P.fp8_dev), 1, 4096) == 200.0
    assert cal.row_peak_tflops(bf, row(P.fp8_block_dev), 1, 4096) == 200.0
    assert cal.row_peak_tflops(bf, row(P.fp8_dev), 1, 1) == 100.0
    assert cal.row_peak_tflops(bf, row(P.fp8_dev), 8, 1) == 100.0
    assert cal.row_peak_tflops(bf, row(P.nvfp4_dev), 1, 4096) == 200.0
    assert cal.row_peak_tflops(bf, row(P.nvfp4_dev_b32), 1, 4096) == 200.0
    assert cal.row_peak_tflops(bf, row(P.nvfp4_dev), 1, 1) == 100.0
    # lm_head prices one vector per row: b=1 decode is M=1 -> bf16
    assert cal.row_peak_tflops(bf, row(P.fp8_dev, "lm_head"), 1, 4096) == 100.0
    assert cal.row_peak_tflops(bf, row(P.bf16), 1, 4096) == 100.0
    missing = {"bw_gbs": 4000.0, "peak_tflops": 100.0, "fp8_peak_tflops": None}
    assert cal.row_peak_tflops(missing, row(P.fp8_dev), 1, 4096) is None
    assert cal.row_peak_tflops(missing, row(P.nvfp4_dev), 1, 4096) is None


def test_calibration_returns_pcie_when_present(tmp_path):
    p = _store(tmp_path, [_row(cal.BW_METRIC, 4000.0, "GB/s", H20),
                          _row(cal.PEAK_METRIC, 989.0, "TFLOP/s", H20),
                          _row(cal.PCIE_METRIC, 50.0, "GB/s", H20)])
    assert cal.calibration(cal.load_rows(p), H20) == {
        "bw_gbs": 4000.0, "peak_tflops": 989.0, "peak_metric": cal.PEAK_METRIC,
        "fp8_peak_tflops": None, "pcie_gbs": 50.0}


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


def _valid_row(metric, value, unit, name=H20, card=0):
    """A row satisfying scripts/benchrec's full REQUIRED schema."""
    return {
        "metric": metric, "value": value, "unit": unit, "target": "sm90",
        "build": "eager", "model": "device", "shape": {"card": card},
        "warm": {"state": "warm", "compiles": None}, "n": 1, "spread": 0,
        "device": {"name": name, "card": card},
        "commit": _HEAD_SHA, "dirty": False,
        "cmd": f"tilerl bench --calibrate --card {card}",
        "floor": {"value": value, "unit": unit, "kind": "measured-best",
                  "derivation": "test row; floor = this measurement"},
    }


def test_append_rows_goes_through_benchrec_validator(tmp_path):
    """calibration.append_rows must route through scripts/benchrec.validate — a malformed
    row is rejected and never reaches the file (no second, unvalidated writer)."""
    p = tmp_path / "measurements.jsonl"
    ids = cal.append_rows([_valid_row(cal.BW_METRIC, 4000.0, "GB/s")], p)
    assert len(ids) == 1 and p.read_text().count("\n") == 1
    # a row missing required fields / with a bogus unit is rejected by benchrec, not us.
    bad = _valid_row(cal.BW_METRIC, 4000.0, "GB/s")
    del bad["commit"]
    with pytest.raises(Exception):  # benchrec.ValueError via append
        cal.append_rows([bad], p)
    assert p.read_text().count("\n") == 1, "rejected row must not reach the store"
    bad_unit = _valid_row(cal.BW_METRIC, 4000.0, "TB/s")
    with pytest.raises(Exception):
        cal.append_rows([bad_unit], p)


def test_resolve_row_kernel_is_the_declared_kernel_not_linear():
    """The timing seam resolves the kernel by the row's weight FACE: nvfp4 faces ->
    linear_fp4, fp8 faces -> linear_fp8, never the bf16 backend.linear — timing the
    latter would divide packed bytes by a 2 B/weight kernel. A fake backend with
    sentinels proves both resolutions and that fused rows, bf16 rows and unknown faces
    resolve to None (pending), not a surrogate. The same row NAME split across faces
    (the 27B's 233 fp8 linears among nvfp4 tiers) must resolve per face."""
    from tilerl import precision as P

    fp4_sentinel = lambda *a, **k: None  # noqa: E731
    fp8_sentinel = lambda *a, **k: None  # noqa: E731

    class FakeBackend:
        linear_fp4 = staticmethod(fp4_sentinel)
        linear_fp8 = staticmethod(fp8_sentinel)

        def linear(self, *a, **k):  # the wrong-face kernel must never be chosen
            raise AssertionError("bf16 linear must never time a quantized row")

    fb = FakeBackend()
    # every nvfp4 face the face map emits resolves the fp4 kernel
    for face in (P.nvfp4, P.nvfp4_dev, P.nvfp4_dev_b32):
        assert cal.resolve_row_kernel(fb, {"name": "down_proj", "face": face}) is fp4_sentinel
    # both fp8 device faces resolve the fp8 kernel — the old name->linear_fp4 map
    # sent every fp8 row to the wrong kernel
    for face in (P.fp8_block_dev, P.fp8_dev):
        assert cal.resolve_row_kernel(fb, {"name": "down_proj", "face": face}) is fp8_sentinel
    # same row name, opposite face, opposite kernel: resolution is per face, not per name
    assert cal.resolve_row_kernel(fb, {"name": "q_proj", "face": P.nvfp4_dev}) is fp4_sentinel
    assert cal.resolve_row_kernel(fb, {"name": "q_proj", "face": P.fp8_dev}) is fp8_sentinel
    # bf16 / faceless / fused rows have no declared timing kernel -> pending
    assert cal.resolve_row_kernel(fb, {"name": "down_proj", "face": P.bf16}) is None
    assert cal.resolve_row_kernel(fb, {"name": "down_proj"}) is None
    assert cal.resolve_row_kernel(fb, {"name": "paged_attention_decode", "face": P.f32}) is None
    assert cal.resolve_row_kernel(fb, {"name": "gdn_decode_fused"}) is None


def test_pack_for_matches_the_resolved_face():
    """The weight tensors _pack_for builds must match the kernel the face resolves to:
    fp4 -> (wq nibbles, scale, oscale) for linear_fp4; fp8 -> (w8, wscale) for
    linear_fp8. A wrong branch feeds linear_fp8 packed nibbles or linear_fp4 fp8
    tensors — this gate pins the pairing without a card."""
    import torch

    from tilerl import precision as P

    w = torch.randn(16, 32, dtype=torch.bfloat16)
    (wq, scale4), kw4 = cal._pack_for(P.nvfp4_dev, w)
    assert wq.dtype == torch.uint8 and tuple(wq.shape) == (16, 16)
    assert set(kw4) == {"oscale"} and kw4["oscale"].shape == (16,)
    # quant_fp8 pads to its 128x128 block grid
    (w8, _wscale), kw8 = cal._pack_for(P.fp8_block_dev, torch.randn(128, 128, dtype=torch.bfloat16))
    assert w8.dtype == torch.float8_e4m3fn and w8.shape == (128, 128)
    assert kw8 == {}  # linear_fp8 synthesizes its ones grid/row


def test_time_row_ms_pending_off_cuda_or_unknown_row(monkeypatch):
    """Off cuda every row is pending; on cuda a row with no declared kernel still is."""
    import torch

    monkeypatch.setattr(torch, "cuda", type("C", (), {"is_available": lambda self: False})())
    assert cal.time_row_ms(
        {"name": "down_proj", "_spec": (4, 4), "face": None}, object(), 1, 8) is None


def test_time_row_ms_identity_assert_survives_bound_methods(monkeypatch):
    """time_row_ms asserts the kernel it resolves twice is the same. A real backend's
    linear_fp4 is a BOUND method, and two getattr calls hand back distinct wrapper
    objects, so a plain `is` always failed — the first card run of bench --kernels
    crashed with AssertionError. Drive time_row_ms with faked cuda and a stubbed event
    timer so the identity assertion runs without a GPU; the bound kernel returns None.
    Mutant: replace the __func__ comparison in time_row_ms with `fn is
    resolve_row_kernel(...)` — this gate goes red."""
    import torch

    monkeypatch.setattr(torch, "cuda", type("C", (), {"is_available": lambda self: True})())
    monkeypatch.setattr(cal, "_event_seconds", lambda fn, iters=20: 0.001)

    from tilerl import precision as P

    class BoundBackend:
        device = torch.device("cpu")

        def linear_fp4(self, x, wq, scale, oscale=None):
            return None

    row = {"name": "down_proj", "_spec": (32, 32), "face": P.nvfp4_dev}  # inn 32 packs
    assert cal.time_row_ms(row, BoundBackend(), 1, 1) == 1.0


def test_time_row_ms_shape_matches_priced_flops_decode_and_prefill(monkeypatch):
    """The timed GEMM is the priced launch: a linear row's per-launch flops are
    2*M*N*K, so the row's declared flops pin M for both ticks (b*s) and prefill
    (b=1,s=S); lm_head prices b vectors. Mutant: row_launch_m returns 1 for every
    row — the prefill/decode rows here go red (the 5,967% table class)."""
    import torch

    monkeypatch.setattr(torch, "cuda", type("C", (), {"is_available": lambda self: True})())
    monkeypatch.setattr(cal, "_event_seconds", lambda fn, iters=20: 0.001)

    from tilerl import precision as P

    class Backend:
        device = torch.device("cpu")

        def linear_fp8(self, x, wq, wscale):
            return None

    n, k = 32, 16
    for name, b, s, m in (("q_proj", 1, 8, 8), ("q_proj", 1, 4096, 4096),
                          ("q_proj", 8, 1, 8), ("lm_head", 8, 1, 8)):
        row = {"name": name, "_spec": (n, k), "face": P.fp8_dev,
               "flops": 2 * m * n * k}
        assert cal.time_row_ms(row, Backend(), b, s) == 1.0

    bad = {"name": "q_proj", "_spec": (n, k), "face": P.fp8_dev,
           "flops": 2 * 4096 * n * k}  # prices a 4096-token prefill launch
    with pytest.raises(AssertionError):
        cal.time_row_ms(bad, Backend(), 1, 1)  # timed at m=1 — the m=1 mutant


def test_kernels_checkpoint_guard_refuses_model_mismatch(tmp_path, monkeypatch):
    """bench --kernels --checkpoint is the command that printed the 5,967% table
    (27B checkpoint on tiny cfg); the shared guard must refuse it before any
    safetensors header is read."""
    import json

    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 48, "hidden_size": 5120, "num_attention_heads": 40,
        "num_key_value_heads": 4, "head_dim": 256}))
    import argparse

    from tilerl import cli

    args = argparse.Namespace(model="tiny", batches=None, context=4096, prefill=0,
                              checkpoint=str(tmp_path), device_name="x",
                              kv_fp8=False)
    with pytest.raises(SystemExit, match="not a tiny checkpoint"):
        cli.cmd_bench_kernels(args)


def test_kernels_table_pending_when_no_calibration(tmp_path, monkeypatch, capsys):
    """With no ledger row the roofline columns render pending-remote, not a datasheet
    number, even though bytes/flops always print."""
    monkeypatch.setenv("TILERL_BENCH_STORE", str(tmp_path / "none.jsonl"))
    import argparse

    from tilerl import cli

    args = argparse.Namespace(model="tiny", batches=None, context=4096, prefill=0,
                              checkpoint=None, device_name="tiny-cpu-no-cal")
    cli.cmd_bench_kernels(args)
    out = capsys.readouterr().out
    assert "pending-remote" in out
    assert "TICK TOTAL" in out


def test_bench_kernels_decode_times_one_token_launch(monkeypatch):
    """A decode tick prices b*s against the pooled context but the timed GEMM is a
    one-token launch (M=b for linears): passing s=4096 into the timer timed a fat
    prefill GEMM and printed a fictional ~10x decode tick (pod 2026-09-11).
    Mutant: render(..., b, args.context) on the decode call -> timed_s=4096."""
    import torch

    seen: list[tuple[str, int, int]] = []

    def fake_time(row, backend, b, s):
        seen.append((row["name"], b, s))
        return None  # suppress ms columns; M only needs to be observed

    monkeypatch.setattr(cal, "time_row_ms", fake_time)
    monkeypatch.setattr(cal, "load_rows", lambda: [])
    monkeypatch.setattr(
        cal, "calibration",
        lambda rows, name, uuid=None: {"bw_gbs": 4000.0, "peak_tflops": 100.0,
                            "fp8_peak_tflops": 200.0})
    class _Props:
        uuid = "stub-uuid"

    fake_cuda = type("C", (), {
        "is_available": lambda self: True,
        "get_device_name": lambda self, *a: "stub",
        "get_device_properties": lambda self, *a: _Props(),
        "_is_compiled": staticmethod(lambda: False)})()
    monkeypatch.setattr(torch, "cuda", fake_cuda)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    import tilerl_kernels.backend as kb
    monkeypatch.setattr(kb, "get_backend", lambda: object())
    import argparse

    from tilerl import cli

    args = argparse.Namespace(model="tiny", batches="1", context=512, prefill=0,
                              checkpoint=None, device_name=None)
    cli.cmd_bench_kernels(args)
    assert seen and all(s == 1 for _, _, s in seen), seen

    seen.clear()
    args2 = argparse.Namespace(model="tiny", batches=None, context=512, prefill=256,
                               checkpoint=None, device_name=None)
    cli.cmd_bench_kernels(args2)
    assert seen and all(s == 256 for _, _, s in seen), seen



def _cal_row(rid, metric, value, name, card, date, supersedes=None):
    r = {"metric": metric, "value": value, "unit": "GB/s" if "gbs" in metric else "TFLOP/s",
         "id": rid, "commit": _HEAD_SHA, "date": date,
         "device": {"name": name, "card": card}}
    if supersedes:
        r["supersedes"] = supersedes
    return r


def _resident_row(rid, name, card, peak, static, transient, date, supersedes=None):
    r = {"metric": cal.RESIDENT_METRIC, "value": peak, "unit": "bytes", "id": rid,
         "commit": _HEAD_SHA, "date": date, "device": {"name": name, "card": card},
         "shape": {"card": card, "static": static, "transient": transient}}
    if supersedes:
        r["supersedes"] = supersedes
    return r


def test_device_section_picks_newest_pair_and_residency_per_device(tmp_path):
    """Two devices, a superseded calibration row and a superseded residency row: each
    section reports the newest non-superseded pair keyed on the exact device name, and a
    device with no residency renders that sub-field None (pending, never a zero)."""
    rows = [
        # H20: old bw superseded by a newer row; peak present.
        _cal_row("h20-bw-old", cal.BW_METRIC, 3900.0, H20, 6, "2026-09-01T00:00Z"),
        _cal_row("h20-bw-new", cal.BW_METRIC, 4000.0, H20, 6, "2026-09-10T00:00Z",
                 supersedes="h20-bw-old"),
        _cal_row("h20-peak", cal.PEAK_METRIC, 148.0, H20, 6, "2026-09-10T00:00Z"),
        # H20 residency: old superseded.
        _resident_row("h20-res-old", H20, 6, 30_000, 29_000, 1_000, "2026-09-02T00:00Z"),
        _resident_row("h20-res-new", H20, 6, 32_000, 30_500, 1_500, "2026-09-11T00:00Z",
                      supersedes="h20-res-old"),
        # V100: calibrated but residency never recorded.
        _cal_row("v100-bw", cal.BW_METRIC, 900.0, V100, 1, "2026-09-03T00:00Z"),
        _cal_row("v100-peak", cal.PEAK_METRIC, 125.0, V100, 1, "2026-09-03T00:00Z"),
        # A device that appears ONLY in an unrelated bench metric: it must not create an
        # all-pending section (the old enumerate-every-name code rendered one).
        _cal_row("cpu-decode", "decode_tok_s", 94.0, "tiny-cpu", 0,
                 "2026-09-04T00:00Z"),
    ]
    p = _store(tmp_path, rows)
    loaded = cal.load_rows(p)
    sections = cal.device_sections(loaded)
    assert [s["device"] for s in sections] == [H20, V100]
    by = {s["device"]: s for s in sections}
    h20 = by[H20]
    assert h20["hbm_bw_gbs"]["value"] == 4000.0  # superseded 3900 skipped
    assert h20["hbm_bw_gbs"]["date"] == "2026-09-10T00:00Z"
    assert h20["bf16_peak_tflops"]["value"] == 148.0
    assert h20["residency"]["peak"] == 32_000   # superseded residency skipped
    assert h20["residency"]["static"] + h20["residency"]["transient"] == 32_000
    # the V100 has floors but no residency -> pending None, shape stays complete.
    v100 = by[V100]
    assert v100["hbm_bw_gbs"]["value"] == 900.0
    assert v100["residency"] is None
    # JSON shape the CLI pins.
    assert set(h20) == {"device", "hbm_bw_gbs", "bf16_peak_tflops", "f16_peak_tflops",
                        "fp8_peak_tflops", "pcie_h2d_gbs", "residency"}
    assert h20["fp8_peak_tflops"] is None  # no fp8 row recorded in this fixture
    assert h20["pcie_h2d_gbs"] is None
    assert h20["f16_peak_tflops"] is None  # this fixture's rows use the bf16 metric
    assert set(h20["hbm_bw_gbs"]) == {"value", "commit", "date"}
    assert set(h20["residency"]) == {"peak", "static", "transient", "commit", "date"}


def test_calibration_uses_f16_peak_for_a_bf16_less_card(tmp_path):
    """A device with an f16 peak but no bf16 peak (sm70 V100) still resolves a roofline
    floor, through the f16 metric; bf16-present devices keep bf16 and ignore f16."""
    rows = [
        _cal_row("v-bw", cal.BW_METRIC, 900.0, V100, 0, "2026-09-11T00:00Z"),
        _cal_row("v-f16", cal.F16_PEAK_METRIC, 125.0, V100, 0, "2026-09-11T00:00Z"),
        _cal_row("h-bw", cal.BW_METRIC, 3292.0, H20, 0, "2026-09-11T00:00Z"),
        _cal_row("h-bf", cal.PEAK_METRIC, 137.0, H20, 0, "2026-09-11T00:00Z"),
        _cal_row("h-f16", cal.F16_PEAK_METRIC, 140.0, H20, 0, "2026-09-11T01:00Z"),
    ]
    p = _store(tmp_path, rows)
    loaded = cal.load_rows(p)
    v = cal.calibration(loaded, V100)
    h = cal.calibration(loaded, H20)
    assert v == {"bw_gbs": 900.0, "peak_tflops": 125.0, "peak_metric": cal.F16_PEAK_METRIC,
                 "fp8_peak_tflops": None, "pcie_gbs": None}
    assert h["bw_gbs"] == 3292.0 and h["peak_tflops"] == 137.0
    assert h["peak_metric"] == cal.PEAK_METRIC  # bf16 wins where present
    by = {s["device"]: s for s in cal.device_sections(loaded)}
    assert by[V100]["bf16_peak_tflops"] is None and by[V100]["f16_peak_tflops"]["value"] == 125.0


def test_device_sections_empty_store_is_all_pending(tmp_path):
    p = _store(tmp_path, [])
    assert cal.device_sections(cal.load_rows(p)) == []


def test_calibration_pcie_floor_is_optional_and_named(tmp_path):
    """The sparse PCIe H2D floor resolves when a pcie_h2d_gbs row exists and stays None
    otherwise (bw/bf16 still required); same exact device-name keying as the others."""
    p = _store(tmp_path, [_row(cal.BW_METRIC, 4000.0, "GB/s", H20),
                          _row(cal.PEAK_METRIC, 989.0, "TFLOP/s", H20),
                          _row(cal.PCIE_METRIC, 24.0, "GB/s", H20)])
    assert cal.calibration(cal.load_rows(p), H20) == {
        "bw_gbs": 4000.0, "peak_tflops": 989.0, "peak_metric": cal.PEAK_METRIC,
        "fp8_peak_tflops": None, "pcie_gbs": 24.0}
    # wrong device name does not pick up the pcie floor either
    assert cal.calibration(cal.load_rows(p), V100) is None


def test_kernels_sparse_table_renders_derived_hbm_and_pcie_bounds(tmp_path, monkeypatch, capsys):
    """--sparse-k renders the selection table: the scorer HBM bound from bw/peak and the
    cold-fetch PCIe bound from the pcie_h2d floor; both are DERIVED (ms not measured).
    With no pcie row the PCIe column is pending while HBM still resolves."""
    import argparse

    from tilerl import cli

    def ns(pcie_name):
        return argparse.Namespace(
            model="tiny", batches="1", context=256, prefill=0, checkpoint=None,
            device_name=pcie_name, sparse_k=4, scorer="index", kv_fp8="")

    # Force the floors (a cached benchrec singleton from another test can shadow a
    # setenv, so patch the lookup rather than rely on the store path).
    monkeypatch.setattr(cal, "load_rows", lambda *a, **k: [
        _row(cal.BW_METRIC, 4000.0, "GB/s", "tiny-cpu-floor"),
        _row(cal.PEAK_METRIC, 100.0, "TFLOP/s", "tiny-cpu-floor"),
        _row(cal.PCIE_METRIC, 24.0, "GB/s", "tiny-cpu-floor")])
    cli.cmd_bench_kernels(ns("tiny-cpu-floor"))
    out = capsys.readouterr().out
    assert "sparse_indexer_score" in out and "sparse_cold_fetch" in out
    assert "sparse selection k_pages=4 scorer=index" in out
    assert "HBM bound" in out and "PCIe bound" in out
    # The scorer moves bytes over HBM only (one HBM bound, its PCIe bound pending);
    # the fetch moves bytes over PCIe only and its bound is derived (not pending).
    score_l = next(l for l in out.splitlines() if "sparse_indexer_score" in l)
    fetch_l = next(l for l in out.splitlines() if "sparse_cold_fetch" in l)
    assert score_l.count("ms") == 1 and "pending" in score_l
    assert fetch_l.count("ms") == 2 and "pending" not in fetch_l
    assert "24,576" in fetch_l  # tiny: 12 hot pages x 2048 B bf16 block


def test_pack_fp4_chunked_matches_whole_pack():
    """Row-chunked packing (the 32 GB-safe fixture path) is byte/value-identical to
    packing the whole weight at once — pack and renorm are per-row."""
    import torch
    from tilerl_kernels import reference

    torch.manual_seed(0)
    w = torch.randn(137, 64) * 0.3  # non-multiple of the 32-row chunk forces a tail
    wq_w, sc_w = reference.pack_fp4(w)
    sc_w, os_w = reference.renorm_fp4_scale(sc_w)
    wq_c, sc_c, os_c = cal._pack_fp4_chunked(w, row_chunk=32)
    assert torch.equal(wq_c, wq_w)
    assert torch.equal(sc_c, sc_w)
    assert torch.equal(os_c, os_w)
