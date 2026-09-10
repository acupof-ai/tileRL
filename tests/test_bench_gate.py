"""The baseline gate's two silent failure modes: a beat that writes itself, and a
lower-is-better number fed into a higher-is-better row."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_harness as bh  # noqa: E402
import benchrec  # noqa: E402


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


def test_client_side_collector_cannot_default_the_server_device(monkeypatch, tmp_path):
    """A client-side collector measures a remote server it cannot see, so it must not
    default any field describing that server. 2026-09-09: bench_chat_cold_warm and two
    siblings defaulted --device-name to H20, and cpu runs landed in the store labeled
    H20 -- a mislabeled row is worse than a missing one, because it enters every
    device-grouped view and every measured-best comparison.

    The known-devices gate reads the store, so this test seeds its own: against the
    real store its verdict would be a function of repo state (delete the cpu rows and
    the accepted case starts rejecting; add a "CPU" row and the typo case goes green)."""
    import argparse

    import pytest

    store = tmp_path / "measurements.jsonl"
    store.write_text(json.dumps({"device": {"name": "cpu"}}) + "\n")
    monkeypatch.setattr(benchrec, "STORE", store)

    ap = argparse.ArgumentParser()
    benchrec.add_record_args(ap, client_side=True)
    args = ap.parse_args(["--build", "eager", "--target", "cpu"])
    with pytest.raises(SystemExit, match="--device-name required"):
        benchrec.record_common(args)

    ok = ap.parse_args(["--build", "eager", "--target", "cpu", "--device-name", "cpu"])
    assert benchrec.record_common(ok)["device"]["name"] == "cpu"

    typo = ap.parse_args(["--build", "eager", "--target", "cpu", "--device-name", "CPU"])
    with pytest.raises(SystemExit, match="Did you mean 'cpu'\\?"):
        benchrec.record_common(typo)


def test_a_cpu_target_never_takes_the_cuda_name(monkeypatch, tmp_path):
    """A server-side write with target=cpu on a machine that HAS a GPU must still
    label the row 'cpu': the probe asks the machine, not the target. 2026-09-10:
    #434 moved `_emit_eval_records` onto `record_common`, whose torch probe ran for
    every target, so a cpu run on the pod labeled itself NVIDIA H20 -- a population
    lie the client-side gate cannot see, since this path is server-side."""
    import argparse

    import torch

    store = tmp_path / "measurements.jsonl"
    store.write_text(json.dumps({"device": {"name": "cpu"}}) + "\n")
    monkeypatch.setattr(benchrec, "STORE", store)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _d: "NVIDIA H20")

    ap = argparse.ArgumentParser()
    benchrec.add_record_args(ap)
    args = ap.parse_args(["--build", "eager", "--target", "cpu"])
    assert benchrec.record_common(args)["device"]["name"] == "cpu"


def test_a_directionless_metric_is_never_judged_and_a_directional_one_still_fails(
        monkeypatch, tmp_path, capsys):
    """rollout_tokens has no monotonic direction: a 13.9-then-347 population (the
    2026-09-10 P1 collapse shape) must draw no PASS/FAIL in either regression
    section. The decode_tok_s pair is the control: an obvious regression there
    must still FAIL -- the first assertion alone is indistinguishable from the
    whole judgment being switched off."""
    def row(metric, value, date, floor_kind, floor_value):
        r = {
            "build": "eager", "cmd": "test", "commit": "0" * 40, "date": date,
            "device": {"card": None, "name": "cpu"}, "dirty": False,
            "floor": {"derivation": "test", "kind": floor_kind, "unit": "x",
                      "value": floor_value},
            "metric": metric, "model": "tiny", "n": 2, "shape": {"steps": 100},
            "spread": 0.0, "target": "cpu", "unit": "x", "value": value,
            "warm": {"state": "warm"},
        }
        r["id"] = benchrec.new_id(r)
        return r

    store = tmp_path / "measurements.jsonl"
    store.write_text("\n".join(json.dumps(r) for r in [
        row("rollout_tokens", 13.9, "2026-09-09", "measured-best", 13.9),
        row("rollout_tokens", 346.8, "2026-09-10", "reference", 346.8),
        row("decode_tok_s", 100.0, "2026-09-09", "measured-best", 100.0),
        row("decode_tok_s", 50.0, "2026-09-10", "measured-best", 100.0),
    ]) + "\n")
    monkeypatch.setattr(benchrec, "STORE", store)

    reg = benchrec.load_registry()["metrics"]
    assert reg["rollout_tokens"]["direction"] == "none"
    assert bh._gap([r for r in benchrec.load_all()
                    if r["metric"] == "rollout_tokens"][-1], reg) is None
    assert bh._gap([r for r in benchrec.load_all()
                    if r["metric"] == "decode_tok_s"][-1], reg) == 2.0

    bh._view_regress()
    out = capsys.readouterr().out
    assert "rollout_tokens" not in out, "a directionless metric drew a judgment"
    assert "FAIL decode_tok_s" in out, "the control stopped judging -- the skip is too wide"

    bh._view_questions()
    out = capsys.readouterr().out
    assert "rollout_tokens" not in out, (
        "a reference floor is a deliberate non-physical floor, not a missing derivation")


def test_every_registered_collector_exists_and_the_094_metrics_have_one():
    """The collector field is the metric -> script map `tilerl bench <name>` dispatches
    on. A renamed script must fail CI, and a weight-0.94 metric with no collector is
    the batch queue, not an oversight -- gsm8k_pct excepted, with its deferral recorded
    in why_no_collector."""
    import json

    root = Path(__file__).resolve().parents[1]
    reg = json.loads((root / "docs" / "bench-metrics.json").read_text())["metrics"]
    for name, m in reg.items():
        c = m.get("collector")
        if c is None:
            continue
        assert (root / c["script"]).is_file(), (
            f"{name} points at {c['script']}, which does not exist")
    for name, m in reg.items():
        if m["weight"] >= 0.94:
            assert m.get("collector") or m.get("why_no_collector"), (
                f"{name} has weight {m['weight']} and neither a collector nor a "
                "why_no_collector -- it is neither runnable nor a recorded deferral")


def test_every_registered_required_flag_exists_in_its_collector():
    """The registry's `required` list must name flags the script actually accepts:
    a typo (--sorce) or a renamed flag leaves `tilerl bench <name>` failing at runtime
    while CI stays green. Positionals are skipped -- a bare name need not match the
    add_argument string -- and a flag is matched as a quoted literal, which is the
    argparse form. This is an existence check, not a parse: it cannot see a flag the
    script accepts but the registry omits."""
    import json

    root = Path(__file__).resolve().parents[1]
    reg = json.loads((root / "docs" / "bench-metrics.json").read_text())["metrics"]
    # add_record_args adds --build/--target/--device-name/--card/--model-name to every
    # collector, so a required flag the script does not declare itself may come from
    # the shared helper instead.
    helper = (root / "scripts" / "benchrec.py").read_text()
    for name, m in reg.items():
        c = m.get("collector")
        if not c:
            continue
        src = (root / c["script"]).read_text()
        for a in c["required"]:
            if a.startswith("--"):
                assert f'"{a}"' in src or f"'{a}" in src or f'"{a}"' in helper, (
                    f"{name}: {c['script']} has no {a}")


def test_a_query_view_never_writes_the_store(tmp_path):
    """--collectors is a read-only query. A fall-through past the view branch once
    ran the training suite and appended two rows to the permanent store (2026-09-09;
    the cmd field was the only tell). The store is redirected to tmp_path via
    TILERL_BENCH_STORE so the failure path can never touch the real store -- the
    first self-check of this test wrote it and needed a manual restore; a test whose
    failure path needs a human to undo it is not safe to fail."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    store = tmp_path / "measurements.jsonl"
    store.write_bytes(b"")
    env = {**os.environ, "TILERL_BENCH_STORE": str(store)}
    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "bench_harness.py"), "--collectors"],
        env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert store.read_bytes() == b""


def test_compiles_window_counts_a_real_compile():
    """warm.compiles is a measured value, not an asserted 0: a compile in the window
    must register as > 0 through the same expression the engine-direct collectors use.
    2026-09-10: every store row carried compiles=0 hand-filled, and a B=1 arm's 756
    compiles in the timed window was invisible because the field could not say it."""
    import benchrec
    from tilerl_kernels.backend import get_backend

    from tilerl.config import tiny
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.model import build_random

    backend = get_backend()
    engine = build_engine(tiny(), build_random(tiny(), seed=7), backend,
                          num_blocks=8, num_slots=2)
    with benchrec.compiles_window(backend) as w:
        engine.submit([1, 2, 3], SamplingParams(max_new_tokens=4, temperature=0.0))
        engine.step()
    assert w["compiles"] > 0, "the first tick compiles kernels; a 0 here means the field is blind"
