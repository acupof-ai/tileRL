"""The run ledger: ids are a function of the inputs, a finished run is not
rerun, gates are data in the manifest and an exit code in the CLI."""

import json

from tilerl.cli import _build_parser, cmd_ledger, cmd_train
from tilerl.ledger import (
    gates_pass,
    lineage,
    list_runs,
    new_manifest,
    read_manifest,
    run_id,
    write_manifest,
)


def test_run_id_is_canonical():
    assert run_id({"a": 1, "b": [2, 3]}) == run_id({"b": [2, 3], "a": 1})
    assert run_id({"a": 1}) != run_id({"a": 2})
    assert len(run_id({})) == 12


def test_manifest_round_trip_and_lineage(tmp_path):
    parent = new_manifest("train", {"x": 1})
    child = new_manifest("eval", {"x": 2}, parents=[parent["id"]])
    child["gates"] = [{"name": "g", "value": 1, "threshold": 0, "passed": True}]
    for m in (parent, child):
        write_manifest(tmp_path, m)
    assert read_manifest(tmp_path, child["id"]) == child and gates_pass(child)
    assert [m["id"] for m in lineage(tmp_path, child["id"])] == [child["id"], parent["id"]]
    assert {m["id"] for m in list_runs(tmp_path)} == {parent["id"], child["id"]}
    assert read_manifest(tmp_path, "missing") is None


def _train(argv):
    """cmd_train exits non-zero on a failed gate; return the code."""
    try:
        cmd_train(_build_parser().parse_args(["train", *argv]))
    except SystemExit as e:
        return e.code
    return 0


def test_train_cli_writes_manifest_and_is_idempotent(tmp_path, monkeypatch, capsys):
    """`tilerl train --rl --data --eval-gsm8k` on the tiny model: the plumbing
    the 27B run uses — JSONL prompts, ChatML, exact-match reward, GSM8K greedy
    eval before and after — and a manifest a second identical call returns
    from instead of retraining."""
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    data = tmp_path / "d.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n{"prompt": "2+2?", "answer": "4"}\n')
    argv = ["--rl", "--data", str(data), "--eval-gsm8k", str(data), "--steps", "2",
            "--group", "2", "--max-new-tokens", "4", "--lora-rank", "4"]
    code = _train(argv)
    (m,) = list_runs(tmp_path / "runs")
    assert [g["name"] for g in m["gates"]] == [
        "reward_rises", "mmlu_holds", "gsm8k_improves", "groups_untied", "ce_falls"]
    # ce_falls carries no ce_first on the RL path, so it passes vacuously here.
    assert m["metrics"].get("ce_first") is None
    assert code == (0 if gates_pass(m) else 1)
    assert isinstance(m["metrics"]["gsm8k_before"], int)
    assert isinstance(m["metrics"]["gsm8k_after"], int)
    assert m["metrics"]["mmlu_before"] is None and m["inputs"]["source"] == "tiny"

    capsys.readouterr()
    assert _train(argv + ["--json"]) == code
    again = json.loads(capsys.readouterr().out)
    assert again["finished"] == m["finished"], "a finished run was retrained"

    cmd_ledger(_build_parser().parse_args(["ledger", "--json"]))
    assert [r["id"] for r in json.loads(capsys.readouterr().out)] == [m["id"]]


def test_sft_writes_a_manifest_and_gates_on_the_loss_falling(tmp_path, monkeypatch, capsys):
    """`tilerl train` without --rl/--opd wrote no manifest at all, so sft-iso-27b
    -- a recipe whose whole purpose is a P3 verdict -- had nowhere to record one.
    The ledger is per-run, not per-algorithm."""
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    argv = ["--model", "tiny", "--steps", "4"]
    code = _train(argv)
    (m,) = list_runs(tmp_path / "runs")
    assert m["inputs"]["algo"] == "sft" and m["inputs"]["optim"] == "adafactor"
    assert code == (0 if gates_pass(m) else 1)
    ce = m["metrics"]
    assert ce["ce_first"] is not None and ce["ce_last"] is not None
    assert ce["secs_per_step_median"] is not None
    # The RL gates have no metrics on this path and must pass vacuously.
    for g in m["gates"]:
        if g["name"] != "ce_falls":
            assert g["passed"] and g["value"] is None, g

    capsys.readouterr()
    assert _train(argv + ["--json"]) == code
    again = json.loads(capsys.readouterr().out)
    assert again["finished"] == m["finished"], "a finished SFT run was retrained"


if __name__ == "__main__":  # runnable check
    test_run_id_is_canonical()
    print("ledger: ids OK")


def test_mmlu_score_reports_the_concurrency_it_used():
    """A score whose value depends on concurrency has to carry it.

    concurrency sets B, B sets M = B*W, and M picks the fp4 linear arm across
    the _MGEMV/_MX boundaries -- so two concurrencies can run two kernels on one
    question, and the 27B showed 4 of 1000 answers moving between them. The two
    callers disagreed silently (cli.py 8, scripts/mmlu.py the default 32).

    Gated on the signature rather than end to end: mmlu_accuracy needs the real
    dataset, and what regresses is a caller unpacking two values again.
    """
    import inspect

    from tilerl.eval import mmlu_accuracy

    src = inspect.getsource(mmlu_accuracy)
    assert "concurrency" in src.split("return")[-1], (
        "mmlu_accuracy must return the concurrency it scored at:\n" + src)

    cli = inspect.getsource(__import__("tilerl.cli", fromlist=["_"]))
    call = next(ln for ln in cli.splitlines() if "mmlu_accuracy(" in ln and "import" not in ln)
    assert call.count(",") >= 2 and "conc" in call, f"cli.py drops the concurrency: {call!r}"
    assert '_concurrency"] = conc' in cli, "cli.py must record it in the manifest"
