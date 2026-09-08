"""The run ledger: ids are a function of the inputs, a finished run is not
rerun, gates are data in the manifest and an exit code in the CLI."""

import json

import pytest

from tilerl.cli import _build_parser, cmd_ledger, cmd_train
from tilerl.ledger import (
    format_run,
    gates_pass,
    lineage,
    list_runs,
    new_manifest,
    now,
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
    # `finished` has to be set for a verdict to mean anything: gates are written by
    # `_finish`, so an unfinished manifest carries none and `gates_pass([])` is True.
    # This line asserted `pass` on a manifest that never finished, which is the defect
    # a killed run hit -- `e069c8ff28b7 train running pass` on cpu.
    assert format_run(child).split()[3] == "killed"
    child["finished"] = now()
    assert format_run(child).split()[3] == "pass"
    # The fourth verdict: finished with no gates DEFINED. Distinct from `skip`, which in
    # this tree means a gate existed and was suppressed, and from `pass`, which is what
    # `gates_pass([])` returned before -- a judgement over zero checks. `tilerl merge` is
    # the producer; `test_merge.py` asserts it end to end.
    gateless = dict(child, gates=[])
    assert format_run(gateless).split()[3] == "none", format_run(gateless)
    assert gates_pass(gateless), "gates_pass is unchanged, so exit codes are unchanged"
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


@pytest.mark.parametrize("mode", ["--rl", "--opd"])
def test_zero_steps_writes_eval_manifest(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n')
    assert _train([mode, "--model", "tiny", "--steps", "0", "--data", str(data),
                   "--eval-gsm8k", str(data), "--eval-max-new-tokens", "4"]) == 0
    (m,) = list_runs(tmp_path / "runs")
    assert m["finished"] and isinstance(m["metrics"]["gsm8k_after"], int)
    assert m["metrics"]["gsm8k_after_tokens"] > 0
    assert not {"reward_first", "reward_last", "ce_last", "secs_per_step_median",
                "secs_total", "tied_group_fraction", "tokens_first", "tokens_last",
                "rollout_secs", "backward_secs", "optimizer_secs"} & m["metrics"].keys()
    assert all(g["skipped"] and g["passed"] is None for g in m["gates"])
    assert gates_pass(m)
    assert format_run(m).split()[3] == "skip"


def test_train_cli_writes_manifest_and_is_idempotent(tmp_path, monkeypatch, capsys):
    """`tilerl train --rl --data --eval-gsm8k` on the tiny model: the plumbing
    the 27B run uses — JSONL prompts, ChatML, exact-match reward, GSM8K greedy
    eval before and after — and a manifest a second identical call returns
    from instead of retraining."""
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    data = tmp_path / "d.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n{"prompt": "2+2?", "answer": "4"}\n')
    # --allow-short-rollouts: max_new_tokens 4 is deliberately below any real
    # completion here, which is exactly what the length guard refuses.
    argv = ["--rl", "--data", str(data), "--eval-gsm8k", str(data), "--steps", "2",
            "--group", "2", "--max-new-tokens", "4", "--lora-rank", "4",
            "--allow-short-rollouts"]
    code = _train(argv)
    (m,) = list_runs(tmp_path / "runs")
    assert [g["name"] for g in m["gates"]] == [
        "rollouts_within_cap", "reward_rises", "mmlu_holds", "gsm8k_improves",
        "groups_untied", "ce_falls"]
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


def test_the_manifest_records_the_engine_config_the_wall_clock_depends_on(tmp_path, monkeypatch):
    """A run's wall clock cannot be compared against another run's without these.

    Six card sessions on the 2.6x rollout tick recovered their two pool sizes only
    because the probe script logged its own flags; the manifest recorded none of the
    bundle. Per-forward device time moves with it, so P5 (against verl+sglang) is
    exactly the comparison a record without it cannot support.

    Two things are asserted, not one. The keys must be present AND `blocks` must
    track the pool -- a key list alone goes green over a hardcoded dict, and
    `--max-new-tokens` is what sizes the training pool
    (`num_blocks = ceil(ctx/16)*8 + 8`, `cli.py:570`), so two runs differing only
    there must report different pools.

    4000, not 200: `ctx` has a 1024 floor (`cli.py:568`), and at 200 both arms land
    on it and report 520 blocks each. Written with 200 first, and the pair-assert is
    what caught it -- a key-presence check would have passed on two identical pools.
    """
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    data = tmp_path / "d.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n')
    seen = {}
    for new in (4, 4000):
        _train(["--rl", "--data", str(data), "--steps", "0", "--group", "2",
                "--max-new-tokens", str(new), "--lora-rank", "4"])
        (m,) = [r for r in list_runs(tmp_path / "runs") if r["inputs"]["max_new_tokens"] == new]
        assert m["engine"].keys() == {
            "blocks", "slots", "max_batch", "max_total_tokens",
            "max_num_batched_tokens", "decode_graph", "prefix_store", "spec_width"}
        assert m["engine"]["slots"] == 8 and m["engine"]["prefix_store"] == "NoPrefixStore"
        seen[new] = m["engine"]["blocks"]
        # Not in `inputs`: the id hashes inputs, so a pool field there would make
        # every pool change a new run instead of a rerun.
        assert "blocks" not in m["inputs"]
    assert seen[4000] > seen[4], f"blocks did not track the context: {seen}"


def test_periodic_rollout_guard_stops_at_first_window_crossing(tmp_path, monkeypatch, capsys):
    from contextlib import suppress

    from tilerl import cli, train
    from tilerl.engine import Engine
    from tilerl.ledger import gates_pass, list_runs

    root = tmp_path / "runs"
    monkeypatch.setenv("TILERL_RUNS", str(root))
    data = tmp_path / "data.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n')
    lengths = [6, 8, 10, 12, 14, 16, 18, 20, 20, 20]
    sampled = []
    on_disk = []
    identified = []

    def rollout(engine, ids, what):
        # Sampled at the START of each step, so it reads what earlier steps wrote:
        # a single write after the loop leaves this all zeros, and a run killed
        # mid-training keeps nothing. Run 2 died on a SIGTERM at step 45.
        found = [p for p in root.glob("*/rollouts.jsonl")]
        on_disk.append(sum(len(p.read_text().splitlines()) for p in found))
        # And the rows are useless without the run that made them: rollouts.jsonl
        # carries no model, cap, group, lr, seed or commit, and the directory name
        # is a hash of those. Measured before the fix: a SIGTERM at step 6 left 12
        # rows, no manifest, and `tilerl ledger --json` printed [].
        identified.append(len(list_runs(root)))
        n = lengths[len(sampled)]
        sampled.append(n)
        return {i: [4] * n for i in ids}

    # Keep the real GRPO loop and CLI; only generation and the expensive update are stubbed.
    def update(*a, timings, **kw):
        timings.update(backward_secs=0.01, optimizer_secs=0.001)
        return 1.0

    monkeypatch.setattr(train, "_drain", rollout)
    monkeypatch.setattr(train, "rl_step", update)
    argv = ["train", "--rl", "--data", str(data), "--steps", "10", "--group", "2",
            "--max-new-tokens", "20", "--lora-rank", "2"]
    for allow, expected in ((False, 9), (True, 10)):
        sampled.clear()
        on_disk.clear()
        identified.clear()
        # No pending requests in the stubbed drain: submit only supplies unique ids.
        requests = iter(range(20))
        monkeypatch.setattr(Engine, "submit", lambda *a, **kw: next(requests))
        with suppress(SystemExit):
            cli.cmd_train(cli._build_parser().parse_args(
                argv + (["--allow-short-rollouts"] if allow else [])))
        assert len(sampled) == expected, "periodic guard stopped at the wrong step"
        # Every completed step is on disk before the next one starts. The baseline is
        # arm 1's rows, still there when arm 2 runs; the deltas are what this asserts.
        assert [n - on_disk[0] for n in on_disk] == [2 * i for i in range(expected)], (
            f"rollout rows are not written per step: {on_disk}")
        # This run is in the ledger from its first step, not only once it finishes.
        # A run interrupted before `_finish` is otherwise a directory of rows nothing
        # can attribute: `list_runs` globs */manifest.json and would return [].
        assert min(identified) >= 1, (
            f"the run was not in the ledger while it ran: list_runs saw {identified}")
        m = next(m for m in list_runs(root) if m["inputs"]["allow_short_rollouts"] == allow)
        gate = next(g for g in m["gates"] if g["name"] == "rollouts_within_cap")
        assert m["metrics"]["steps_completed"] == expected
        assert (root / m["id"] / m["artifacts"]["adapter"]).is_file()
        if allow:
            assert gate["skipped"] and gate["passed"] is None
        else:
            assert not gate["skipped"] and gate["passed"] is False and not gates_pass(m)
            assert gate["value"] == pytest.approx(17.6)
            assert (gate["threshold"], gate["step"]) == (16.0, 9)
            # The after-arm never ran, so these must read "not measured", not "pass":
            # _finish scores a None value as passed, which is how a stopped run would
            # report mmlu_holds and gsm8k_improves green having measured neither.
            for name in ("mmlu_holds", "gsm8k_improves"):
                g = next(x for x in m["gates"] if x["name"] == name)
                assert g["skipped"] and g["passed"] is None, name
            assert m["metrics"].get("mmlu_after") is None
            assert "step 9" in gate["reason"] and "--max-new-tokens is 20" in gate["reason"]
            assert gate["reason"] in capsys.readouterr().out


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
