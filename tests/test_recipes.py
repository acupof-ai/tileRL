"""A recipe is the defaults; typed flags win; the manifest records the name."""

import json

import pytest

from tilerl.cli import _build_parser
from tilerl.ledger import gates_pass
from tilerl.recipes import RECIPES, flags


@pytest.mark.parametrize("argv", [
    ["--recipe", "grpo-gsm8k-27b"],
    ["--recipe", "opd-gsm8k-27b"],
    ["--rl"],
    ["--opd"],
    ["--recipe", "grpo-tiny-smoke", "--model", "qwen38-27b"],
])
def test_rl_opd_recipe_requires_data(argv, monkeypatch):
    from tilerl import cli

    monkeypatch.setattr("sys.argv", ["tilerl", "train", *argv])
    monkeypatch.setattr(cli, "_train_adapters", lambda args: None)
    with pytest.raises(SystemExit, match="^error: --data is required for RL/OPD training$"):
        cli.main()


def test_every_recipe_parses_and_flags_override():
    for name in RECIPES:
        args = _build_parser(name).parse_args(["train", "--recipe", name])
        for k, v in flags(name).items():
            assert getattr(args, k) == v, (name, k)
    smoke = _build_parser("grpo-tiny-smoke")
    assert smoke.parse_args(["train", "--recipe", "grpo-tiny-smoke", "--steps", "3"]).steps == 3


def test_recipe_runs_and_is_recorded(tmp_path, monkeypatch, capsys):
    from tilerl.cli import main

    monkeypatch.setenv("TILERL_RUNS", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["tilerl", "train", "--recipe", "grpo-tiny-smoke", "--json"])
    main()  # a passing run returns; _finish exits non-zero only on a failed gate
    # TileLang's kernel-cache warnings go to stdout from C++, so --json cannot
    # promise a lone object; the manifest is the last one printed.
    out = capsys.readouterr().out
    m = json.loads(out[out.index("{"):])
    assert m["inputs"]["recipe"] == "grpo-tiny-smoke"
    assert m["inputs"]["steps"] == flags("grpo-tiny-smoke")["steps"]
    # The CPU smoke recipe is the only one that can fail a gate here, so its verdict
    # IS the assertion: suppressing the exit is what stops it being a gate.
    assert gates_pass(m), m["gates"]
    saved = json.loads((tmp_path / m["id"] / "manifest.json").read_text())
    metrics = saved["metrics"]
    phases = [metrics[k] for k in ("rollout_secs", "backward_secs", "optimizer_secs")]
    assert all(s > 0 for s in phases), metrics
    assert abs(sum(phases) - metrics["secs_total"]) <= 0.2 * metrics["secs_total"], metrics


def test_rl_refuses_a_data_file_with_no_rows(tmp_path, monkeypatch):
    """An empty --data file is the failure #99's flag check cannot see: the flag is
    present, the path exists, and cmd_train's `or [...]` quietly substitutes random
    prompts -- so a 100-step GRPO run trains on noise and still reports a reward."""
    from tilerl import cli

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n  \n")  # blank lines only: `if ln.strip()` drops them all
    monkeypatch.setattr("sys.argv", ["tilerl", "train", "--rl", "--data", str(empty)])
    with pytest.raises(SystemExit, match="has no rows"):
        cli.main()
