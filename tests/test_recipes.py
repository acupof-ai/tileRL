"""A recipe is the defaults; typed flags win; the manifest records the name."""

import contextlib
import json

from tilerl.cli import _build_parser
from tilerl.recipes import RECIPES, flags


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
    with contextlib.suppress(SystemExit):  # the demo reward's gates may fail; the manifest is checked
        main()
    out = capsys.readouterr().out
    m = json.loads(out[out.index("{"):])
    assert m["inputs"]["recipe"] == "grpo-tiny-smoke" and m["inputs"]["steps"] == 2
