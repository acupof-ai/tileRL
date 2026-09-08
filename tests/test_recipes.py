"""A recipe is the defaults; typed flags win; the manifest records the name."""

import json
import os

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
    # The four published phases must reconstruct the step EXACTLY: other_secs is a
    # derived remainder, so the identity is arithmetic, not a measurement. A percentage
    # band would be looser than other_secs itself (0.054% of the tiny step) and would
    # pass with the remainder zeroed -- measured; that is the mutant this arm exists for.
    parts = phases + [metrics["other_secs"]]
    assert abs(sum(parts) - metrics["secs_total"]) <= 1e-9 * metrics["steps_completed"], (
        f"phases do not reconstruct the step: {sum(parts)} vs {metrics['secs_total']}",
        metrics)
    # forward_secs is carved out of backward_secs, so it must be a strict part of it,
    # and backward_only_secs is what remains. A forward timed with no device sync
    # reads ~0 on cuda and this is the arm that catches it.
    assert 0 < metrics["forward_secs"] < metrics["backward_secs"], metrics
    assert abs(metrics["forward_secs"] + metrics["backward_only_secs"]
               - metrics["backward_secs"]) <= 1e-6, metrics


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


def test_math_reward_scores_what_the_number_matcher_cannot():
    """MATH answers are symbolic, and `--reward number` does not fail on them -- it
    scores them WRONG, which is worse: `last_number` reads the denominator, so
    `\\frac{1}{2}` and `\\frac{3}{2}` both parse to 2.0 and a wrong rollout is
    rewarded. The recipe therefore carries `reward="boxed"`, and this asserts the
    two matchers genuinely disagree -- a flag selecting an equivalent function is
    not a fix."""
    from tilerl.eval import answer_match
    from tilerl.math_answer import boxed_match

    wrong = r"we get \boxed{\frac{3}{2}}"
    assert answer_match(wrong, r"\frac{1}{2}"), "last_number compares the denominators"
    assert not boxed_match(wrong, r"\frac{1}{2}"), "boxed_match must reject it"
    assert boxed_match(r"we get \boxed{\frac{1}{2}}", r"\frac{1}{2}")
    assert flags("grpo-math-27b")["reward"] == "boxed"


def test_math_prompts_ask_for_the_box():
    """`boxed_match` scores nothing unless the model boxes, and nothing downstream
    asks it to: `render_chat` emits bare ChatML and the recipe runs thinking off.
    Without the instruction every group ties at the FLOOR, which --level cannot fix.
    Asserted against the generator's own constant, so removing it fails here."""
    import re
    from pathlib import Path

    from tilerl.math_answer import boxed_match

    src = Path(__file__).resolve().parents[1] / "scripts" / "math_jsonl.py"
    text = src.read_text()
    assert re.search(r"_INSTRUCTION\s*=", text), "the generator lost its instruction constant"
    assert '+ _INSTRUCTION' in text, "the instruction is defined but not appended to the prompt"
    assert r"\\boxed" in text
    # and the matcher it exists to satisfy really does need the box:
    assert not boxed_match("the answer is 2", "2")


def test_math_jsonl_reads_every_subject_and_writes_the_level_it_filtered(tmp_path, monkeypatch):
    """The generator must not ask for a config the dataset does not have, and the file it
    writes must carry the level it filtered on.

    `EleutherAI/hendrycks_math` has SEVEN configs, one per subject, and no "all" -- so
    `load_dataset(repo, "all")` raises before reading a row, and this script could never
    have run. Run 2's files came from a throwaway builder instead, which is how the eval
    file ended up levels 3-5 under a level-5 name
    (errors/2026-09-05-the-eval-file-was-not-the-level-it-was-named.md).

    Hermetic: CI has neither `datasets` nor network, so the module is stubbed with the
    real repo's behaviour -- an unknown config RAISES. That is what makes a revert to
    "all" fail here rather than only on a machine with the dataset.
    """
    import json
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    stub = tmp_path / "datasets.py"
    stub.write_text(textwrap.dedent('''
        _SUBJECTS = ["algebra", "geometry"]
        _ROWS = {
            "algebra": [("2+2?", "Level 5", r"so \\\\boxed{4}"),
                        ("1+1?", "Level 3", r"so \\\\boxed{2}")],
            "geometry": [("area?", "Level 5", r"so \\\\boxed{9}"),
                         ("nobox?", "Level 5", "no box here")],
        }

        def get_dataset_config_names(repo):
            return list(_SUBJECTS)

        def load_dataset(repo, config, split=None):
            if config not in _SUBJECTS:
                raise ValueError(
                    f"BuilderConfig {config!r} not found. Available: {_SUBJECTS}")
            return [{"problem": p, "level": lv, "solution": s, "type": config}
                    for p, lv, s in _ROWS[config]]
    '''))
    out = tmp_path / "l5.jsonl"
    env = {**os.environ, "PYTHONPATH": f"{tmp_path}:{root / 'src'}"}
    r = subprocess.run([sys.executable, str(root / "scripts" / "math_jsonl.py"),
                        "test", str(out), "--level", "5"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-800:]
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    # Both subjects reached, so it did not stop at one config; the Level 3 row is
    # filtered out; the no-boxed row is DROPPED rather than given an empty answer.
    assert [r_["answer"] for r_ in rows] == ["4", "9"], rows
    assert {r_["level"] for r_ in rows} == {"Level 5"}, rows
    assert "1 dropped" in r.stdout, r.stdout
    # The histogram prints what was WRITTEN, which is the check the 09-05 defect lacked.
    assert "levels written: {'Level 5': 2}" in r.stdout, r.stdout
