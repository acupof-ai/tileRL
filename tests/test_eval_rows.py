"""The eval must score with the reward's matcher, and write down which problem
went which way."""

from __future__ import annotations

import json

from tilerl.engine import SamplingParams
from tilerl.eval import gsm8k_accuracy
from tilerl.math_answer import boxed_match


def _run(monkeypatch, texts, rows, match):
    """No engine: generate_ids is replaced, so this exercises the scoring and the
    row bookkeeping only. ids[i] is i repeated i+1 times, so token counts differ
    per problem and a row that copied the wrong length would show.

    `render_chat` is patched where it is DEFINED, not on tilerl.eval: gsm8k_accuracy
    imports it inside the function body, so an attribute set on the eval module is
    never read."""
    ids = [[i] * (i + 1) for i, _ in enumerate(texts)]
    monkeypatch.setattr("tilerl.eval.generate_ids", lambda *a, **k: ids)
    monkeypatch.setattr("tilerl.prompt.render_chat", lambda *a, **k: "")
    tok = type("T", (), {"decode": staticmethod(lambda i: texts[i[0]])})()
    out: list = []
    c, n, ntok = gsm8k_accuracy(None, tok, rows, SamplingParams(),
                                match=match, per_problem=out)
    return c, n, ntok, out


def test_last_number_scores_a_wrong_fraction_correct(monkeypatch):
    """`\\frac{3}{2}` against gold `\\frac{1}{2}`: both have last number 2, so the
    default matcher calls a wrong answer right. This is the whole reason the eval
    takes the reward's matcher on MATH -- with it, the arm reads 100%."""
    texts = [r"\boxed{\frac{3}{2}}"]
    rows = [{"prompt": "p", "answer": r"\frac{1}{2}"}]

    c, _, _, _ = _run(monkeypatch, texts, rows, None)
    assert c == 1, "documents the defect: last_number scores the wrong fraction correct"

    c, _, _, _ = _run(monkeypatch, texts, rows, boxed_match)
    assert c == 0, "boxed_match must reject it"


def test_every_reward_name_has_a_matcher():
    """The reward and the eval read one table, so they cannot disagree. They did:
    the reward matched \\boxed{} while the eval matched the last number. A new
    `--reward` choice that forgets its row fails here rather than at run time,
    where it looks like a training effect."""
    from tilerl.cli import _build_parser
    from tilerl.eval import MATCHERS

    choices = next(a.choices for a in _build_parser()._subparsers._group_actions[0]
                   .choices["train"]._actions if a.dest == "reward")
    assert set(choices) == set(MATCHERS), (
        f"--reward accepts {sorted(choices)} but MATCHERS has {sorted(MATCHERS)}: "
        "a reward with no matcher scores with the wrong one")
    assert all(callable(m) for m in MATCHERS.values())


def test_per_problem_rows_make_the_pairing_recoverable(monkeypatch, tmp_path):
    """Totals alone force the unpaired interval. One row per problem, with its
    index, is the minimum that lets two arms over the same set be compared paired."""
    texts = [r"\boxed{1}", r"\boxed{9}"]
    rows = [{"prompt": "a", "answer": "1"}, {"prompt": "b", "answer": "2"}]

    c, n, ntok, out = _run(monkeypatch, texts, rows, boxed_match)
    assert (c, n) == (1, 2)
    assert [r["i"] for r in out] == [0, 1]
    assert [r["correct"] for r in out] == [True, False]
    assert sum(r["tokens"] for r in out) == ntok, "row tokens must sum to the total"

    # A second arm disagreeing on problem 1 is only visible per problem.
    _, _, _, out2 = _run(monkeypatch, [r"\boxed{1}", r"\boxed{2}"], rows, boxed_match)
    assert [r["correct"] for r in out2] == [True, True]
    disagree = sum(a["correct"] != b["correct"] for a, b in zip(out, out2))
    assert disagree == 1, "the discordant pair McNemar needs"

    (tmp_path / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    assert len((tmp_path / "eval.jsonl").read_text().splitlines()) == 2


def test_the_rollout_cap_must_clear_the_measured_completion_length():
    """A cap below what the policy needs truncates every rollout before the answer,
    so every group ties at the floor and GRPO trains on a constant reward.

    Measured on MATH level 5: the base policy averages 1038 tokens, the recipe's cap
    was 512, and 5 of the first 6 steps came back reward 0.0000 tied 1.00 tok 512.
    The one untied step is the one whose completions fit (tok 259, reward 0.75).
    """
    import pytest

    from tilerl.cli import _ROLLOUT_HEADROOM, _refuse_short_rollouts

    _refuse_short_rollouts(1038, 2048)  # the fix: headroom, no raise
    _refuse_short_rollouts(None, 512)  # no before-arm measured: nothing to compare

    with pytest.raises(SystemExit) as e:
        _refuse_short_rollouts(1038, 512)  # what run 2 actually did
    assert "1038" in str(e.value) and "512" in str(e.value), (
        "the message must name both numbers; 'rollouts too short' sends the reader "
        "back to the log to find them")

    # The boundary is the mean, so a cap the mean exactly fits still truncates half.
    _refuse_short_rollouts(_ROLLOUT_HEADROOM * 1000, 1000)
    with pytest.raises(SystemExit):
        _refuse_short_rollouts(_ROLLOUT_HEADROOM * 1000 + 1, 1000)


def test_training_stops_on_a_cap_the_policy_cannot_answer_in(tmp_path, monkeypatch):
    """The guard through the CLI, because everything above it passes with the call
    site deleted -- a guard that exists and never runs.

    `--max-new-tokens 4` on the tiny model against a 32-token eval arm: the refusal
    must name both numbers, and `--allow-short-rollouts` must train through it.
    """
    import pytest

    from tilerl.cli import _build_parser, cmd_train

    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    data = tmp_path / "d.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n{"prompt": "2+2?", "answer": "4"}\n')
    argv = ["train", "--rl", "--data", str(data), "--eval-gsm8k", str(data), "--steps", "1",
            "--group", "2", "--max-new-tokens", "4", "--lora-rank", "4",
            "--eval-max-new-tokens", "32"]

    with pytest.raises(SystemExit) as e:
        cmd_train(_build_parser().parse_args(argv))
    assert "32" in str(e.value) and "4" in str(e.value), (
        "the refusal must name the measured length and the cap")

    with pytest.raises(SystemExit) as ok:  # gate failure on a 1-step run, not the guard
        cmd_train(_build_parser().parse_args([*argv, "--allow-short-rollouts"]))
    assert "truncated" not in str(ok.value), (
        "--allow-short-rollouts must reach training; it hit the refusal instead")


def test_eval_rows_are_written_before_the_manifest_exists(tmp_path, monkeypatch):
    """`_finish` creates runs/<id>/ and runs AFTER both eval arms, so a
    `if not is_dir(): return` in the row writer silently wrote nothing -- which is
    what it did on the first MATH run: 500 before-arm rows measured, zero on disk.
    """
    from tilerl.cli import _write_eval_rows

    monkeypatch.setattr("tilerl.ledger.runs_root", lambda: tmp_path)
    rows = [{"i": 0, "correct": True, "tokens": 10, "answer": "1"},
            {"i": 1, "correct": False, "tokens": 30, "answer": "2"}]
    assert not (tmp_path / "r1").exists(), "the run dir must not exist yet"

    mean = _write_eval_rows("r1", "before", rows)

    written = (tmp_path / "r1" / "eval-before.jsonl").read_text().splitlines()
    assert len(written) == 2, "the rows went nowhere; the guard returned early"
    assert mean == 20.0, "the mean completion length is what the rollout guard reads"


def test_before_eval_cache_reuses_rows_and_invalidates_length(tmp_path, monkeypatch):
    from contextlib import suppress

    from tilerl import cli
    from tilerl import eval as eval_mod
    from tilerl.ledger import list_runs

    root = tmp_path / "runs"
    monkeypatch.setenv("TILERL_RUNS", str(root))
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n')
    monkeypatch.setattr(eval_mod, "mmlu_questions", lambda *a: (["1+1? A. 2 B. 3"], ["A"], ["math"]))
    calls = []
    generate = eval_mod.generate_ids

    def counted(*args, **kwargs):
        calls.append(1)
        return generate(*args, **kwargs)

    monkeypatch.setattr(eval_mod, "generate_ids", counted)

    def run(lr, cap, load=None):
        previous = {m["id"] for m in list_runs(root)}
        calls.clear()
        argv = [
            "train", "--rl", "--model", "tiny", "--steps", "1", "--group", "2",
            "--data", str(data), "--eval-gsm8k", str(data), "--eval-mmlu", "1",
            "--max-new-tokens", "4", "--eval-max-new-tokens", str(cap),
            "--lora-rank", "2", "--lr", str(lr), "--allow-short-rollouts"]
        args = cli._build_parser().parse_args(argv + (["--load-adapter", str(load)] if load else []))
        with suppress(SystemExit):
            cli.cmd_train(args)
        (m,) = [m for m in list_runs(root) if m["id"] not in previous]
        rows = (root / m["id"] / "eval-before.jsonl").read_text()
        return m, rows, len(calls)

    first, rows, first_calls = run(0.001, 4)
    second, cached_rows, second_calls = run(0.002, 4)
    assert first_calls == 4 and second_calls == 2, "only the after arm may sample on a cache hit"
    assert not first["eval_before_cache"]["cache_hit"]
    assert second["eval_before_cache"]["cache_hit"]
    key = second["eval_before_cache"]["key"]
    assert key == first["eval_before_cache"]["key"] and len(key) == 64
    assert cached_rows == rows
    saved = json.loads((root / "eval-cache" / f"{key}.json").read_text())
    assert saved["rows"] == [json.loads(line) for line in rows.splitlines()]
    assert all(second["metrics"][k] == v for k, v in saved["metrics"].items())
    third, _, third_calls = run(0.002, 8)
    assert third_calls == 4 and not third["eval_before_cache"]["cache_hit"]
    assert third["eval_before_cache"]["key"] != key
    loaded, _, loaded_calls = run(0.002, 4, root / first["id"] / "adapter.safetensors")
    assert loaded_calls == 4 and "eval_before_cache" not in loaded


def test_before_eval_cache_tracks_checkpoint_file_mtime(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    from tilerl import cli
    from tilerl.config import tiny

    monkeypatch.setattr(cli, "_QWEN38_SOURCE", str(tmp_path))
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"a")
    args = cli._build_parser().parse_args(["train", "--model", "qwen38-27b"])
    backend = SimpleNamespace(target="cpu", precision="bf16")
    key = cli._before_eval_key(args, tiny(), backend, SamplingParams(), None)
    stat = weights.stat()
    weights.write_bytes(b"b")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert cli._before_eval_key(args, tiny(), backend, SamplingParams(), None) != key


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
