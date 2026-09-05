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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
