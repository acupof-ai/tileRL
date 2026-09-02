"""Stage 4(b): judge verdicts reach GRPO as group-normalised advantages.

The end-to-end property, not per-function coverage: a group whose rollouts
tests cannot separate still produces a usable gradient signal, and a judge that
cannot separate them produces none. Both halves matter -- the second is what
stops a useless judge from injecting noise that looks like learning.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import numpy as np
import pytest

from tilerl.judge import copeland_scores, judge_rewards, pair_verdict
from tilerl.train import group_advantages


def test_tests_outrank_the_judge_whatever_it_says():
    """No judged ordering lifts a failing rollout above a passing one."""
    # A judge that always prefers the second argument, i.e. maximally hostile
    # to the pass ordering it is shown.
    scores, _ = judge_rewards(["p1", "p2", "f1", "f2"], [True, True, False, False],
                              lambda a, b: ("B", "B"))
    assert min(scores[:2]) > max(scores[2:]), scores
    adv = group_advantages(scores, 4)
    assert adv[0] > 0 and adv[1] > 0, adv
    assert adv[2] < 0 and adv[3] < 0, adv


def test_an_all_pass_group_gets_signal_from_the_judge_alone():
    """The case the judge exists for: tests tie, the judge orders them.

    Without a judge every reward here is 1.0, the group is tied, and
    group_advantages returns zeros -- a whole group of rollouts with no
    gradient. This is the gap stage 4(b) fills.
    """
    ranked = {("a", "b"): ("A", "A"), ("a", "c"): ("A", "A"), ("b", "c"): ("A", "A")}

    def judge(x, y):
        return ranked[(x, y)]

    scores, rows = judge_rewards(["a", "b", "c"], [True] * 3, judge, group_id="t1")
    adv = group_advantages(scores, 3)
    assert not np.allclose(adv, 0.0), (scores, adv)
    assert adv[0] > adv[1] > adv[2], adv
    # Without the judge: identical rewards, zero advantage, no learning.
    assert np.allclose(group_advantages([1.0, 1.0, 1.0], 3), 0.0)
    # Every row is labelled for the scorer's validation split.
    assert {r["subgroup"] for r in rows} == {"all_pass"}
    assert [r["pair_id"] for r in rows] == ["t1:0v1", "t1:0v2", "t1:1v2"]


def test_a_judge_that_cannot_separate_injects_no_gradient():
    """Flat scores -> zero advantages. A useless judge must be silent, not noisy."""
    for verdicts in (("tie", "tie"), ("A", "B")):  # always-tie, always-inconsistent
        scores, _ = judge_rewards(list("abcd"), [True] * 4, lambda a, b: verdicts)
        assert len(set(scores)) == 1, (verdicts, scores)
        assert np.allclose(group_advantages(scores, 4), 0.0)


def test_non_transitive_verdicts_produce_no_order():
    """a>b, b>c, c>a is a cycle: equal win counts, so no invented ranking."""
    cycle = {(0, 1): "A", (1, 2): "A", (0, 2): "B"}
    scores = copeland_scores(3, cycle, (0.6, 1.0))
    assert len(set(scores)) == 1, scores
    assert np.allclose(group_advantages(scores, 3), 0.0)


def test_position_swap_is_required_not_assumed():
    """A judge disagreeing with itself across orders abstains."""
    assert pair_verdict("A", "B") == "abstain"
    assert pair_verdict("B", "A") == "abstain"
    assert pair_verdict("tie", "A") == "abstain"
    assert pair_verdict("A", "A") == "A"
    assert pair_verdict("tie", "tie") == "tie"
    assert pair_verdict("A", "garbage") == "abstain"


def test_the_judge_is_never_shown_a_pair_tests_can_separate():
    def explode(a, b):
        raise AssertionError("judge saw a mixed pair")

    scores, rows = judge_rewards(["pass", "fail"], [True, False], explode)
    assert rows == []
    assert scores[0] > scores[1]


def test_mismatched_outcome_length_is_refused():
    with pytest.raises(ValueError, match="rollouts but"):
        judge_rewards(["a", "b"], [True], lambda a, b: ("A", "A"))
