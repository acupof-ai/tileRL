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


def test_tiebreak_generates_all_56_judgements_in_one_batched_call():
    """The P1 entry claims one judged group of 8 issues C(8,2)x2 = 56 one-token
    generations in ONE batched generate call (28 pairs, both prompt orders for the
    position-bias control). A per-pair loop costs 28 round trips per group and
    silently costs more than the training step. This gate counts the calls and the
    batch width at the generate seam; a per-pair loop goes red.

    The tiebreaker closes over eval.generate, so it is stubbed here — what is under
    test is the batching in cli._judge_tiebreak, not the engine.
    """
    from tilerl.cli import _judge_tiebreak

    calls = []

    def fake_generate(engine, tok, prompts, sp, concurrency):
        calls.append(list(prompts))
        # one-token answers; mix A/B/tie so the verdicts are usable downstream
        ans = ["A", "B", "tie"]
        return [ans[k % 3] for k in range(len(prompts))]

    import tilerl.eval as eval_mod

    from tilerl.engine import SamplingParams
    from tilerl.tokenizer import ByteTokenizer

    orig = eval_mod.generate
    eval_mod.generate = fake_generate
    try:
        params = SamplingParams(temperature=1.0, max_new_tokens=8, max_think_tokens=0)
        tok = ByteTokenizer()
        tiebreak = _judge_tiebreak(engine=None, tok=tok, params=params)
        prompt = list(range(3, 32))  # any token ids; decoded to the question text
        comps = [list(range(20 + i, 26 + i)) for i in range(8)]  # 8 distinct completions
        passed = [False] * 8  # all-fail band -> every pair judged
        tiebreak(prompt, comps, passed)
    finally:
        eval_mod.generate = orig

    # Exactly ONE generate call for the whole group.
    assert len(calls) == 1, f"expected 1 batched judge call, got {len(calls)} (per-pair loop?)"
    batch = calls[0]
    # C(8,2) pairs x 2 orders = 56 one-token generations.
    assert len(batch) == 56, f"expected 56 judge prompts, got {len(batch)}"
    # Both orders are present: the 56 prompts are 28 ab + 28 ba prompts, every one of
    # the 56 names both solution slots, and each pair's two orders differ (A/B swap).
    assert all("[A]" in p and "[B]" in p for p in batch)
    ab, ba = batch[:28], batch[28:]
    assert len(set(ab)) == 28 and len(set(ba)) == 28
    assert ab != ba and not set(ab) & set(ba), "the order halves must be distinct prompts"
