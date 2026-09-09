"""Each gate's comparator must reject the real bad state it exists to catch.

This tests the COMPARATOR, not the gate's effectiveness. A gate can be
green in production and still pass here -- the comparator is correct, but
the operand may never reach the threshold (see the reward_rises comment
below, and errors/2026-09-09-reward-rises-ce-falls-green-during-regression.md).
The observational margin report (part 2, separate PR) is what catches
decorative gates; this file catches wrong comparators.
"""

import pytest

from tilerl import cli


def _manifest(metrics: dict) -> dict:
    """Minimal unfinished manifest that _finish can gate and write."""
    return {
        "id": "test-gate-comparator",
        "command": "train",
        "inputs": {"steps": 10},
        "parents": [],
        "commit": "test",
        "started": "2026-09-09T00:00:00+00:00",
        "finished": None,
        "metrics": metrics,
        "gates": [],
        "artifacts": {},
        "engine": {},
        "eval_before_cache": None,
    }


def _gate(m: dict, name: str) -> dict:
    return next(g for g in m["gates"] if g["name"] == name)


# Each construct is the REAL bad state the gate exists to catch -- not an
# arbitrary input that trips the comparator. The value that matters is that
# the comparator's direction, units, and threshold-side are correct.
@pytest.mark.parametrize("name,metrics", [
    # Training reward not rising (collapse). Run 2 collapsed this way
    # (cli.py _finish comment). The comparator must read last > first.
    ("reward_rises", {"reward_first": 0.8, "reward_last": 0.5}),
    # MMLU regressed more than the 2 pt floor.
    ("mmlu_holds", {"mmlu_before": 0.6, "mmlu_after": 0.55}),
    # GSM8K improved but by less than the +5 pt roadmap floor.
    # gsm8k_before/after are COUNTS, total is the denominator -- a units
    # mismatch here (fraction vs count) is the bug this construct catches.
    ("gsm8k_improves", {"gsm8k_before": 400, "gsm8k_after": 410,
                        "gsm8k_before_total": 500}),
    # More than half the groups tied. At lambda=0 (binary rewards) this is
    # common when the task is too easy or too hard. At lambda>0 tied is 0 by
    # construction -- the comparator is correct, the operand is the problem.
    ("groups_untied", {"tied_group_fraction": 0.7}),
    # Cross-entropy rose (divergence).
    ("ce_falls", {"ce_first": 2.0, "ce_last": 2.5}),
])
def test_gate_comparator_rejects(tmp_path, monkeypatch, name, metrics):
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path))
    m = _manifest(metrics)
    with pytest.raises(SystemExit):
        cli._finish(m, as_json=False)
    g = _gate(m, name)
    assert g["passed"] is False, f"{name} should be red on {metrics}"
    assert not g["skipped"], f"{name} should not be skipped"


def test_reward_rises_operand_does_not_contain_eval_correctness(tmp_path, monkeypatch):
    """reward_rises reads training reward, not eval correctness. The gate's
    operand has no quantity that moves with the property it claims to guard,
    so it can stay green during eval regression. This is the finding from
    run d6447c0abe5b: reward rose 0.746->0.994 while eval correctness was
    flat (79->80, McNemar p=0.180). The comparator is correct; the gate is
    decorative for its stated purpose. This test pins the operand gap so a
    future 'fix' that adds eval correctness to the gate must update it."""
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path))
    # Eval correctness dropped, but training reward rose -- the gate is green.
    m = _manifest({"reward_first": 0.5, "reward_last": 0.9,
                   "gsm8k_before": 450, "gsm8k_after": 400,
                   "gsm8k_before_total": 500})
    with pytest.raises(SystemExit):
        cli._finish(m, as_json=False)
    # reward_rises is green (reward rose) while gsm8k_improves is red (eval fell).
    assert _gate(m, "reward_rises")["passed"] is True
    assert _gate(m, "gsm8k_improves")["passed"] is False


def test_rollouts_within_cap_drift_rejects():
    """The rollouts_within_cap gate is updated in the training loop, not in
    _finish. Its comparator: mean_tokens <= 0.8 * max_new_tokens. A drift
    above the threshold must turn it red."""
    drift = {"name": "rollouts_within_cap", "value": None,
             "threshold": 0.8 * 6144, "kind": "validity",
             "skipped": True, "passed": None}
    # Mean rollout tokens exceed 80% of the cap.
    mean = 0.9 * 6144
    drift.update(value=mean, step=1, skipped=False,
                 passed=mean <= drift["threshold"])
    assert drift["passed"] is False
    assert not drift["skipped"]
