"""Pairwise-judge rewards for GRPO: tests decide first, the judge breaks ties.

Stage 4(b) of the Claude Code RL line. The protocol lives in aupai
(``docs/lessons/pairwise_judge_protocol.md``); this is the tileRL half that
turns judgements into the numbers ``grpo_loop`` consumes.

Two rules from the protocol are enforced here rather than documented, because
both are silent when they break:

* **The judge never sees a pair tests can separate.** Rollouts are split by
  test outcome first; the judge is asked only within the all-pass or all-fail
  subgroup. A judge asked to rank a passing rollout against a failing one would
  be scored against a fact it cannot see, and would drift toward whatever it
  prefers stylistically.
* **Only the ORDER is used.** Verdicts become a win count (Copeland), which is
  then mapped to evenly spaced scores inside the subgroup's band. The judge's
  confidence, its wording, and any notion of margin are discarded -- GRPO
  normalises within the group anyway, so a fabricated magnitude would be noise
  with a scale.

# ponytail: Copeland win-count, O(K^2) judge calls per group. RULER-style
# whole-group ranking in one call is the upgrade when K > 4 costs too much.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

__all__ = ["pair_verdict", "copeland_scores", "judge_rewards", "judgement_rows"]

#: Passing rollouts occupy the upper band, failures the lower, so no judged
#: ordering inside a subgroup can lift a failure above a pass. The bands do not
#: touch: sharing an endpoint would let the worst pass tie the best failure,
#: which is the one ordering tests already settled.
_PASS_BAND = (0.6, 1.0)
_FAIL_BAND = (0.0, 0.4)


def pair_verdict(order_ab: str, order_ba: str) -> str:
    """One pair's verdict from two position-swapped judgements.

    Same call both ways wins; anything inconsistent abstains, including one
    side calling a tie. Position bias is a known judge failure mode, so a
    disagreement between orders means the judge could not separate this pair --
    abstaining is the honest reading, not picking the first answer.
    """
    valid = {"A", "B", "tie"}
    if order_ab not in valid or order_ba not in valid:
        return "abstain"
    if order_ab != order_ba:
        return "abstain"
    return order_ab


def copeland_scores(n: int, verdicts: dict[tuple[int, int], str],
                    band: tuple[float, float]) -> list[float]:
    """Win counts -> evenly spaced scores in ``band``, ties and abstentions flat.

    Copeland rather than Elo or Bradley-Terry: pairwise LLM judgements are not
    guaranteed transitive, and a rating model would invent a magnitude from an
    inconsistent set. Counting wins degrades gracefully -- a cycle just leaves
    everyone on the same count, which is exactly "no signal".
    """
    wins = [0.0] * n
    for (i, j), v in verdicts.items():
        if v == "A":
            wins[i] += 1.0
        elif v == "B":
            wins[j] += 1.0
        elif v == "tie":
            wins[i] += 0.5
            wins[j] += 0.5
    lo, hi = band
    spread = sorted(set(wins))
    if len(spread) <= 1:  # every rollout tied, or nothing was judged
        return [(lo + hi) / 2.0] * n
    rank = {w: k for k, w in enumerate(spread)}
    return [lo + (hi - lo) * rank[w] / (len(spread) - 1) for w in wins]


def judgement_rows(group_id: str, verdicts: dict[tuple[int, int], str],
                   subgroup: str, orders: dict[tuple[int, int], tuple[str, str]],
                   test_winner: dict[tuple[int, int], str | None] | None = None
                   ) -> list[dict[str, Any]]:
    """The JSONL rows aupai's ``eval/pairwise_judge.py`` scores.

    ``subgroup`` is carried per row because the scorer needs it: acceptance runs
    on ``mixed`` pairs (where tests are a real gold standard) and production
    runs inside a subgroup, and a file that cannot tell them apart cannot be
    validated -- a tie-everything judge scored 1.0 before that field existed.
    """
    rows = []
    for (i, j), (ab, ba) in sorted(orders.items()):
        rows.append({
            "pair_id": f"{group_id}:{i}v{j}",
            "order_ab": ab,
            "order_ba": ba,
            "test_winner": (test_winner or {}).get((i, j)),
            "subgroup": subgroup,
        })
    return rows


def judge_rewards(rollouts: list[Any], passed: list[bool],
                  judge: Callable[[Any, Any], tuple[str, str]],
                  group_id: str = "g") -> tuple[list[float], list[dict[str, Any]]]:
    """Per-rollout scores for one group, plus the rows the scorer consumes.

    ``passed`` is the test outcome per rollout (the reward that needs no judge).
    ``judge(a, b) -> (order_ab, order_ba)`` must run BOTH orders; this function
    does not call it twice for you, because a judge implementation that ignores
    the swap would then look like agreement rather than a missing control.

    A group where tests already separate everything never reaches the judge:
    that is the protocol's precondition, and it is also the cheap path.
    """
    if len(rollouts) != len(passed):
        raise ValueError(f"{len(rollouts)} rollouts but {len(passed)} test outcomes")
    scores = [0.0] * len(rollouts)
    rows: list[dict[str, Any]] = []
    for members, band, name in ((
            [i for i, p in enumerate(passed) if p], _PASS_BAND, "all_pass"),
            ([i for i, p in enumerate(passed) if not p], _FAIL_BAND, "all_fail")):
        if len(members) < 2:  # nothing to break: 0 or 1 rollout in this band
            for i in members:
                scores[i] = sum(band) / 2.0
            continue
        local = {m: k for k, m in enumerate(members)}
        verdicts: dict[tuple[int, int], str] = {}
        orders: dict[tuple[int, int], tuple[str, str]] = {}
        for i, j in combinations(members, 2):
            ab, ba = judge(rollouts[i], rollouts[j])
            orders[(i, j)] = (ab, ba)
            verdicts[(local[i], local[j])] = pair_verdict(ab, ba)
        local_scores = copeland_scores(len(members), verdicts, band)
        for i in members:
            scores[i] = local_scores[local[i]]
        rows += judgement_rows(group_id, verdicts, name, orders)
    return scores, rows


if __name__ == "__main__":  # pragma: no cover - self-check
    # A pass never scores below a failure, whatever the judge says about either.
    def always_a(a, b):
        return ("A", "A")

    s, _ = judge_rewards(["p1", "p2", "f1", "f2"], [True, True, False, False], always_a)
    assert min(s[0], s[1]) > max(s[2], s[3]), s

    # Position-swapped disagreement abstains rather than picking the first call.
    assert pair_verdict("A", "B") == "abstain"
    assert pair_verdict("tie", "A") == "abstain"
    assert pair_verdict("tie", "tie") == "tie"
    assert pair_verdict("A", "A") == "A"

    # A judge that always ties gives a flat group -- GRPO's advantage is then
    # zero for every member, which is the correct "no signal", not a ranking.
    def always_tie(a, b):
        return ("tie", "tie")

    flat, _ = judge_rewards(["a", "b", "c"], [True] * 3, always_tie)
    assert len(set(flat)) == 1, flat

    # A non-transitive cycle (a>b, b>c, c>a) leaves equal win counts, so it also
    # produces no signal instead of an invented order.
    cyc = {(0, 1): "A", (1, 2): "A", (0, 2): "B"}
    assert len(set(copeland_scores(3, cyc, _PASS_BAND))) == 1, copeland_scores(3, cyc, _PASS_BAND)

    # A clean sweep orders the band end to end.
    sweep = {(0, 1): "A", (0, 2): "A", (1, 2): "A"}
    assert copeland_scores(3, sweep, _PASS_BAND) == [1.0, 0.8, 0.6]

    # Tests alone separate a group: no judge call is made at all.
    def explode(a, b):
        raise AssertionError("judge must not see a pair tests can separate")

    ok, rows = judge_rewards(["p", "f"], [True, False], explode)
    assert ok == [0.8, 0.2] and rows == [], (ok, rows)

    # Every row carries its subgroup, which is what the scorer validates on.
    _, rows = judge_rewards(["a", "b"], [True, True], always_a, group_id="task7")
    assert rows == [{"pair_id": "task7:0v1", "order_ab": "A", "order_ba": "A",
                     "test_winner": None, "subgroup": "all_pass"}], rows
    print("judge self-check ok")
