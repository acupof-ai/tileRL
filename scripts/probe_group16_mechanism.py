#!/usr/bin/env python3
"""Does drawing 16 samples instead of 8 explain group 16's higher idle?

Measured on card 3: at 17 matched prompts, group 8 pools to 74.4% idle and group 16 to
77.3% -- +2.9 points. The mechanism I asserted, before checking it, was that `max` over 16
draws exceeds `max` over 8 while `sum` merely doubles, so occupancy's denominator grows
faster than its numerator. That is a correct piece of reasoning and it was not a statement
about this data, which is the distinction that matters: nothing in the summary rows could
have refuted it.

This makes it refutable. Resampling groups of 16 from the group-8 run's individual row
lengths predicts what idle a 16-wide group should show, and the group-16 arm measured that
independently. **A model fitted on one arm, tested on an arm it never saw** -- so the
prediction can miss, and if it misses by enough the mechanism is not the explanation.

Two things it does not do. It assumes the two arms' rows are draws from one length
distribution, which is exactly what it would mean for batch width alone to cause the gap;
if the arms differ for another reason the prediction fails and that is the finding. And
resampling with replacement from 8 observations per prompt cannot invent a tail longer
than the longest row seen, so at the cap the prediction is biased low -- reported, not
corrected.

Reads /work/rollout_tail.json, written by probe_rollout_tail.py when its run completes.
No card.
"""
import argparse
import json
import random
import statistics
import sys


def occupancy(steps: list[list[int]], k: int) -> float:
    """Pooled Sigma sum / Sigma (max * k), the quantity `1 - idle` reports."""
    return sum(sum(s) for s in steps) / sum(max(s) * k for s in steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", default="/work/rollout_tail.json")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.tail) as f:
        data = json.load(f)
    by_group = {int(g): rs for g, rs in data["by_group"].items()}
    if 8 not in by_group or 16 not in by_group:
        raise SystemExit(f"need both group 8 and 16 in {args.tail}, have {sorted(by_group)}")

    g8 = [r["lengths"] for r in by_group[8]]
    g16 = [r["lengths"] for r in by_group[16]]
    n = min(len(g8), len(g16))
    g8, g16 = g8[:n], g16[:n]  # same prompts, in order
    obs8, obs16 = 1 - occupancy(g8, 8), 1 - occupancy(g16, 16)
    print(f"{n} matched prompts")
    print(f"  measured idle   group 8 {100 * obs8:.1f}%   group 16 {100 * obs16:.1f}%   "
          f"delta {100 * (obs16 - obs8):+.1f} pt")

    rng = random.Random(args.seed)
    preds = []
    for _ in range(args.trials):
        resampled = [[rng.choice(s) for _ in range(16)] for s in g8]
        preds.append(1 - occupancy(resampled, 16))
    lo, mid, hi = (statistics.quantiles(preds, n=40)[0], statistics.median(preds),
                   statistics.quantiles(preds, n=40)[38])
    print(f"  predicted idle at 16 from the group-8 rows: {100 * mid:.1f}% "
          f"[{100 * lo:.1f}, {100 * hi:.1f}] (95% over {args.trials} resamples)")
    print(f"  predicted delta {100 * (mid - obs8):+.1f} pt   measured {100 * (obs16 - obs8):+.1f} pt")

    capped = sum(1 for s in g8 for x in s if x >= data["gen_cap"])
    inside = lo <= obs16 <= hi
    print(f"\n{'CONFIRMED' if inside else 'REFUTED'}: the measured group-16 idle is "
          f"{'inside' if inside else 'outside'} the interval predicted by resampling the "
          f"group-8 rows, so wider sampling of the same length distribution "
          f"{'accounts for' if inside else 'does NOT account for'} the gap.")
    if capped:
        print(f"{capped} of {sum(len(s) for s in g8)} group-8 rows sat at the "
              f"{data['gen_cap']} cap; resampling cannot draw a row longer than the longest "
              "observed, so the prediction is biased low and a CONFIRMED verdict is the "
              "weaker of the two.")
    return 0


if __name__ == "__main__":
    # Two synthetic populations, before touching the real file. Same distribution: the
    # 16-wide idle must land in the predicted interval. Heavier tail at 16: it must not,
    # or the check would confirm any pair of arms handed to it.
    _r = random.Random(1)
    _same = [[_r.randrange(100, 2000) for _ in range(8)] for _ in range(20)]
    _pred = [1 - occupancy([[_r.choice(s) for _ in range(16)] for s in _same], 16)
             for _ in range(400)]
    _wide = [[_r.randrange(100, 2000) for _ in range(16)] for _ in range(20)]
    _lo, _hi = min(_pred), max(_pred)
    assert _lo <= 1 - occupancy(_wide, 16) <= _hi, "same distribution must confirm"
    _heavy = [[_r.randrange(100, 8000) for _ in range(16)] for _ in range(20)]
    assert not (_lo <= 1 - occupancy(_heavy, 16) <= _hi), "a heavier tail must refute"
    print("self-check: 2 asserts passed (a same-distribution arm confirms, "
          "a heavier-tailed one refutes)")
    # The analysis runs only when invoked with arguments. Bare `python3 <this>` is how
    # CI runs a script's self-check, and a CI host has no /work, so reaching main()
    # there failed the gate on a missing pod path. This is not a skip: the asserts above
    # ran and can fail. An explicit `--tail X` still runs main() and still raises if X
    # is absent, so a typo on the pod is not swallowed.
    if len(sys.argv) == 1:
        raise SystemExit(0)
    raise SystemExit(main())
