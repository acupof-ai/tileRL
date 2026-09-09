#!/usr/bin/env python3
"""Replay raw-score patience=1 over a run's curve points at several curve-n.

The curve subset is the same fixed-seed shuffle production uses
(`_curve_rows`, seed 0), so the replay matches what a run with
`--eval-curve-n N` would have scored on the same questions.

Pricing model: stop_step * --train-step + evals_run * --eval-full-sec * n/500.
Eval seconds scale linearly with rows; the model is an estimate, not a
measurement — the fine grid measured 1026 s/eval at n=500 against the 800 s
default.

Usage: curve_table.py <run_dir> [--train-step 20] [--eval-full-sec 800]
"""

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path


def load_rows(path):
    return [json.loads(l) for l in path.open()]


def curve_subset(n, seed=0):
    perm = list(range(500))
    random.Random(seed).shuffle(perm)
    return sorted(perm[:n])


def score_on(rows, idx):
    by_i = {r["i"]: r["correct"] for r in rows}
    return sum(by_i[i] for i in idx) / len(idx)


def paired_2se(a, b, idx):
    by_a = {r["i"]: r["correct"] for r in a}
    by_b = {r["i"]: r["correct"] for r in b}
    d = [int(by_a[i]) - int(by_b[i]) for i in idx]
    return 2 * statistics.stdev(d) / len(d) ** 0.5


def replay(before, points, idx, train_step, eval_full):
    """points: [(step, rows)] sorted. Raw patience=1, best=base, strict."""
    best = score_on(before, idx)
    evals = 0
    for step, rows in points:
        evals += 1
        sc = score_on(rows, idx)
        if sc > best:
            best = sc
        else:
            return step, best, evals, step * train_step + evals * eval_full * len(idx) / 500
    last = points[-1][0]
    return last, best, evals, last * train_step + evals * eval_full * len(idx) / 500


def self_check():
    # synthetic: base .50, step1 .60 (improve), step2 .60 (flat) -> stop@2
    def mk(score):
        return [{"i": i, "correct": i < round(score * 500)} for i in range(500)]

    before, pts = mk(0.50), [(1, mk(0.60)), (2, mk(0.60)), (3, mk(0.61))]
    idx = curve_subset(500)
    stop, best, evals, _ = replay(before, pts, idx, 20, 800)
    assert stop == 2 and evals == 2, (stop, best, evals)
    assert curve_subset(100) == curve_subset(100), "shuffle must be deterministic"
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--train-step", type=float, default=20.0)
    ap.add_argument("--eval-full-sec", type=float, default=800.0)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
    if not args.run_dir:
        sys.exit(0)

    d = Path(args.run_dir)
    before = [r for r in load_rows(d / "eval-before.jsonl") if r.get("dataset", "gsm8k") == "gsm8k"]
    steps = sorted(
        int(m.group(1))
        for p in d.glob("eval-curve-*.jsonl")
        if (m := re.match(r"eval-curve-(\d+)\.jsonl", p.name))
    )
    points = [(s, load_rows(d / f"eval-curve-{s}.jsonl")) for s in steps]

    print(f"{'n':>5} {'stop':>5} {'retained':>9} {'evals':>6} {'cost_s':>8}   2xSE per pair (pt)")
    for n in (500, 100, 50):
        idx = curve_subset(n)
        stop, best, evals, cost = replay(before, points, idx, args.train_step, args.eval_full_sec)
        ses = [
            f"{steps[k-1] if k else 'base'}->{steps[k]}: {100*paired_2se(points[k-1][1] if k else before, points[k][1], idx):.1f}"
            for k in range(len(points))
        ]
        print(f"{n:>5} {stop:>5} {best:>9.3f} {evals:>6} {cost:>8.0f}   " + " ".join(ses))
    print("retained = best-step full-subset score; confirm against the n=500 anchor before quoting")


if __name__ == "__main__":
    main()
