#!/usr/bin/env python3
"""Paired 2x2 for a run's before/after arms: kept / lost / fixed / untouched,
plus at_cap transitions and churn.

The arms score the eval file in order (no shuffle), so row index is identity.
Asserts gold alignment like floor_diff.py. Usage: paired_2x2.py <before.jsonl> <after.jsonl>
"""
import json
import sys

CAP = 6144


def load(path: str) -> list[dict]:
    return [r for r in (json.loads(l) for l in open(path)) if r.get("dataset", "gsm8k") == "gsm8k"]


def main() -> None:
    b, a = load(sys.argv[1]), load(sys.argv[2])
    assert len(b) == len(a), f"row count differs: {len(b)} vs {len(a)}"
    bad = [i for i, (x, y) in enumerate(zip(b, a)) if x["answer"] != y["answer"]]
    assert not bad, f"{len(bad)} rows pair on different golds (first: {bad[:3]})"

    kept = lost = fixed = untouched = 0
    for x, y in zip(b, a):
        if x["correct"] and y["correct"]:
            kept += 1
        elif x["correct"] and not y["correct"]:
            lost += 1
        elif not x["correct"] and y["correct"]:
            fixed += 1
        else:
            untouched += 1
    gross = lost + fixed
    print(f"2x2: kept {kept}  lost {lost}  fixed {fixed}  untouched {untouched}")
    print(f"score: {sum(r['correct'] for r in b)} -> {sum(r['correct'] for r in a)} "
          f"(net {sum(r['correct'] for r in a) - sum(r['correct'] for r in b):+d}, "
          f"gross flips {gross})")

    bcap = [i for i, r in enumerate(b) if r["tokens"] >= CAP]
    acap = [i for i, r in enumerate(a) if r["tokens"] >= CAP]
    freed = [i for i in bcap if a[i]["tokens"] < CAP]
    freed_correct = [i for i in freed if a[i]["correct"]]
    print(f"at_cap: before {len(bcap)} -> after {len(acap)}")
    print(f"  before-truncated now finished: {len(freed)} (of which correct: {len(freed_correct)})")
    print(f"  new truncations at after: {len([i for i in acap if b[i]['tokens'] < CAP])}")
    print(f"  still truncated: {len(set(bcap) & set(acap))}")
    mb = sum(r["tokens"] for r in b) / len(b)
    ma = sum(r["tokens"] for r in a) / len(a)
    print(f"mean tokens: {mb:.1f} -> {ma:.1f} ({ma / mb:.2f}x)")


if __name__ == "__main__":
    main()
