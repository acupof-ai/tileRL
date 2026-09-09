#!/usr/bin/env python3
"""Classify a card's owner from aupai's card_assignment.json (read-only).

Prints ours / theirs / unknown. pod_run.sh refuses on theirs/unknown unless a
lend is recorded. Prefix rules per tilerl-27, 2026-09-09.
"""
import json
import re
import sys

#: Same rules as src/tilerl/engine.py — if you change these, change both.
OURS = re.compile(r"^\s*(tile[_-]?rl|rl[_-]?team)\b", re.IGNORECASE)
THEIRS = re.compile(r"^\s*(granted\b|\d{4}-\d{2}-\d{2})", re.IGNORECASE)


def main() -> int:
    card = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "/work/aupai/runs/card_assignment.json"
    with open(path) as f:
        note = json.load(f).get("cards", {}).get(str(card), "")
    if OURS.match(note):
        print("ours")
    elif THEIRS.match(note):
        print("theirs")
    else:
        print("unknown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
