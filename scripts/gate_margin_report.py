#!/usr/bin/env python3
"""Observational margin report: for each gate, how close did its operand get
to the threshold, across all recorded runs?

A gate whose operand never comes close to its threshold is decorative -- the
comparator is correct (test_gate_comparators.py proves that) but the gate
cannot object in production. This is the empty-gate detector that reads
values the system actually produced, not imagined failures.

Usage: python scripts/gate_margin_report.py [runs_dir]
  (default: runs/)
"""

import json
import sys
from pathlib import Path

# margin = distance from threshold in the PASSING direction.
# Positive = passed, negative = failed, zero = exactly at threshold.
# "gt": pass when value > threshold (margin = v - t)
# "lt": pass when value < threshold (margin = t - v)
_DIRECTION = {
    "reward_rises": "gt",
    "mmlu_holds": "gt",       # v >= t, margin = v - t
    "gsm8k_improves": "gt",   # v >= t, margin = v - t
    "groups_untied": "lt",
    "ce_falls": "lt",
    "rollouts_within_cap": "lt",  # v <= t, margin = t - v
}


def load_manifests(runs_dir: Path) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(runs_dir.glob("*/manifest.json"))]


def gate_margins(manifests: list[dict]) -> dict[str, list[float]]:
    """Per-gate list of margins (only scored, non-skipped, non-None values)."""
    out: dict[str, list[float]] = {}
    for m in manifests:
        for g in m.get("gates", []):
            name = g["name"]
            if g.get("skipped") or g.get("value") is None or g.get("threshold") is None:
                continue
            direction = _DIRECTION.get(name)
            if direction is None:
                continue
            v, t = g["value"], g["threshold"]
            margin = v - t if direction == "gt" else t - v
            out.setdefault(name, []).append(margin)
    return out


def report(manifests: list[dict]) -> str:
    margins = gate_margins(manifests)
    lines = [f"{'gate':<22} {'n':>3}  {'min':>8}  {'max':>8}  {'mean':>8}  note"]
    for name in sorted(_DIRECTION):
        vals = margins.get(name, [])
        if not vals:
            lines.append(f"{name:<22} {'0':>3}  {'—':>8}  {'—':>8}  {'—':>8}  never scored")
            continue
        mn, mx = min(vals), max(vals)
        mean = sum(vals) / len(vals)
        note = ""
        if mn <= 0:
            note = "← FAILED or at threshold"
        elif all(v > 0.1 * abs(mx) for v in vals):
            note = "← never close to red"
        lines.append(f"{name:<22} {len(vals):>3}  {mn:>8.4f}  {mx:>8.4f}  {mean:>8.4f}  {note}")
    return "\n".join(lines)


if __name__ == "__main__":
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    manifests = load_manifests(runs_dir)
    print(f"runs: {len(manifests)} from {runs_dir}")
    print(report(manifests))
