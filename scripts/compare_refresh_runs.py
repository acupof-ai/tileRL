#!/usr/bin/env python3
"""#805 quality comparator for async-refresh evaluation.

Compares two run directories of per-prompt output json (the ref_NNN.json shape
probe_serve_sm70_w2048 writes: {"output": [ids]}). Used for:

  - NOISE FLOOR: run the synchronous R=8 configuration twice (same prompts,
    temp 0, same seed), compare run A vs run B -> per-prompt agreement floor.
    sm70 temp 0 is not assumed bit-deterministic, so the delayed configuration
    is judged against this floor, never against 100%.
  - DELAYED vs SYNC: compare the 1-tick-delay run against one synchronous run;
    per-prompt agreement and first-divergence position feed the pre-registered
    gate in scripts/REFRESH_1TICK_DELAY_PLAN.md.

Metrics per prompt (n prompts must match 1:1 by index):
  agreement      matched prefix / min(len a, len b)
  first_diverge  index of first mismatching token (null = identical over
                 the shared length; reported relative to generated output,
                 i.e. 0 = the first generated token)
  mod8           first_diverge mod 8, to detect divergence clustering at the
                 refresh boundary (position 0 mod 8)

No engine, no CUDA: pure file analysis.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def _load(directory: str) -> dict[int, list[int]]:
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as f:
            obj = json.load(f)
        if "output" not in obj or not isinstance(obj["output"], list):
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            idx = int(base.rsplit("_", 1)[-1])
        except ValueError:
            continue
        out[idx] = [int(x) for x in obj["output"]]
    return out


def compare_pair(a: list[int], b: list[int]) -> dict:
    n = min(len(a), len(b))
    first = None
    matched = 0
    for i in range(n):
        if a[i] == b[i]:
            matched += 1
        elif first is None:
            first = i
    # Identical over the shared length but different total length: the
    # divergence is the first token one run has and the other lacks.
    if first is None and len(a) != len(b):
        first = n
    return {
        "len_a": len(a), "len_b": len(b),
        "agreement": round(matched / n, 6) if n else 1.0,
        "matched": matched, "shared_len": n,
        "first_diverge": first,
        "mod8": (first % 8) if first is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", help="run directory A (ref_NNN.json files)")
    ap.add_argument("dir_b", help="run directory B")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    a, b = _load(args.dir_a), _load(args.dir_b)
    common = sorted(set(a) & set(b))
    if not common:
        print(f"no matching prompt indices in {args.dir_a!r} vs {args.dir_b!r}",
              file=sys.stderr)
        return 14
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))

    per = {str(i): compare_pair(a[i], b[i]) for i in common}
    agr = [v["agreement"] for v in per.values()]
    diverged = [v for v in per.values() if v["first_diverge"] is not None]
    firsts = [v["first_diverge"] for v in diverged]
    mod8_zero = sum(1 for v in diverged if v["mod8"] == 0)

    summary = {
        "n_prompts_compared": len(common),
        "indices_only_in_a": only_a,
        "indices_only_in_b": only_b,
        "min_prompt_agreement": round(min(agr), 6),
        "mean_agreement": round(statistics.mean(agr), 6),
        "median_agreement": round(statistics.median(agr), 6),
        "n_diverged_prompts": len(diverged),
        "median_first_divergence": statistics.median(firsts) if firsts else None,
        "min_first_divergence": min(firsts) if firsts else None,
        # Refresh-boundary clustering: share of divergences at position 0 mod 8.
        "divergence_mod8_zero_fraction": round(
            mod8_zero / len(diverged), 4) if diverged else 0.0,
    }
    report = {"summary": summary, "per_prompt": per}
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(json.dumps(summary, indent=2))
    # Missing prompts make the comparison incomplete but not a value mismatch.
    return 0 if not only_a and not only_b else 14


if __name__ == "__main__":
    sys.exit(main())
