#!/usr/bin/env python3
"""Per-problem contingency tables over a run's eval-curve points.

Single-run: the 2^k table over the given steps, with marginals checked against
each point's score. Cross-run (--cross): the dip-set intersection between two
runs for a step pair, pairing by eval-file row.

Pairing across runs needs the eval file and each run's curve seed: pre-#329
runs took the file's first N rows in order (identity); post-#329 runs shuffle
a copy with --eval-curve-seed (recorded in the manifest's inputs) and take the
prefix. Row numbers do NOT align across runs -- only file-row indices do.

Usage:
  curve_table.py <run_dir> <step> [<step> ...]
  curve_table.py --cross <run0> <run1> <from_step> <to_step> \\
      --eval-file <path> [--curve-seed0 <int>] [--curve-seed1 <int>]
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def load_curve(run_dir: Path, step: int) -> list[dict]:
    rows = [json.loads(l) for l in (run_dir / f"eval-curve-{step}.jsonl").open()]
    return rows


def curve_seed(run_dir: Path) -> int | None:
    """None = pre-#329 identity ordering."""
    m = json.loads((run_dir / "manifest.json").read_text())
    return m.get("inputs", {}).get("eval_curve_seed")


def eval_n(run_dir: Path) -> int:
    m = json.loads((run_dir / "manifest.json").read_text())
    return m.get("inputs", {}).get("eval_n", 0)


def file_row_perm(run_dir: Path, n_rows: int) -> list[int]:
    """Curve position -> eval-file row index."""
    seed = curve_seed(run_dir)
    n = eval_n(run_dir) or n_rows
    if seed is None:
        return list(range(n_rows))
    p = list(range(n))
    random.Random(seed).shuffle(p)
    return p[:n_rows]


def assert_aligned(run_dir: Path, steps: list[int]) -> None:
    """Same problem at every position: gold answers identical across points."""
    curves = {s: load_curve(run_dir, s) for s in steps}
    n = len(curves[steps[0]])
    for s in steps:
        assert len(curves[s]) == n, f"step {s}: {len(curves[s])} rows != {n}"
    for i in range(n):
        answers = {curves[s][i]["answer"] for s in steps}
        assert len(answers) == 1, f"row {i}: gold answers differ across steps"


def single_table(run_dir: Path, steps: list[int]) -> None:
    assert_aligned(run_dir, steps)
    curves = {s: load_curve(run_dir, s) for s in steps}
    n = len(curves[steps[0]])
    cells = Counter(tuple(bool(curves[s][i]["correct"]) for s in steps) for i in range(n))
    print(f"run {run_dir.name}  steps {steps}  n={n}")
    for k in sorted(cells):
        print(f"  {' '.join(f'{s}={str(v)[0]}' for s, v in zip(steps, k))}: {cells[k]}")
    # marginals must match each point's own score
    for j, s in enumerate(steps):
        got = sum(c for k, c in cells.items() if k[j])
        want = sum(r["correct"] for r in curves[s])
        assert got == want, f"step {s}: marginal {got} != score {want}"
    print("marginals close")


def cross(run0: Path, run1: Path, from_step: int, to_step: int, eval_file: Path,
          seed0: int | None, seed1: int | None) -> None:
    gold = [json.loads(l)["answer"] for l in eval_file.open()]
    steps = [from_step, to_step]
    assert_aligned(run0, steps)
    assert_aligned(run1, steps)
    c0 = {s: load_curve(run0, s) for s in steps}
    c1 = {s: load_curve(run1, s) for s in steps}
    p0 = file_row_perm(run0, len(c0[from_step]))
    p1 = file_row_perm(run1, len(c1[from_step]))
    # explicit seeds override the manifests (the manifests are the default)
    if seed0 is not None:
        p0 = list(range(eval_n(run0) or len(p0))); random.Random(seed0).shuffle(p0); p0 = p0[:len(c0[from_step])]
    if seed1 is not None:
        p1 = list(range(eval_n(run1) or len(p1))); random.Random(seed1).shuffle(p1); p1 = p1[:len(c1[from_step])]
    # pairing check: gold answer at curve position must equal the file row's
    for pos, fr in enumerate(p0):
        assert c0[from_step][pos]["answer"] == gold[fr], f"run0 row {pos} != file row {fr}"
    for pos, fr in enumerate(p1):
        assert c1[from_step][pos]["answer"] == gold[fr], f"run1 row {pos} != file row {fr}"
    # map file row -> correctness per run per step
    m0 = {s: {fr: bool(c0[s][pos]["correct"]) for pos, fr in enumerate(p0)} for s in steps}
    m1 = {s: {fr: bool(c1[s][pos]["correct"]) for pos, fr in enumerate(p1)} for s in steps}
    common = sorted(set(m0[from_step]) & set(m1[from_step]))
    L0 = {fr for fr in common if m0[from_step][fr] and not m0[to_step][fr]}
    L1 = {fr for fr in common if m1[from_step][fr] and not m1[to_step][fr]}
    print(f"runs {run0.name} x {run1.name}  dip {from_step}->{to_step}  paired={len(common)}")
    print(f"  |L0| = {len(L0)}   |L1| = {len(L1)}")
    print(f"  |L0 ∩ L1| = {len(L0 & L1)}")
    print(f"  |L0 \\ L1| = {len(L0 - L1)}   |L1 \\ L0| = {len(L1 - L0)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("steps", nargs="*", type=int)
    ap.add_argument("--cross", nargs=4, metavar=("RUN0", "RUN1", "FROM", "TO"))
    ap.add_argument("--eval-file")
    ap.add_argument("--curve-seed0", type=int)
    ap.add_argument("--curve-seed1", type=int)
    args = ap.parse_args()
    if args.cross:
        r0, r1, f, t = args.cross
        assert args.eval_file, "--cross needs --eval-file"
        cross(Path(r0), Path(r1), int(f), int(t), Path(args.eval_file),
              args.curve_seed0, args.curve_seed1)
    else:
        assert args.run_dir and args.steps, "need <run_dir> <step> ..."
        single_table(Path(args.run_dir), args.steps)


if __name__ == "__main__":
    sys.exit(main())
