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
    """Curve position -> eval-file row index.

    Post-#329 runs reproduce the shuffle locally; CPython's shuffle is an
    implementation detail, so a --perm* map produced by the run's own
    ``_curve_rows`` on the pod overrides this.
    """
    seed = curve_seed(run_dir)
    n = eval_n(run_dir) or n_rows
    if seed is None:
        return list(range(n_rows))
    p = list(range(n))
    random.Random(seed).shuffle(p)
    return p[:n_rows]


def hypergeom_z(obs: int, x: int, y: int, n: int) -> tuple[float, float]:
    """Standard score of an overlap against the independent-draw null.

    Raw overlaps (and their ratios) are not comparable across steps: the
    ceiling moves with the set sizes (62/62 on 500 caps the ratio at 8.06x,
    below a healthy step's measured 9.45x). The z eats the marginals.
    Returns (E, z); var is E-adjacent so callers print both.
    """
    e = x * y / n
    var = y * (x / n) * (1 - x / n) * (n - y) / (n - 1)
    return e, (obs - e) / (var ** 0.5)


def print_overlap(name: str, obs: int, x: int, y: int, n: int) -> None:
    """The four numbers a reader needs to recompute the judgement: obs, E,
    z, and the ceiling (max). When max sits on obs the z is distorted too."""
    e, z = hypergeom_z(obs, x, y, n)
    print(f"  {name}: obs={obs}  E={e:.2f}  max={min(x, y)}  z={z:+.1f}")


def check_perm_against_gold(rows: list[dict], perm: list[int], gold: list[str],
                            tag: str) -> None:
    """Tripwire: a wrong permutation mismatches gold on most rows.

    NOT a proof of identity: GSM8K golds collide (195 distinct of 500), so a
    same-gold wrong row passes silently. The exact checks are the eval-file
    hash and a --perm* map from the run's own code.
    """
    bad = [pos for pos, fr in enumerate(perm) if rows[pos]["answer"] != gold[fr]]
    assert not bad, f"{tag}: {len(bad)} rows' gold disagrees with the permutation (first: {bad[:3]})"


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
          perm0: list[int] | None, perm1: list[int] | None) -> None:
    gold = [json.loads(l)["answer"] for l in eval_file.open()]
    print(f"gold distinct: {len(set(gold))} of {len(gold)} (collision rows are blind to the check below)")
    steps = [from_step, to_step]
    assert_aligned(run0, steps)
    assert_aligned(run1, steps)
    c0 = {s: load_curve(run0, s) for s in steps}
    c1 = {s: load_curve(run1, s) for s in steps}
    p0 = perm0 or file_row_perm(run0, len(c0[from_step]))
    p1 = perm1 or file_row_perm(run1, len(c1[from_step]))
    check_perm_against_gold(c0[from_step], p0, gold, "run0")
    check_perm_against_gold(c1[from_step], p1, gold, "run1")
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
    print_overlap("L0∩L1 vs difficulty null", len(L0 & L1), len(L0), len(L1), len(common))


def baseline(run0: Path, run1: Path, step: int, eval_file: Path,
             perm0: list[int] | None, perm1: list[int] | None) -> None:
    """The null hypothesis for a dip overlap: how much two HEALTHY policies'
    wrong sets overlap anyway (hard problems are hard for both). At a step
    before either curve dipped, the overlap is pure problem difficulty."""
    gold = [json.loads(l)["answer"] for l in eval_file.open()]
    print(f"gold distinct: {len(set(gold))} of {len(gold)} (collision rows are blind to the check below)")
    r0 = load_curve(run0, step)
    r1 = load_curve(run1, step)
    p0 = perm0 or file_row_perm(run0, len(r0))
    p1 = perm1 or file_row_perm(run1, len(r1))
    check_perm_against_gold(r0, p0, gold, "run0")
    check_perm_against_gold(r1, p1, gold, "run1")
    w0 = {p0[i] for i, r in enumerate(r0) if not r["correct"]}
    w1 = {p1[i] for i, r in enumerate(r1) if not r["correct"]}
    m0 = {p0[i]: bool(r["correct"]) for i, r in enumerate(r0)}
    m1 = {p1[i]: bool(r["correct"]) for i, r in enumerate(r1)}
    print(f"runs {run0.name} x {run1.name}  step {step}  paired={len(set(p0) & set(p1))}")
    print(f"  |A| (run0 wrong) = {len(w0)}   |B| (run1 wrong) = {len(w1)}")
    print(f"  |A ∩ B| = {len(w0 & w1)}   |A \\ B| = {len(w0 - w1)}   |B \\ A| = {len(w1 - w0)}")
    print_overlap("A∩B vs difficulty null", len(w0 & w1), len(w0), len(w1), len(set(p0) & set(p1)))
    # the net score difference as a swap, not a margin
    common = sorted(set(m0) & set(m1))
    r0_right_r1_wrong = sum(m0[fr] and not m1[fr] for fr in common)
    r1_right_r0_wrong = sum(m1[fr] and not m0[fr] for fr in common)
    s0, s1 = sum(r["correct"] for r in r0), sum(r["correct"] for r in r1)
    print(f"  scores {s0} vs {s1} (net {s1 - s0:+d}); "
          f"run0-right/run1-wrong = {r0_right_r1_wrong}, "
          f"run1-right/run0-wrong = {r1_right_r0_wrong}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("steps", nargs="*", type=int)
    ap.add_argument("--cross", nargs=4, metavar=("RUN0", "RUN1", "FROM", "TO"))
    ap.add_argument("--baseline", nargs=3, metavar=("RUN0", "RUN1", "STEP"))
    ap.add_argument("--eval-file")
    ap.add_argument("--perm0", type=Path, help="JSON list: run0 curve position -> eval-file row "
                        "(produced by the run's own _curve_rows; overrides local shuffle)")
    ap.add_argument("--perm1", type=Path)
    args = ap.parse_args()
    if args.cross:
        r0, r1, f, t = args.cross
        assert args.eval_file, "--cross needs --eval-file"
        p0 = json.loads(args.perm0.read_text()) if args.perm0 else None
        p1 = json.loads(args.perm1.read_text()) if args.perm1 else None
        cross(Path(r0), Path(r1), int(f), int(t), Path(args.eval_file), p0, p1)
    elif args.baseline:
        r0, r1, s = args.baseline
        assert args.eval_file, "--baseline needs --eval-file"
        p0 = json.loads(args.perm0.read_text()) if args.perm0 else None
        p1 = json.loads(args.perm1.read_text()) if args.perm1 else None
        baseline(Path(r0), Path(r1), int(s), Path(args.eval_file), p0, p1)
    else:
        assert args.run_dir and args.steps, "need <run_dir> <step> ..."
        single_table(Path(args.run_dir), args.steps)


if __name__ == "__main__":
    sys.exit(main())
