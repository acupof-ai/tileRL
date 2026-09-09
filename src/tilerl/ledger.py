"""The run ledger: one ``manifest.json`` per run under ``$TILERL_RUNS``
(default ``./runs``). ``id = hash(inputs)``, so a rerun is a no-op and a changed
input is a new run. Gates are data here and exit codes in the CLI. Stdlib only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def runs_root() -> Path:
    return Path(os.environ.get("TILERL_RUNS", "runs"))


def run_id(inputs: dict) -> str:
    """First 12 hex of sha256 over canonical JSON: key order does not matter."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]


def file_hash(path: str | os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def new_manifest(command: str, inputs: dict, parents: list[str] | None = None) -> dict:
    return {"id": run_id(inputs), "command": command, "inputs": inputs,
            "parents": list(parents or ()), "commit": inputs.get("commit", commit()),
            "started": now(),
            "finished": None, "metrics": {}, "gates": [], "artifacts": {}}


def write_manifest(root: str | os.PathLike, m: dict) -> Path:
    d = Path(root) / m["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    return d


def read_manifest(root: str | os.PathLike, id: str) -> dict | None:
    p = Path(root) / id / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def list_runs(root: str | os.PathLike) -> list[dict]:
    """Newest first."""
    ms = [json.loads(p.read_text()) for p in Path(root).glob("*/manifest.json")]
    return sorted(ms, key=lambda m: m["finished"] or m["started"], reverse=True)


def lineage(root: str | os.PathLike, id: str) -> list[dict]:
    """The run, then its parents, breadth first."""
    out: list[dict] = []
    todo = [id]
    while todo:
        m = read_manifest(root, todo.pop(0))
        if m and all(m["id"] != x["id"] for x in out):
            out.append(m)
            todo += m["parents"]
    return out


def gates_pass(m: dict) -> bool:
    """Every gate, both classes. UNCHANGED, deliberately: this is the process exit code.

    The verdict/validity split is recorded on each gate (`kind`) and read by
    `verdict_of`, not enforced here -- a validity failure still exits non-zero, because
    an uninterpretable run is not a success either. What the split fixes is the
    CONFLATION: `all(...)` over a flat list let a validity gate PASSING contribute to
    "P1 passed", and `reward_rises` must never be able to do that -- reward is the
    quantity GRPO optimizes, so it rising is the optimizer working, not evidence that RL
    moved a downstream number.
    """
    return all(g.get("skipped", False) or g["passed"] for g in m["gates"])


def verdict_of(m: dict, kind: str = "verdict") -> bool | None:
    """Did the gates of one class pass? None when that class has none that were scored.

    None is a third state and the caller must not collapse it to False: a run whose
    verdict gates were all skipped has not failed P1, it has not tested P1.
    """
    scored = [g for g in m["gates"]
              if g.get("kind", "verdict") == kind and not g.get("skipped", False)]
    return all(g["passed"] for g in scored) if scored else None


def curve_point_se(pt: dict, n: int = 0) -> float | None:
    """Binomial SE of one curve point, in POINTS, at the point's own rate.

    The width of ONE point against a constant target -- what `time_to_score` answers.
    Comparing two points is a different question: `paired_se` over the per-problem rows,
    or `unpaired_diff_se` when the rows are absent.
    """
    total = pt.get("total") or n
    if not total:
        return None
    p = (pt.get("correct") or 0) / total
    return round(100.0 * (p * (1 - p) / total) ** 0.5, 2)


def paired_se(rows_a: list[dict], rows_b: list[dict], key: str = "i") -> float | None:
    """Paired SE (points) of the difference between two points scored on the SAME rows:
    ``100 x sqrt(b + c) / n`` over the discordant pairs. None when the rows do not join.

    Curve points are a paired quantity -- every point scores the same subset -- so the
    width of a difference between two points is this, not a binomial width: measured
    2026-09-08, 1.9x narrower than the two-arm unpaired one at an 8.6% discordant rate.
    ``b + c == 0`` gives 0.0: the points agreed on every row, so there is no noise to
    measure and the difference is exactly 0. No `dataset` filter: one side is the live
    in-memory rows, which never carry that key.
    """
    ja = {r[key]: bool(r["correct"]) for r in rows_a if key in r}
    jb = {r[key]: bool(r["correct"]) for r in rows_b if key in r}
    if not ja or ja.keys() != jb.keys():
        return None
    disc = sum(ja[k] != jb[k] for k in ja)
    return round(100.0 * (disc / len(ja) ** 2) ** 0.5, 2)


def unpaired_diff_se(pt_a: dict, pt_b: dict) -> float:
    """Two-arm unpaired SE (points) of a score difference -- the CONSERVATIVE fallback
    when the per-problem rows that would make it paired are absent. Wider than the paired
    width by construction, so a caller that used it must mark the result as conservative."""
    na, nb = pt_a["total"], pt_b["total"]
    pa, pb = pt_a["correct"] / na, pt_b["correct"] / nb
    return round(100.0 * (pa * (1 - pa) / na + pb * (1 - pb) / nb) ** 0.5, 2)


def new_best_point(pt: dict, best: dict | None, se: float | None = None) -> bool:
    """Whether ``pt`` replaces the incumbent best curve point.

    SIGNIFICANTLY greater, not merely greater: the eval's own floor is 0.2 pt
    (measured 2026-09-08 -- one fixed set of weights, re-scored across processes,
    moved one question in 500), and a one-question lead has bought extra training
    for a reading inside the instrument. A tie keeps the earlier point:
    `time_to_score` is the objective, so at equal score the cheaper point wins.

    ``se`` is the PAIRED width of ``pt - best`` in points, from `paired_se` over the
    per-problem rows. None falls back to the conservative unpaired width -- the caller
    must mark that result, because a conservative "not significantly greater" must not
    be read as "the two points are the same".
    """
    if best is None:
        return True
    if se is None:
        se = unpaired_diff_se(pt, best)
    return pt["score"] - best["score"] > 2 * se / 100.0


def time_to_score(m: dict, target: float) -> dict | None:
    """When this run first scored >= ``target``, as a MEASUREMENT not a fit.

    The objective is ``time_to_score = steps_to_score x seconds_per_step``, and the
    curve is the only record that carries the step. Returns None when the run has no
    curve at all -- distinct from a curve that never reached the target, which returns
    ``reached=False``, because "not instrumented" and "instrumented and did not get
    there" are different facts about a run.

    The target usually falls BETWEEN two scoring points, so the answer is the point
    that crossed it plus the interval it was crossed in: ``step 50``, ``after 40``. No
    interpolation. An interpolated step is a number nobody measured, and this one is
    the project's headline metric -- a fitted headline is the failure mode the whole
    curve exists to avoid.
    """
    curve = m.get("eval_curve")
    if not curve or not curve.get("points"):
        return None
    # The curve scores a SUBSET, so its score is a different quantity from the run's
    # `gsm8k_after` over `--eval-n` rows -- and at small n the crossing step is set by
    # sampling as much as by the policy: n=20 resolves 5 pt per cell and carries a
    # binomial SE of 11.1 pt at that subset's own rate, against P1's +5 pt target. `n`
    # and `se_pt` travel with the answer so a caller cannot read the step without the
    # width. (tilerl-0a named the resolution; the SE is the operand that makes it
    # decisive.)
    #
    # `p(1-p)` at the POINT's own rate, not the 0.25 of p=0.5. The rate is in the point
    # and p=0.5 is its maximum, so the hardcode overstated the width -- 2.0x at the
    # measured 0.932, which fires `_se_note` at n=50 and n=100 on subsets that do
    # resolve the effect. The old test could not see it: its fixtures score 0.45 and
    # 0.60, where p(1-p) is flat and the constant is right to 2%.
    # This is the width of ONE point against a constant target, which is what this
    # function answers. Comparing two POINTS is a different question and a wider
    # interval (x sqrt(2) unpaired, or McNemar over `eval-curve-<step>.jsonl`).
    n = curve.get("n") or (curve["points"][0].get("total") or 0)

    def _se(pt: dict) -> float | None:
        return curve_point_se(pt, n)

    prev = 0
    for i, pt in enumerate(curve["points"]):
        if pt["score"] >= target:
            # `held` says whether every LATER point stayed at or above the target, so a
            # transient crossing is visible instead of being reported as arrival. Not a
            # precondition on `reached`: requiring it would turn one noisy dip into
            # "never reached" -- at X=0.91 a 90.8 point is 0.2 pt low against a 1.29 pt
            # SE, 0.15 sigma -- and that is a false negative on a number later runs are
            # priced against. Both facts, and the reader decides. `dipped_at` names the
            # first offender so the check does not need the caller to re-scan.
            later = curve["points"][i + 1:]
            below = [q["step"] for q in later if q["score"] < target]
            return {"reached": True, "target": target, "n": n, "se_pt": _se(pt),
                    "step": pt["step"], "after_step": prev,
                    "secs": pt["secs"], "score": pt["score"],
                    "correct": pt["correct"], "total": pt["total"],
                    "held": not below, "dipped_at": below[0] if below else None}
        prev = pt["step"]
    last = curve["points"][-1]
    # The width of the BEST point, since `best` is the number a reader compares to the
    # target -- not the last point's, which can be a lower score with a different width.
    best = max(curve["points"], key=lambda pt: pt["score"])
    return {"reached": False, "target": target, "n": n, "se_pt": _se(best),
            "steps_run": last["step"], "secs": last["secs"],
            "best": best["score"]}


def format_run(m: dict) -> str:
    mt = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                  for k, v in m["metrics"].items() if v is not None)
    # `finished` first: gates are written by `_finish`, so an interrupted run carries
    # only whatever was pre-seeded -- an empty list for opd, which `gates_pass` reads
    # as True, so a run killed mid-training printed `pass`. Measured on cpu: SIGTERM
    # after the manifest write, `e069c8ff28b7 train running pass`. No gate was
    # evaluated, so the only honest verdict is that the run did not reach one.
    if not m["finished"]:
        verdict = "killed"
    elif not m["gates"]:
        # A finished run with no gates DEFINED, which is not a pass: `gates_pass([])` is
        # `all([])` = True, so every `tilerl merge` row read `pass` over zero checks.
        # Measured on cpu with a real merge: `144b31c31f4d merge <ts> pass tensors=1`,
        # manifest `gates: []`. `none` rather than `skip`, which in this tree means a gate
        # existed and was suppressed (`gates_skip_after`, the drift gate under
        # --allow-short-rollouts) -- merge defines none, so the two states stay distinct.
        verdict = "none"
    elif all(g.get("skipped", False) for g in m["gates"]):
        verdict = "skip"
    else:
        verdict = "pass" if gates_pass(m) else "FAIL"
    # Annotated ONLY when the two classes disagree, which is the case one word cannot
    # say: `FAIL` while the verdict gates passed means a validity gate stopped the run
    # from being interpretable, not that P1 failed -- `docs/roadmap.md:57-58` already
    # draws that line ("else the task is too easy ... and the run says nothing"). When
    # they agree the string is unchanged, so every existing reader of field 3 still works.
    if verdict == "FAIL" and verdict_of(m, "verdict") is True:
        verdict = "novalid"
    return f"{m['id']}  {m['command']:<6} {m['finished'] or 'running':<25} {verdict:<7} {mt}"


if __name__ == "__main__":  # runnable check
    assert run_id({"a": 1, "b": [2]}) == run_id({"b": [2], "a": 1})
    assert run_id({"a": 1}) != run_id({"a": 2})
    # Best-point selection on the run's FINE curve (n=500): 94.2/94.2/94.6/92.8/93.2/
    # 82.4/91.2. The run shipped the last point (91.2); the snapshot must keep a top one.
    # It keeps step 5, not the numerical peak at step 15: the peak led by 0.4 pt, inside
    # the 2.0 pt floor, so the earliest top point wins. Negative control: with the rule
    # replaced by "every point wins" (take the last) this assertion goes red -- the last
    # point is exactly the 91.2 the run shipped without a snapshot.
    pts = [{"step": s, "correct": c, "total": 500, "score": c / 500}
           for s, c in ((5, 471), (10, 471), (15, 473), (25, 464),
                        (50, 466), (75, 412), (100, 456))]
    best = None
    for pt in pts:
        if new_best_point(pt, best):
            best = pt
    assert best["step"] == 5 and best["score"] == 0.942, best
    # Two points one question apart (0.2 pt at n=500): inside the floor, the earlier one
    # keeps it. This is the error the run actually made -- step 50 led step 25 by exactly
    # one question, and taking the numerically higher point bought 501.2 s of extra
    # training for a reading inside the instrument. Negative control: with the criterion
    # as plain `>` this assertion goes red.
    a = {"step": 25, "correct": 470, "total": 500, "score": 0.94}
    b = {"step": 50, "correct": 471, "total": 500, "score": 0.942}
    assert new_best_point(a, None)
    assert not new_best_point(b, a), (a["score"], b["score"])
    # Cell 3 proves WHICH SE the criterion uses -- cells 1-2 cannot, both widths give
    # the same answer on them. Two points 3.0 pt apart (435 -> 450 of 500) with 43
    # discordant pairs: paired SE = 100 x sqrt(43)/500 = 1.31 pt, so 2x = 2.6 < 3.0 and
    # the point replaces; the unpaired two-arm width is 2.02 pt, so 2x = 4.0 > 3.0 and
    # it does not. A criterion that silently used the unpaired width goes red here.
    va = [{"i": i, "correct": i < 435} for i in range(500)]
    vb = [dict(r) for r in va]
    for i in range(14):  # was right, now wrong
        vb[i]["correct"] = False
    for i in range(435, 464):  # was wrong, now right (29)
        vb[i]["correct"] = True
    assert paired_se(va, vb) == 1.31
    a3 = {"step": 25, "correct": 435, "total": 500, "score": 0.87}
    b3 = {"step": 50, "correct": 450, "total": 500, "score": 0.90}
    assert new_best_point(b3, a3, paired_se(va, vb))
    assert not new_best_point(b3, a3)
    print("ledger: ids + best-point selection OK")
