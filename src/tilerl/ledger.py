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


def new_best_point(pt: dict, best: dict | None, se: float | None = None,
                   mode: str = "significant") -> bool:
    """Whether ``pt`` replaces the incumbent best curve point.

    SIGNIFICANTLY greater, not merely greater: the eval's NET-difference floor is
    0.2-0.4 pt (two readings: 1 and 2 questions in 500, 2026-09-08/09;
    errors/2026-09-09-the-dip-hit-problems-a-healthy-seed-solves.md), and a
    one-question lead has bought extra training for a reading inside the instrument.
    The gross-flip floor is a different number -- 15-16/500, see `curve_churn` -- and
    the two must not be mixed. A tie keeps the earlier point:
    `time_to_score` is the objective, so at equal score the cheaper point wins.

    ``se`` is the PAIRED width of ``pt - best`` in points, from `paired_se` over the
    per-problem rows. None falls back to the conservative unpaired width -- the caller
    must mark that result, because a conservative "not significantly greater" must not
    be read as "the two points are the same".

    ``mode="raw"`` compares raw scores (strict >) instead of the 2xSE ruler. It exists
    for small curve subsets, where the paired width is wider than the signal -- at
    n=100 the paired 2xSE is 5.6-7.4 pt against a +6.0 pt step gain -- so the
    significant ruler never fires and patience stops at the second point on every
    curve, a guard that always fires. Three risks, the whole of why raw is not the
    default:

    1. It can fire on a difference below the instrument floor -- the eval's
       net-difference floor is 0.2-0.4 pt (1-2 questions / 500). Seed 1 in raw mode
       stops at step 50 on 94.2 <= 94.4, one question.
    2. It is safe only on curves whose gain is concentrated in the first point -- a
       step, then flat -- where "stop at the first non-improving point" loses nothing.
       On a noisy rise it follows the noise: best drifts up on sub-floor gains, and
       the decline veto is then measured against a noise-inflated peak (cell F). A
       flat point stops BOTH modes under patience=1 -- the mode-specific hazard is
       the drift, not the stop.
    3. It is coupled to the adapter-best snapshot: stopping early loses only "might
       improve later", never what was -- the only reason raw is acceptable. Without
       the snapshot, raw is a net loss.
    4. It ships the expensive twin: `adapter-best` follows `best`, so a sub-floor gain
       does not just reset patience -- it makes the SNAPSHOT, the weights that get
       delivered, a one-question choice. p@25 = 94.0, p@50 = 94.2: significant calls
       it a tie and keeps step 25, raw keeps step 50 -- 25 more training steps for a
       pair statistics cannot separate. Raw knowingly waives the tie-keeps-the-
       earlier-point rule above, in exchange for a guard that still works at small n.
    """
    if best is None:
        return True
    if mode == "raw":
        return pt["score"] > best["score"]
    if se is None:
        se = unpaired_diff_se(pt, best)
    return pt["score"] - best["score"] > 2 * se / 100.0


def curve_churn(prev: list[dict] | None, cur: list[dict]) -> tuple[int, int] | None:
    """Per-question flips between two adjacent curve points, paired by row position:
    ``(right->wrong, wrong->right)``. The run's own noise floor, recorded per point,
    so a "these two points differ by N questions" claim has N's instrument beside it.

    Comparable ONLY within one run: ``curve_rows`` is sliced once outside the loop
    and ``per_problem`` is written in input order, so position pairs the same
    question at both points. Across runs the positions mean different questions --
    never subtract two runs' churn. None when there is no predecessor (the first
    point) or the rows do not pair (different lengths): null, not 0, because 0 means
    "no flips" and null means "no comparable point".

    The same-batch-composition GROSS-flip floor is 15-16 questions/500 (measured
    2026-09-09 on adjacent plateau points; errors/2026-09-09-the-dip-hit-problems-
    a-healthy-seed-solves.md): a churn at or below that is the instrument's own
    jitter, not a policy difference. The cross-batch-composition rate is a different
    caliber -- 52/500 = 10.4% with only the batch composition changed -- and the two
    must never be compared. Gross flip and net difference are also different
    calibers: the net-difference floor is 0.2-0.4 pt (see `new_best_point`).
    """
    if not prev or len(prev) != len(cur):
        return None
    rb = sum(1 for a, b in zip(prev, cur) if a["correct"] and not b["correct"])
    br = sum(1 for a, b in zip(prev, cur) if not a["correct"] and b["correct"])
    return rb, br


def significant_decline(pt: dict, best: dict | None, se: float | None) -> bool:
    """Whether ``pt`` is significantly BELOW ``best`` -- the 2xSE ruler of
    `new_best_point`, pointed down.

    A collapse is not a slow day: seed 0's step 75 was -11.00 pt (6.62 sigma), and
    under patience alone it would cost `patience` more points before the run reacted
    -- the one event this feature exists for. Stopping never loses anything, because
    the best snapshot is kept; and seed 0's recovery (412 -> 456) still ended below
    the peak (467), so waiting for it bought less than keeping the peak. Never fires
    without the paired width -- an unpaired guess at a decline is not a verdict.
    """
    return bool(best) and se is not None and best["score"] - pt["score"] > 2 * se / 100.0


class EarlyStop:
    """Patience over CURVE POINTS, not steps: the curve samples every `--eval-every`
    steps, so one patience unit is one eval interval, and the same patience means
    different things at eval-every 25 and 5.

    A point that does not significantly improve on the best adds one to the count; a
    new best resets it. Stop when the count reaches `patience`, or immediately on a
    `significant_decline` -- the collapse is the event this feature exists for, not a
    slow day to be patient with. `patience=0` never stops on either path; it is the
    default, and flipping it on needs the seed-1 verdict, not this code.
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.stale = 0
        self.reason: str | None = None

    def update(self, replaced: bool, declined: bool = False) -> str | None:
        """Record the point; return the stop reason (``"patience"``/``"decline"``) or
        None to continue. A decline is a veto: it spends no patience. ``patience=0``
        disables BOTH paths -- the default is fully off, decline veto included."""
        if self.reason is not None:
            return self.reason
        if not self.patience:
            return None
        if declined:
            self.reason = "decline"
            return "decline"
        self.stale = 0 if replaced else self.stale + 1
        if self.stale >= self.patience:
            self.reason = "patience"
            return "patience"
        return None


def require_paired_width(se: float | None, patience: int, kept_step: int | None = None) -> None:
    """Early stopping is a paired verdict: it compares each point to the best over the
    same rows. A missing width with ``patience > 0`` is a broken environment, not a
    fallback -- refusing loudly beats the two silent failures, a guard that never fires
    (unpaired width too wide to ever cross) and a stop decided on noise (no width at
    all). ``patience=0`` never asks, so it never refuses. ``kept_step`` names the best
    snapshot already on disk, so a reader meeting this exit mid-run knows it loses
    nothing -- the run is not wasted, the switch just cannot work on these records."""
    if patience and se is None:
        saved = (f" The best snapshot through step {kept_step} is already saved at "
                 "adapter-best.safetensors -- this exit loses nothing but the steps "
                 "a width-less stop would have decided on noise.") if kept_step is not None else ""
        raise SystemExit(
            "--patience needs the paired per-problem rows: the best point's "
            "eval-curve-<step>.jsonl is missing or does not join this point's rows, so "
            "no paired width exists and early stopping cannot fire. Fix the run's "
            "records; do not run to --steps behind a switch that cannot work." + saved)


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
    # Two points one question apart (0.2 pt at n=500): inside the 0.2-0.4 pt net floor,
    # the earlier one keeps it. This is the error the run actually made -- step 50 led
    # step 25 by exactly one question, and taking the numerically higher point bought
    # 501.2 s of extra training for a reading inside the instrument. Negative control:
    # with the criterion as plain `>` this assertion goes red.
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
    # Early stopping. Patience is in POINTS, not steps: one unit is one eval interval.
    # Cell A: a plateau from the start -- the fine curve's shape, no adjacent pair
    # crossing 2xSE. patience=1 stops at the second point and keeps the first.
    es = EarlyStop(1)
    assert [es.update(rep) for rep in (True, False, False, False)] == \
        [None, "patience", "patience", "patience"]
    # Cell B: plateau then rise. patience=3 lets the rise arrive before the stop;
    # patience=2 stops one point earlier. A stub that never refuses fails cell A; one
    # that always refuses fails this one -- both must be able to fail.
    es = EarlyStop(3)
    assert [es.update(rep) for rep in (True, False, False, True, False)] == \
        [None, None, None, None, None]
    es = EarlyStop(2)
    assert [es.update(rep) for rep in (True, False, False, True)] == \
        [None, None, "patience", "patience"]
    # patience=0 is the default and never stops -- not even on a decline, so the
    # default is fully off.
    assert not any(EarlyStop(0).update(rep) for rep in (False, False, False, False))
    assert EarlyStop(0).update(False, declined=True) is None
    # Cell C: a significant decline stops immediately, spending no patience, and keeps
    # the pre-decline best. seed 0's shape: plateau at 93.4, then -11.00 pt (6.62
    # sigma) at step 75 -- under patience=2 it would otherwise have trained two more
    # points (50 steps at --eval-every 25) past the collapse.
    es = EarlyStop(2)
    assert [es.update(rep, dec) for rep, dec in
            ((True, False), (False, False), (False, True))] == [None, None, "decline"]
    assert es.stale == 1  # the decline did not add to the patience count
    # The decline ruler itself: 11.0 pt against a 1.66 pt paired SE (6.62 sigma) is a
    # decline; 0.2 pt is inside the 0.2-0.4 pt net floor; and no paired width means no
    # verdict.
    peak = {"step": 50, "correct": 467, "total": 500, "score": 0.934}
    assert significant_decline({"score": 0.824}, peak, 1.66)
    assert not significant_decline({"score": 0.932}, peak, 1.66)
    assert not significant_decline({"score": 0.824}, peak, None)
    # Cell D: patience > 0 with no paired width REFUSES -- never the silent fallbacks
    # (a guard that never fires, or a stop decided on noise). patience=0 never asks.
    # With a kept step the message says the snapshot is already saved: a mid-run exit
    # must not read as a wasted run.
    try:
        require_paired_width(None, 2, kept_step=50)
        raise AssertionError("no raise")
    except SystemExit as exc:
        assert "paired" in str(exc) and "step 50" in str(exc)
    require_paired_width(1.31, 2)
    require_paired_width(None, 0)
    # Patience mode raw. The paired width at n=100 is 5.6-7.4 pt against a +6.0 pt
    # step gain, so the significant ruler never fires there; raw compares raw scores.
    # Cell E: a gradual rise at small n. Significant stops at the third point and
    # keeps the first -- the guard that always fires. Raw follows the rise and never
    # stops. This is why raw exists, and the proof the modes differ.
    rise = [{"step": s, "score": x} for s, x in ((5, .940), (10, .950), (15, .960), (20, .970))]
    es_sig, es_raw, best_sig, best_raw = EarlyStop(2), EarlyStop(2), None, None
    for p in rise:
        rep_s = new_best_point(p, best_sig, 6.0)
        rep_r = new_best_point(p, best_raw, 6.0, mode="raw")
        if rep_s:
            best_sig = p
        if rep_r:
            best_raw = p
        es_sig.update(rep_s)
        es_raw.update(rep_r)
    assert es_sig.reason == "patience" and best_sig["step"] == 5
    assert es_raw.reason is None and best_raw["step"] == 20
    # Cell F: raw's risk, pinned. A one-question gain at n=100 (1.0 pt, under the
    # 6.0 pt width) resets patience and drifts best up in raw; significant does
    # neither. The drift has two consequences. It feeds the decline veto: measured
    # against the drifted peak, a later point vetoes in raw where significant --
    # whose best never moved -- sees no decline. And it ships the expensive twin:
    # significant keeps the earlier point (the tie rule), raw the later one -- more
    # training for a pair statistics cannot separate. Same curve, different kept
    # point and different verdict.
    p1, p2 = {"step": 5, "score": .940}, {"step": 10, "score": .950}
    assert not new_best_point(p2, p1, 6.0)       # significant: tie, keeps step 5
    assert new_best_point(p2, p1, 6.0, mode="raw")  # raw: keeps step 10
    assert EarlyStop(1).update(False) == "patience"
    assert EarlyStop(1).update(True) is None
    p3 = {"step": 15, "score": .820}
    assert significant_decline(p3, p2, 6.0)       # vs the drifted raw peak: -13.0 pt
    assert not significant_decline(p3, p1, 6.0)   # vs the significant peak: -12.0 pt, not > 2xSE
    # Curve churn: the run's own noise floor, recorded per point. Adjacent points pair
    # by position within one run; across runs the positions mean different questions.
    # The first point has no predecessor: null, not 0 -- 0 means "no flips", null means
    # "no comparable point". Different lengths are null too, not a partial count.
    rows = [{"i": i, "correct": bool(c)} for i, c in enumerate((1, 1, 0, 0))]
    flipped = [{"i": i, "correct": bool(c)} for i, c in enumerate((1, 0, 0, 1))]
    assert curve_churn(rows, flipped) == (1, 1)  # one right->wrong, one wrong->right
    assert curve_churn(None, rows) is None       # first point: no predecessor
    assert curve_churn(rows, rows[:3]) is None   # different n: not comparable
    print("ledger: ids + best-point selection OK")
