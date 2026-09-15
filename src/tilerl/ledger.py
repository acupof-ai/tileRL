"""The run ledger: one ``manifest.json`` per run under ``$TILERL_RUNS``
(default ``./runs``). ``id = hash(inputs)``, so a rerun is a no-op and a changed
input is a new run. Gates are data here and exit codes in the CLI. Stdlib only."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import sys
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


def find_run_for_artifact(root: str | os.PathLike, path: str | os.PathLike) -> str | None:
    """The run id whose manifest artifact resolves to ``path`` (file or dir), or None.

    A merge input specialist or a ``--load-adapter`` file carries no run id itself —
    only the producing run's manifest names it in ``artifacts``. Relative artifact
    values (``adapter.safetensors``) resolve against the run directory; absolute ones
    (a merge ``out``) are used as given. An unlinked path contributes no parent.
    """
    target = Path(path).resolve()
    for m in list_runs(root):
        rd = Path(root) / m["id"]
        for v in m.get("artifacts", {}).values():
            ap = Path(v)
            if not ap.is_absolute():
                ap = rd / ap
            if ap.resolve() == target:
                return m["id"]
    return None


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

    A gate with passed=None (not measured) and skipped=False is falsy here: the exit
    code is the safety net, and a run that produced no measurement is not a success.
    `verdict_of` reads the same None differently -- as "not tested", not "failed" --
    because the exit code and the verdict answer different questions.
    """
    return all(g.get("skipped", False) or g["passed"] for g in m["gates"])


def verdict_of(m: dict, kind: str = "verdict") -> bool | None:
    """Did the gates of one class pass? None when that class has none that were scored.

    None is a third state and the caller must not collapse it to False: a run whose
    verdict gates were all skipped has not failed P1, it has not tested P1.
    A gate with passed=None (not measured) is not scored: only gates with a real
    True/False contribute to the verdict. This is the other side of `gates_pass`,
    which treats the same None as falsy for the exit code -- the exit code is the
    safety net (fail-loud), the verdict is the interpretation (not-tested ≠ failed).
    """
    scored = [g for g in m["gates"]
              if g.get("kind", "verdict") == kind and not g.get("skipped", False)
              and g.get("passed") is not None]
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
    It is a SAMPLING width -- would the difference survive a different set of problems --
    not an instrument width: under fixed weights the same-batch instrument is exact (see
    `curve_churn`). ``b + c == 0`` gives 0.0: the points agreed on every row, so the
    difference is exactly 0 and no sampling width exists to estimate. No `dataset`
    filter: one side is the live in-memory rows, which never carry that key.
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

    SIGNIFICANTLY greater, not merely greater -- two independent judgments. The 2xSE
    ruler is a SAMPLING question: would the gain survive a different set of problems?
    It is not an instrument question -- the same-batch instrument is exact (fixed
    weights, same rows and order, bit-identical on re-score; GSM8K 500, 2026-09-09),
    so a one-question lead is a real gain, not jitter. Whether a real gain is worth
    more training is a second, separate judgment: `time_to_score` is the objective,
    and a tie keeps the earlier point because at equal score the cheaper point wins.

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


def curve_churn(prev: list[dict] | None, cur: list[dict]) -> tuple[int, int] | None:
    """Per-question flips between two adjacent curve points, paired by the ``i`` key:
    ``(right->wrong, wrong->right)``. The run's own instrument reading, recorded per
    point, so a "these two points differ by N questions" claim has N's measurement
    beside it. The same-batch instrument floor is 0 -- fixed weights re-scored on the
    same rows in the same order are bit-identical (GSM8K 500, 2026-09-09) -- so every
    flip between two points of one run is a real policy change, not jitter. The only
    operating floor is CROSS-batch: 52 flips / 500 = 10.4%, and it applies only when
    the two evals batched the problems differently (an after-arm against a curve
    point, or two runs) -- never quote it inside one run.

    Comparable ONLY within one run: ``curve_rows`` is sliced once outside the loop,
    so the same ``i`` is the same question at both points. (Rows land in COMPLETION
    order since the eval arm writes incrementally -- that is why pairing is by ``i``,
    not position.) Across runs the keys mean different questions -- never subtract
    two runs' churn. None when there is no predecessor (the first point), the rows
    do not pair (different lengths or ``i`` sets), or a row lacks ``i``: null, not 0,
    because 0 means "no flips" and null means "no comparable point".
    """
    if not prev or len(prev) != len(cur) or any("i" not in r for r in prev + cur):
        return None
    pb = {r["i"]: r for r in prev}
    cb = {r["i"]: r for r in cur}
    if pb.keys() != cb.keys():
        return None
    rb = sum(1 for k in pb if pb[k]["correct"] and not cb[k]["correct"])
    br = sum(1 for k in pb if not pb[k]["correct"] and cb[k]["correct"])
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
    (unpaired width too wide to ever cross) and a stop decided without a sampling width
    at all). ``patience=0`` never asks, so it never refuses. ``kept_step`` names the best
    snapshot already on disk, so a reader meeting this exit mid-run knows it loses
    nothing -- the run is not wasted, the switch just cannot work on these records."""
    if patience and se is None:
        saved = (f" The best snapshot through step {kept_step} is already saved at "
                 "adapter-best.safetensors -- this exit loses nothing but the steps "
                 "a width-less stop would have spent on a difference it could not place.") if kept_step is not None else ""
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



@functools.lru_cache(maxsize=1)
def _benchrec():
    """The ruler's validator/store. Packaged as tilerl.benchrec; the
    scripts/benchrec.py file is only a compatibility shim now, so a wheel
    install (no scripts/ directory) works."""
    from . import benchrec

    return benchrec


def residency_row(
    device_name: str,
    card: int | None,
    peak_bytes: int,
    static_bytes: int,
    transient_bytes: int,
    target: str,
    model: str,
    build: str = "eager",
    uuid: str | None = None,
) -> dict:
    """One measured row: steady-state device residency with its static/transient
    split, so occupancy lives in the same measurements.jsonl as the kernel
    roofline (peak = static + transient). A card-less sm* row is benchrec-
    rejected, so the CLI refuses --record-residency off cuda."""
    br = _benchrec()
    device = {"name": device_name, "card": card}
    if uuid is not None:
        device["uuid"] = uuid
    return {
        "metric": "device_resident_bytes",
        "value": int(peak_bytes),
        "unit": "bytes",
        "target": target,
        "build": build,
        "model": model,
        "shape": {
            "card": card,
            "static": int(static_bytes),
            "transient": int(transient_bytes),
        },
        "warm": {"state": "warm", "compiles": None},
        "n": 1,
        "spread": 0,
        "device": device,
        "commit": br.git_commit(),
        "dirty": br.git_dirty(),
        "cmd": "tilerl serve --dry-run --record-residency",
        "floor": {
            "value": int(peak_bytes),
            "unit": "bytes",
            "kind": "reference",
            "derivation": f"measured resident peak = static {int(static_bytes)} + "
            f"transient {int(transient_bytes)}",
        },
    }


def append_residency(row: dict, path: str | None = None) -> str:
    """Validate + append through benchrec, the one schema-writer."""
    br = _benchrec()
    old = br.STORE
    if path is not None:
        br.STORE = Path(path)
    try:
        return br.append(row)
    finally:
        br.STORE = old


def _paired_delta(run_dir: Path) -> dict | None:
    """`_mcnemar` over the two arms' `eval-{before,after}.jsonl`, or None if either is absent.

    Reads the record rather than in-memory state so it works on the cache-hit path too:
    a cached before-arm goes through `_write_eval_rows` like a fresh one, so the file
    exists either way.
    """
    def rows(tag):
        f = run_dir / f"eval-{tag}.jsonl"
        if not f.is_file():
            return None
        return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]

    before, after = rows("before"), rows("after")
    if before is None or after is None:
        return None
    from .train import _mcnemar  # lazy: train imports ledger inside its orchestration fns

    return _mcnemar(before, after)


def _timing_snapshot(m: dict) -> None:
    """Compare this run's speed against the SOTA baseline row and record the verdict.

    steps/SECOND, not seconds/step: every row in bench-baseline.json is higher-is-better
    and the gate's three comparisons are all `>`, so raw seconds would make a SLOWER run
    read as a new record (tests/test_bench_gate.py holds that).

    Best-effort: a run's result is the manifest, and a missing bench harness must not
    fail the run that produced it.
    """
    import importlib.util


    secs = (m.get("metrics") or {}).get("secs_per_step_median")
    if not secs:
        return
    hp = Path(__file__).resolve().parents[2] / "scripts" / "bench_harness.py"
    spec = importlib.util.spec_from_file_location("bench_harness", hp)
    if spec is None or spec.loader is None:
        return
    try:
        bh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bh)
        i = m["inputs"]
        shape = f"{i['model']}-{i['algo']}-g{i.get('group')}-t{i.get('max_new_tokens')}"
        # seed_only=False AND never dirty: seeding writes the tracked json, and every
        # pytest run of grpo-tiny-smoke would seed a row -- five junk CPU rows landed in
        # it the first time this ran. A run reports against the baseline; it never
        # edits it. Adding a key stays a deliberate act.
        gate = bh.Gate(os.environ.get("TILERL_TARGET", "cpu"))
        gate.check("train-run", shape, 1.0 / secs, unit="step/s")
        gate.dirty = False
        gate.finish(Path(runs_root()) / m["id"] / "baseline-candidate.json")
    except Exception as exc:  # noqa: BLE001 - the manifest is already written
        print(f"  (timing snapshot skipped: {exc})")


#: Gates whose value comes from the after-arm. A guard stop skips that arm, so these
#: are "not measured" rather than passed -- `_finish` scores a None value as True.
_AFTER_GATES = frozenset({"mmlu_holds", "gsm8k_improves"})


#: The two classes a gate can belong to, recorded on the gate itself.
#:
#: VERDICT answers "did P1 pass": the two exit criteria `docs/roadmap.md:57-58` states.
#: VALIDITY answers "is this run interpretable at all", and the roadmap already draws
#: that line -- the tied-group criterion reads "< 50% (else the task is too easy for
#: this model and the run says nothing)". Saying nothing is not failing.
#:
#: `reward_rises` is the reason this split matters. Reward is the quantity GRPO
#: optimizes, so it rising is the definition of the optimizer working, not evidence for
#: P1's claim that RL moves a DOWNSTREAM number -- and rising reward is fully
#: compatible with a falling eval, which is what reward hacking looks like. So it must
#: never be able to make P1 read `pass`. Reward NOT rising is informative (run 2
#: collapsed that way), and that failure is coarse enough for a zero threshold to
#: catch, which is why this needs no invented number.
_VALIDITY_GATES = frozenset({"groups_untied", "reward_rises", "ce_falls",
                             "rollouts_within_cap"})


def finish_run(m: dict, as_json: bool) -> None:
    """Gate, write the manifest, print it, exit non-zero on a failed gate.
    A gate whose metric was not evaluated reports passed=None (not measured)."""

    if not m["finished"]:
        g = m["metrics"]
        # .get, not [...]: a metric set that never had the key (an SFT run's
        # manifest) reads as None, which the gate below records as not-measured.
        # UNITS, and they differ 13 lines apart in the writer: `mmlu_{tag}` is a
        # FRACTION (`c / n`, :575) and `gsm8k_{tag}` is a COUNT (`c`, :588), with the
        # denominator alongside it as `gsm8k_{tag}_total` (:590). So the roadmap's two
        # exit numbers encode differently, and a threshold is meaningless without the
        # units of the quantity it thresholds.
        mmlu_floor = None if g.get("mmlu_before") is None else g["mmlu_before"] - 0.02
        # roadmap P1: "GSM8K held-out (500 q) after - before >= +5 pt (SE ~ 2 pt)". The
        # +5 is a sampling margin, not a taste -- the same-batch instrument is exact, so
        # `after > before` is a real +0.2 pt, but +1 question of 500 is a gain a
        # symmetric null passes about half the time on a different set. Derived from
        # `_total`, never hardcoded to 25: `--eval-n` is a flag and the recipe's 500 is
        # not a constant.
        gsm_total = g.get("gsm8k_after_total") or g.get("gsm8k_before_total")
        gsm_floor = (None if g.get("gsm8k_before") is None or not gsm_total
                     else g["gsm8k_before"] + 0.05 * gsm_total)
        # The paired test, RECORDED beside the threshold rather than replacing it. The
        # threshold is the roadmap's exit criterion and stays the gate; McNemar says
        # whether the observed move is resolvable at all, which the threshold cannot --
        # an unpaired read of n=500 has an 80%-power MDE of 7.70 pt, above the +5 pt the
        # gate asks for. Falls back silently to threshold-only when the per-question rows
        # are absent, which is what P1 did once already for want of them.
        paired = _paired_delta(Path(runs_root()) / m["id"])
        if paired is not None:
            m["metrics"]["gsm8k_paired"] = paired
        skipped = m["inputs"].get("steps") == 0
        after_skipped = bool(m.pop("gates_skip_after", False))
        # `ce_falls` has no threshold on the RL path: `ce_first` is written only by the
        # SFT loop (:281), never by the GRPO branch (:679-690), so the vacuous-pass rule
        # below made it report `passed` over nothing on every RL run. Not measured is the
        # honest record, and the gate stays live where the SFT path does write both.
        unmeasured = frozenset() if g.get("ce_first") is not None else frozenset({"ce_falls"})
        # Symmetric: RL gates have no metrics on the SFT path.
        if g.get("reward_first") is None:
            unmeasured |= frozenset({"reward_rises", "groups_untied"})
        m["gates"] += [
            {"name": n, "value": v, "threshold": t,
             "kind": "validity" if n in _VALIDITY_GATES else "verdict",
             "skipped": skipped or n in unmeasured or (after_skipped and n in _AFTER_GATES),
             "passed": None if skipped or n in unmeasured
             or (after_skipped and n in _AFTER_GATES)
             or v is None or t is None
             else ok(v, t)}
            for n, v, t, ok in (
                ("reward_rises", g.get("reward_last"), g.get("reward_first"), lambda v, t: v > t),
                ("mmlu_holds", g.get("mmlu_after"), mmlu_floor, lambda v, t: v >= t),
                ("gsm8k_improves", g.get("gsm8k_after"), gsm_floor, lambda v, t: v >= t),
                ("groups_untied", g.get("tied_group_fraction"), 0.5, lambda v, t: v < t),
                ("ce_falls", g.get("ce_last"), g.get("ce_first"), lambda v, t: v < t),
            )]
        m["finished"] = now()
        write_manifest(runs_root(), m)
        _timing_snapshot(m)
    print(json.dumps(m, indent=1) if as_json else format_run(m))
    if not gates_pass(m):
        sys.exit(1)


def refuse_blind_curve(n: int, target_pt: float) -> None:
    """Refuse a curve whose subset cannot resolve the effect it exists to locate,
    BEFORE the run spends the time. Worst-case binomial SE (p=0.5), in points:
    50/sqrt(n). The post-run `_se_note` warns at the point's own rate, but a
    warning after a 99-minute run cannot un-spend it (errors/2026-09-08). The
    comparison is strict: SE exactly equal to the target is the documented
    knife-edge (n=100, 5.0 pt against P1's +5.6 pt effect, "1.1 sigma")."""
    se = 50.0 / (n ** 0.5)
    if n > 0 and se > target_pt:
        raise SystemExit(
            f"--eval-curve-n {n} carries a worst-case binomial SE of {se:.1f} pt, "
            f"above the --curve-target-pt {target_pt:g} effect the curve locates: "
            f"the crossing step would be chosen by which rows fell where. Raise "
            f"--eval-curve-n to >={(50.0 / target_pt) ** 2:.0f}, or raise the target")


def _se_note(r: dict) -> str:
    """The subset's sampling width, when it is wide enough to set the answer.

    5.0 pt is P1's own target effect (`roadmap.md`), so an SE at or above it means the
    crossing step is chosen by which rows are in the subset as much as by the policy.
    Silent below that: a note on every line would be read as boilerplate and skipped.

    The width comes from the point's own rate (`ledger.time_to_score`), so this fires on
    the subset's real resolution rather than on p=0.5's worst case -- which at n=50 and
    n=100 warned about subsets that do resolve the effect.
    """
    se = r.get("se_pt")
    if se is None or se < 5.0:
        return ""
    return (f"  [subset n={r['n']}, binomial SE {se:.1f} pt >= P1's +5 pt target: the "
            f"crossing step is sampling-limited, raise --eval-curve-n to narrow it]")

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
    # decline; 0.2 pt is inside the floor; and no paired width means no verdict.
    peak = {"step": 50, "correct": 467, "total": 500, "score": 0.934}
    assert significant_decline({"score": 0.824}, peak, 1.66)
    assert not significant_decline({"score": 0.932}, peak, 1.66)
    assert not significant_decline({"score": 0.824}, peak, None)
    # Cell D: patience > 0 with no paired width REFUSES -- never the silent fallbacks
    # (a guard that never fires, or a stop decided without a sampling width). patience=0 never asks.
    # With a kept step the message says the snapshot is already saved: a mid-run exit
    # must not read as a wasted run.
    try:
        require_paired_width(None, 2, kept_step=50)
        raise AssertionError("no raise")
    except SystemExit as exc:
        assert "paired" in str(exc) and "step 50" in str(exc)
    require_paired_width(1.31, 2)
    require_paired_width(None, 0)
    # Significant-decline veto against two peaks: the same later point vetoes
    # against the higher peak but not the lower one, at the same width.
    p1, p2 = {"step": 5, "score": .940}, {"step": 10, "score": .950}
    assert EarlyStop(1).update(False) == "patience"
    assert EarlyStop(1).update(True) is None
    p3 = {"step": 15, "score": .820}
    assert significant_decline(p3, p2, 6.0)       # vs step-10 peak: -13.0 pt, > 2xSE
    assert not significant_decline(p3, p1, 6.0)   # vs step-5 peak: -12.0 pt, not > 2xSE
    # Curve churn: the run's own instrument reading, recorded per point. Adjacent
    # points pair by the ``i`` key within one run (rows land in completion order
    # since the eval arm writes incrementally); across runs the keys mean different
    # questions. The first point has no predecessor: null, not 0 -- 0 means "no
    # flips", null means "no comparable point". Different lengths or i sets are
    # null too, not a partial count.
    rows = [{"i": i, "correct": bool(c)} for i, c in enumerate((1, 1, 0, 0))]
    flipped = [{"i": i, "correct": bool(c)} for i, c in enumerate((1, 0, 0, 1))]
    assert curve_churn(rows, flipped) == (1, 1)  # one right->wrong, one wrong->right
    assert curve_churn(rows, flipped[::-1]) == (1, 1)  # completion order: pairs by i
    assert curve_churn(None, rows) is None       # first point: no predecessor
    assert curve_churn(rows, rows[:3]) is None   # different n: not comparable
    assert curve_churn(rows, [{"i": i + 1, "correct": True} for i in range(4)]) is None
    print("ledger: ids + best-point selection OK")
