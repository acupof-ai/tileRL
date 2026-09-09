#!/usr/bin/env python3
"""Replay raw-score patience on a RECORDED n=100 curve (no subsampling).

Unlike patience_replay.py (which subsamples n=500 curves down to n=100/50 with
a fixed shuffle), this reads the curve exactly as the run recorded it: each
eval-curve-<step>.jsonl IS the n=100 subset. For patience in 1/2/3 it reports
the stop step, the kept best step, and the variable-part cost (training +
curve evals up to the stop, from the manifest's own secs fields).

Also prints the adjacent-difference distribution and the running-max exceedance
rate p, the quantity a geometric stop-step model needs.

Usage: replay_recorded_curve.py <run_dir>
"""

import json
import sys
from pathlib import Path


def curve_points(run_dir: Path) -> list[dict]:
    m = json.loads((run_dir / "manifest.json").read_text())
    pts = sorted(m.get("eval_curve", {}).get("points", []), key=lambda p: p["step"])
    assert pts, "manifest has no eval_curve points"
    # sanity: the per-problem file's score agrees with the manifest point
    for p in pts:
        rows = [json.loads(l) for l in (run_dir / f"eval-curve-{p['step']}.jsonl").open()]
        c = sum(r["correct"] for r in rows)
        assert c == p["correct"], f"step {p['step']}: file {c} != manifest {p['correct']}"
        assert len(rows) == p["total"], f"step {p['step']}: {len(rows)} rows != {p['total']}"
    return pts


def replay(pts: list[dict], patience: int, train_step_s: float) -> dict:
    best = pts[0]
    since = 0
    stopped = None
    for p in pts[1:]:
        if p["score"] > best["score"]:  # raw, strict — the raw mode's rule
            best, since = p, 0
        else:
            since += 1
            if since >= patience:
                stopped = p
                break
    last = stopped or pts[-1]
    evals = [p for p in pts if p["step"] <= last["step"]]
    cost = last["secs"] + sum(p["eval_secs"] for p in evals)
    return {"patience": patience, "stop_step": last["step"] if stopped else None,
            "ran_full": stopped is None, "kept_step": best["step"],
            "kept_score": best["score"], "evals": len(evals),
            "variable_cost_s": round(cost, 1),
            "note": "first point compiles shapes (jit); its eval_secs is 5.6x steady state"
                    if evals[0].get("jit") else ""}


def main() -> None:
    pts = curve_points(Path(sys.argv[1]))
    train_step = None
    m = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
    train_step = m["metrics"].get("secs_per_step_median", 20.0)
    print(f"curve: {[(p['step'], round(100*p['score'],1)) for p in pts]}")
    print(f"{'patience':>8} {'stop':>5} {'kept':>5} {'kept%':>6} {'evals':>5} {'cost_s':>8}")
    for pat in (1, 2, 3):
        r = replay(pts, pat, train_step)
        stop = str(r["stop_step"]) if r["stop_step"] else "full"
        print(f"{pat:>8} {stop:>5} {r['kept_step']:>5} {100*r['kept_score']:>5.1f} "
              f"{r['evals']:>5} {r['variable_cost_s']:>8.0f}")
    # adjacent differences and running-max exceedance
    diffs = [round(100 * (pts[i]["score"] - pts[i - 1]["score"]), 1) for i in range(1, len(pts))]
    run_max = pts[0]["score"]
    exceeded = 0
    for p in pts[1:]:
        if p["score"] > run_max:
            exceeded += 1
            run_max = p["score"]
    print(f"adjacent diffs (pt): {diffs}")
    print(f"running-max exceedances: {exceeded} of {len(pts) - 1} post-first points "
          f"(p_hat = {exceeded / max(1, len(pts) - 1):.2f})")
    print("note: p_hat on ~19 plateau points is crude; the geometric model "
          "E[stop] ≈ first + every/(1-p) assumes independent noise, plateau points are not iid")


if __name__ == "__main__":
    main()
