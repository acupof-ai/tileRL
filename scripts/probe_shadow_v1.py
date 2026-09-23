#!/usr/bin/env python3
"""PROBE-ONLY #805: shadow v1 go/no-go driver.

For each mode in quest,h2d,both builds one graph_w2048 engine (mode comes from
TILERL_SPARSE_SHADOW at Engine build) and runs ONE prompt through alternating
OFF/ON segments of >=50 graph ticks each (a placement control, same process),
recording per graph-tick wall time and, on ON ticks, the SparseShadow's
background device time and the previous->current tick interval. Emits a
verdict against the pre-registered go/no-go line in
scripts/REFRESH_1TICK_DELAY_PLAN.md:

  GO iff (1) background select+promote p90 <= graph-tick-interval p50, and
          (2) graph-tick p50 ON is <= 1.05 x graph-tick p50 OFF.

CPU --smoke runs the same loop (side stream is inline there, so it is a
plumbing/token check, not the timing answer).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from probe_serve_sm70_w2048 import (
    _sampling,
    build_arm_engine,
    build_smoke_engine,
    load_prompts,
)

SEGMENT = 50  # graph ticks per off/on segment; n >= 50 per side, alternating
MODES = ("quest", "h2d", "both")


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    i = min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1))))
    return round(s[i], 4)


def run_one_mode(arm_mode, e, ids, max_new):
    """Run one prompt; collect graph tick wall off/on and on-tick background."""
    sh = e._sparse_shadow
    sh.set_active(False)
    sh.reset_timing()

    rid = e.submit(list(ids), _sampling(max_new))
    tm = e._step_timing
    seg = {"state": "off", "n": 0}
    off_w, on_w = [], []
    intervals = []
    last_graph_t = None
    import time

    for _ in range(200000):
        live = any(r.req_id == rid for r in e._running)
        f0 = e._decode_forwards
        t0 = time.perf_counter()
        e.step()
        wall = (time.perf_counter() - t0) * 1000.0
        df = e._decode_forwards - f0
        alive = any(r.req_id == rid for r in e._running)
        if live and df and tm is not None and tm.fwd_sparse \
                and tm.fwd_path == "graph":
            if last_graph_t is not None:
                intervals.append(wall)  # tick wall as the conservative interval
            last_graph_t = wall
            (on_w if seg["state"] == "on" else off_w).append(wall)
            seg["n"] += 1
            if seg["n"] >= SEGMENT:
                seg["state"] = "on" if seg["state"] == "off" else "off"
                seg["n"] = 0
                sh.set_active(seg["state"] == "on")
                sh.reset_timing()
        if not alive:
            break

    bg = sh.background_ms() if sh is not None else []
    info = sh.info() if sh is not None else {}
    return {"mode": arm_mode, "n_graph_off": len(off_w), "n_graph_on": len(on_w),
            "graph_p50_off": pct(off_w, 50), "graph_p90_off": pct(off_w, 90),
            "graph_p50_on": pct(on_w, 50), "graph_p90_on": pct(on_w, 90),
            "interval_p50": pct(intervals, 50),
            "interval_p90": pct(intervals, 90),
            "bg_p50": pct(bg, 50), "bg_p90": pct(bg, 90), "n_bg": len(bg),
            "shadow": info}


def verdict_for(rep):
    g_off, g_on = rep["graph_p50_off"], rep["graph_p50_on"]
    bg90, interval50 = rep["bg_p90"], rep["interval_p50"]
    if None in (g_off, g_on, bg90, interval50) or rep["n_bg"] == 0:
        return 14, "insufficient timing samples"
    slow = g_on / g_off if g_off else float("inf")
    fits = bg90 <= interval50
    slow_ok = slow <= 1.05
    rep["graph_p50_slowdown_ratio"] = round(slow, 4)
    rep["go_bg_p90_le_interval_p50"] = fits
    rep["go_slowdown_le_1p05"] = slow_ok
    return (0 if (fits and slow_ok) else 14), (
        f"bg_p90 {bg90} <= interval_p50 {interval50}: {fits}; "
        f"graph p50 slowdown x{slow:.4f} <=1.05: {slow_ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--source", default="")
    ap.add_argument("--draft", default="")
    ap.add_argument("--prompts", default="")
    ap.add_argument("--expect-tree", default="")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out", default="shadow_v1.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TILERL_STEP_TIMING", "1")
    os.environ.setdefault("TILERL_STEP_TIMING_SLOW_MS", "0")

    if args.smoke:
        reps = []
        ids = [7 + (i % 300) for i in range(400)]
        for mode in MODES:
            os.environ["TILERL_SPARSE_SHADOW"] = mode
            e, _be, _cfg = build_smoke_engine("graph_w2048")
            rep = run_one_mode(mode, e, ids, 80)
            e.shutdown()
            reps.append(rep)
        with open(args.out, "w") as f:
            json.dump(reps, f, indent=2)
        print(f"[smoke] wrote {args.out} modes={len(reps)}", flush=True)
        return 0

    import subprocess

    if not args.expect_tree:
        print("--expect-tree required", file=sys.stderr)
        return 14
    sha = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if sha != args.expect_tree[:8]:
        print(f"tree {sha} != {args.expect_tree[:8]}", file=sys.stderr)
        return 14

    from tilerl.cli import _qwen38_tokenizer

    tok = _qwen38_tokenizer()
    prompts = load_prompts(args.prompts, tok, 1, 20000, 40000)
    reps, worst = [], 0
    for mode in MODES:
        os.environ["TILERL_SPARSE_SHADOW"] = mode
        e, _be, _cfg = build_arm_engine(args.model, args.source, args.draft,
                                        "graph_w2048")
        try:
            rep = run_one_mode(mode, e, prompts[0], args.max_new_tokens)
        finally:
            e.shutdown()
            torch.cuda.synchronize()
        rc, note = verdict_for(rep)
        rep["verdict_rc"] = rc
        rep["verdict"] = note
        worst = max(worst, rc)
        reps.append(rep)
        print(f"[{mode}] rc={rc} {note}", flush=True)
    with open(args.out, "w") as f:
        json.dump(reps, f, indent=2)
    print(f"wrote {args.out}; worst rc {worst}", flush=True)
    return worst


if __name__ == "__main__":
    sys.exit(main())
