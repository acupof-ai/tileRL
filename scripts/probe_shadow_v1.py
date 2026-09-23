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
# quest measures SM contention (no H2D volume). h2d/both run at three synthetic
# volumes: 107 = offers_pages p90 (proxy), 206 = observed max eviction (real
# bound seen on device), 512 = 4*k supremum. The gate is read at the volume
# matching the measured real promotion p90 (reported per run).
QUEST_MODE = ("quest", None)
H2D_VOLUMES = (107, 206, 512)


def configs(smoke: bool):
    if smoke:
        # Tiny CPU pool cannot hold the device volumes; one small h2d point.
        return [QUEST_MODE, ("h2d", 16), ("both", 16)]
    return [QUEST_MODE] + [(m, p) for p in H2D_VOLUMES for m in ("h2d", "both")]


def label(mode, pages):
    return mode if pages is None else f"{mode}{pages}"


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    i = min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1))))
    return round(s[i], 4)


def run_one_mode(arm_mode, e, ids, max_new):
    """One prompt, alternating OFF/ON graph-tick segments.

    Timing (start-to-start, not the wall of a tick that may itself wait on the
    background):
      - graph tick wall per segment (off/on p50,p90) for the slowdown gate;
      - interval_off: real graph->graph start gaps in OFF segments, the honest
        budget denominator (ON-segment gaps include background wait by design);
      - bg_over_1_interval: at each ON graph tick start, query whether the
        PREVIOUS tick's background end event is already complete. Fraction NOT
        complete = refreshes that would miss a one-tick deadline (tail gate).
    """
    sh = e._sparse_shadow
    sh.set_active(False)

    rid = e.submit(list(ids), _sampling(max_new))
    tm = e._step_timing
    seg_state = "off"
    seg_n = 0
    off_w, on_w, intervals_off = [], [], []
    bg_complete = 0
    bg_missed = 0
    # The engine is the SOLE emitter: each ON graph tick's step() launches the
    # background and stores its end event on sh.last_event. At the NEXT graph
    # tick we non-blocking-query whether it finished within one interval.
    prev_was_on = False
    prev_start = None
    import time

    for _ in range(200000):
        live_row = next((x for x in e._running if x.req_id == rid), None)
        live = live_row is not None
        f0 = e._decode_forwards
        t_start = time.perf_counter()
        if prev_start is not None:
            gap = (t_start - prev_start) * 1000.0
        # Tail gate, checked BEFORE this tick's step: did the previous ON
        # tick's background complete in the intervening interval? Consumed
        # immediately: an eager refresh tick between two graph ticks must NOT
        # re-query the same event (that would inflate the denominator and can
        # turn a real miss into a later "complete").
        if prev_was_on:
            prev_was_on = False
            ev = sh.last_event
            if ev is not None:
                done = True if ev == "inline-done" else bool(ev.query())
                if done:
                    bg_complete += 1
                else:
                    bg_missed += 1
        e.step()
        df = e._decode_forwards - f0
        alive = any(r.req_id == rid for r in e._running)
        is_graph = (live and df and tm is not None and tm.fwd_sparse
                    and tm.fwd_path == "graph")
        if is_graph:
            wall = (time.perf_counter() - t_start) * 1000.0
            if prev_start is not None and seg_state == "off":
                intervals_off.append(gap)
            (on_w if seg_state == "on" else off_w).append(wall)
            seg_n += 1
            prev_was_on = seg_state == "on"
            if seg_n >= SEGMENT:
                if seg_state == "on":
                    sh.wait_pending()  # don't leak background across boundary
                seg_state = "on" if seg_state == "off" else "off"
                seg_n = 0
                sh.set_active(seg_state == "on")
                prev_was_on = False  # boundary tick: no predecessor in new seg
            prev_start = t_start
        if not alive:
            break

    bg = sh.background_ms() if sh is not None else []
    info = sh.info() if sh is not None else {}
    promo = list(getattr(e._sparse, "refresh_promotions", []))
    bg_total = bg_complete + bg_missed
    # Single-emitter invariant: the engine launches background exactly once
    # per ON graph tick. quest/h2d/both all emit; a mismatch means a double
    # emitter (the bug this guards) or a missing launch.
    if len(on_w) and len(bg) != len(on_w):
        raise AssertionError(
            f"{arm_mode}: n_bg {len(bg)} != n_graph_on {len(on_w)} "
            f"(shadow must emit exactly once per ON graph tick)")
    # Each launch is tail-queried at most once (the segment's last ON tick has
    # no successor and is not queried), so queries cannot exceed ON ticks.
    if bg_total > len(on_w):
        raise AssertionError(
            f"{arm_mode}: tail queries {bg_total} > n_graph_on {len(on_w)} "
            f"(an event was queried more than once)")
    return {"mode": arm_mode, "h2d_pages": info.get("carved_scratch_pages"),
            "n_graph_off": len(off_w), "n_graph_on": len(on_w),
            "graph_p50_off": pct(off_w, 50), "graph_p90_off": pct(off_w, 90),
            "graph_p50_on": pct(on_w, 50), "graph_p90_on": pct(on_w, 90),
            "interval_off_p50": pct(intervals_off, 50),
            "interval_off_p90": pct(intervals_off, 90),
            "bg_p50": pct(bg, 50), "bg_p90": pct(bg, 90),
            "bg_p99": pct(bg, 99), "n_bg": len(bg),
            "bg_finished_within_1_interval": bg_complete,
            "bg_exceeded_1_interval": bg_missed,
            "bg_exceed_fraction": round(bg_missed / bg_total, 4)
                if bg_total else None,
            # Real per-refresh cold-promotion distribution observed this run
            # (HostKvPages.take deltas); the h2d gate is read at its p90.
            "refresh_promotions_n": len(promo),
            "refresh_promotions_p50": pct(promo, 50),
            "refresh_promotions_p90": pct(promo, 90),
            "refresh_promotions_max": max(promo) if promo else None,
            "shadow": info}


def verdict_for(rep):
    g_off, g_on = rep["graph_p50_off"], rep["graph_p50_on"]
    bg90, interval_off50 = rep["bg_p90"], rep["interval_off_p50"]
    exceed = rep["bg_exceed_fraction"]
    if None in (g_off, g_on, bg90, interval_off50, exceed):
        return 14, "insufficient timing samples"
    slow = g_on / g_off if g_off else float("inf")
    fits = bg90 <= interval_off50
    slow_ok = slow <= 1.05
    tail_ok = exceed <= 0.05
    rep["graph_p50_slowdown_ratio"] = round(slow, 4)
    rep["go_bg_p90_le_interval_off_p50"] = fits
    rep["go_slowdown_le_1p05"] = slow_ok
    rep["go_tail_exceed_fraction_le_0p05"] = tail_ok
    go = fits and slow_ok and tail_ok
    return (0 if go else 14), (
        f"bg_p90 {bg90} <= interval_off_p50 {interval_off50}: {fits}; "
        f"graph p50 slowdown x{slow:.4f} <=1.05: {slow_ok}; "
        f"bg exceed-1-interval fraction {exceed} <=0.05: {tail_ok}")


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
        for mode, pages in configs(True):
            os.environ["TILERL_SPARSE_SHADOW"] = mode
            if pages is not None:
                os.environ["TILERL_SPARSE_SHADOW_PAGES"] = str(pages)
            e, _be, _cfg = build_smoke_engine("graph_w2048")
            rep = run_one_mode(label(mode, pages), e, ids, 80)
            e.shutdown()
            reps.append(rep)
        with open(args.out, "w") as f:
            json.dump(reps, f, indent=2)
        print(f"[smoke] wrote {args.out} configs={len(reps)}", flush=True)
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
    for mode, pages in configs(False):
        os.environ["TILERL_SPARSE_SHADOW"] = mode
        if pages is not None:
            os.environ["TILERL_SPARSE_SHADOW_PAGES"] = str(pages)
        else:
            os.environ.pop("TILERL_SPARSE_SHADOW_PAGES", None)
        e, _be, _cfg = build_arm_engine(args.model, args.source, args.draft,
                                        "graph_w2048")
        try:
            rep = run_one_mode(label(mode, pages), e, prompts[0],
                               args.max_new_tokens)
        finally:
            e.shutdown()
            torch.cuda.synchronize()
        # quest has no H2D volume: report SM contention/slowdown only, no
        # fits/tail verdict. h2d/both carry the three pre-registered gates.
        if mode == "quest":
            rep["verdict_rc"] = 0
            rep["verdict"] = "informational (quest SM contention; no H2D gate)"
        else:
            rc, note = verdict_for(rep)
            rep["verdict_rc"] = rc
            rep["verdict"] = note
            worst = max(worst, rc)
        reps.append(rep)
        print(f"[{rep['mode']}] rc={rep['verdict_rc']} {rep['verdict']}",
              flush=True)
    with open(args.out, "w") as f:
        json.dump(reps, f, indent=2)
    print(f"wrote {args.out}; worst rc {worst}", flush=True)
    return worst


if __name__ == "__main__":
    sys.exit(main())
