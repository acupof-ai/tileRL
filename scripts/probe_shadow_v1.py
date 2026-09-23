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
# Trimmed to a realistic set. cold_promotions measured 40-66 in the crashed
# window, so 107 brackets the real promotion p90 and 206 is the observed
# eviction max / realistic upper bound. 512 (4*k supremum) dropped: it is 8x
# the measured rate and only ever saturated PCIe. "off" runs the same engine
# with the shadow disabled for an in-process baseline.
H2D_VOLUMES = (107, 206)


def configs(smoke: bool):
    if smoke:
        # Tiny CPU pool cannot hold the device volumes; small h2d points only.
        return [("off", None), ("quest", None), ("h2d", 16),
                ("both", 16)]
    return [("off", None), ("quest", None),
            ("h2d", 107), ("both", 107), ("h2d", 206)]


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
    sh = e._sparse_shadow  # None for the dedicated "off" baseline config
    if sh is not None:
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
        if prev_was_on and sh is not None:
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
            if sh is not None and seg_n >= SEGMENT:
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
    out = list(e.poll().get(rid, []))
    return {"mode": arm_mode, "h2d_pages": info.get("carved_scratch_pages"),
            "n_out": len(out), "output": out,
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
    rep["go"] = go
    # rc1 = a measured gate red (no-go, v2 not built); rc14 above =
    # insufficiency (could not even measure). The two must be separable.
    return (0 if go else 1), (
        f"bg_p90 {bg90} <= interval_off_p50 {interval_off50}: {fits}; "
        f"graph p50 slowdown x{slow:.4f} <=1.05: {slow_ok}; "
        f"bg exceed-1-interval fraction {exceed} <=0.05: {tail_ok}")


def run_worker(mode, pages, args):
    """One config in its own process: build one engine, run one prompt, write
    one json, exit so the OS reclaims the ~31 GiB of model memory before the
    next config (building all configs in one process OOMs on V100)."""
    os.environ["TILERL_SPARSE_SHADOW"] = mode
    if pages is not None:
        os.environ["TILERL_SPARSE_SHADOW_PAGES"] = str(pages)
    else:
        os.environ.pop("TILERL_SPARSE_SHADOW_PAGES", None)
    os.environ.setdefault("TILERL_STEP_TIMING", "1")
    os.environ.setdefault("TILERL_STEP_TIMING_SLOW_MS", "0")

    name = label(mode, pages)
    out_json = f"{args.out_prefix}_{name}.json"
    if args.smoke:
        ids = [7 + (i % 300) for i in range(400)]
        e, _be, _cfg = build_smoke_engine("graph_w2048")
        try:
            rep = run_one_mode(name, e, ids, 80)
        finally:
            e.shutdown()
    else:
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
        e, _be, _cfg = build_arm_engine(args.model, args.source, args.draft,
                                        "graph_w2048")
        try:
            rep = run_one_mode(name, e, prompts[0], args.max_new_tokens)
        finally:
            e.shutdown()
            torch.cuda.synchronize()

    # Smoke is a plumbing/token check: CPU background runs inline and ~3ms
    # graph ticks are noise, so the timing gates are device-only. The driver
    # enforces token identity in both.
    if mode == "off" or args.smoke:
        rep["verdict_rc"] = 0
        rep["verdict"] = (
            "informational baseline (shadow off)" if mode == "off"
            else "smoke: timing gates skipped (CPU; token-identity is the gate)")
        rep["go"] = None
        rc = 0
    elif mode == "quest":
        rep["verdict_rc"] = 0
        rep["verdict"] = "informational (quest SM contention; no H2D gate)"
        rep["go"] = None
        rc = 0
    else:
        rc, note = verdict_for(rep)
        rep["verdict_rc"] = rc
        rep["verdict"] = note
    with open(out_json, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[{name}] rc={rc} {rep['verdict']} -> {out_json}", flush=True)
    return rc


def token_identity_gate(reps):
    """Compare every enabled config's output to the 'off' baseline. Returns
    {config_name: rc}: length mismatch -> 14 (cannot align), aligned but
    differing -> 1 (a real shadow side effect), equal -> absent (0)."""
    out = {}
    base = next((r for r in reps if r["mode"] == "off"), None)
    if base is None:
        return {"_baseline": 14}
    bo = base["output"]
    for r in reps:
        if r["mode"] == "off":
            continue
        o = r["output"]
        if len(o) != len(bo):
            r["token_identity"] = "LENGTH_MISMATCH"
            out[r["mode"]] = 14
        elif o != bo:
            d = next(i for i, (a, b) in enumerate(zip(o, bo)) if a != b)
            r["token_identity"] = f"DIVERGE@{d}"
            out[r["mode"]] = 1
        else:
            r["token_identity"] = f"OK ({len(o)} tokens)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--source", default="")
    ap.add_argument("--draft", default="")
    ap.add_argument("--prompts", default="")
    ap.add_argument("--expect-tree", default="")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out-prefix", default="shadow_v1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", default="",
                    help="worker entry: 'mode' or 'mode:pages' (e.g. h2d:107)")
    args = ap.parse_args()

    if args.worker:
        try:
            if ":" in args.worker:
                wmode, wpagestr = args.worker.split(":", 1)
                wpages = int(wpagestr) if wpagestr else None
            else:
                wmode, wpages = args.worker, None
            if wmode not in ("off", "quest", "h2d", "both"):
                print(f"unknown worker mode {wmode!r}", file=sys.stderr)
                return 14
            return run_worker(wmode, wpages, args)
        except Exception:
            # Any crash (build failure, OOM, invariant) is an INSTRUMENT
            # failure: rc14, never rc1 (a measured gate red). Separable from
            # the exit code alone.
            import traceback

            traceback.print_exc()
            print(f"INSTRUMENT ERROR(rc14) [{args.worker}]: uncaught exception; "
                  f"this is not a gate red", file=sys.stderr)
            return 14

    # Driver: one subprocess per config. It only spawns and aggregates; it
    # never builds an engine itself.
    import subprocess

    tag = "smoke" if args.smoke else "dev"
    rcs = {}
    for mode, pages in configs(args.smoke):
        name = label(mode, pages)
        cfg_arg = mode if pages is None else f"{mode}:{pages}"
        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "--worker", cfg_arg, "--model", args.model,
               "--source", args.source, "--draft", args.draft,
               "--prompts", args.prompts, "--expect-tree", args.expect_tree,
               "--max-new-tokens", str(args.max_new_tokens),
               "--out-prefix", args.out_prefix]
        if args.smoke:
            cmd.append("--smoke")
        out_json = f"{args.out_prefix}_{name}.json"
        # Remove a stale json first: after a crash only the json THIS worker
        # writes may count.
        if os.path.exists(out_json):
            os.remove(out_json)
        with open(f"{args.out_prefix}_{name}.{tag}.out", "w") as out_f, \
                open(f"{args.out_prefix}_{name}.{tag}.err", "w") as err_f:
            rcs[name] = subprocess.run(cmd, stdout=out_f,
                                       stderr=err_f).returncode
        # A crashed worker writes no json: rc14, do not aggregate a missing
        # result as a green.
        if not os.path.exists(out_json):
            print(f"{name}: no {out_json} (worker rc={rcs[name]}) -> rc14",
                  file=sys.stderr)
            rcs[name] = 14

    reps = []
    for mode, pages in configs(args.smoke):
        name = label(mode, pages)
        out_json = f"{args.out_prefix}_{name}.json"
        if not os.path.exists(out_json):
            continue  # already counted as rc14 above
        with open(out_json) as f:
            reps.append(json.load(f))

    # Token-identity gate (plan, Shadow v1): shadow launches background work
    # but changes no residency, so every enabled config MUST reproduce the
    # "off" output token-for-token.
    tok_rc = token_identity_gate(reps)
    for k, v in tok_rc.items():
        rcs[f"token:{k}"] = v
        if k == "_baseline":
            print("token gate: no 'off' baseline json -> rc14", file=sys.stderr)
        elif v == 14:
            print(f"token gate: {k} length mismatch -> rc14", file=sys.stderr)
        elif v == 1:
            print(f"token gate: {k} diverges from off -> rc1", file=sys.stderr)

    agg = f"{args.out_prefix}.json"
    with open(agg, "w") as f:
        json.dump(reps, f, indent=2)
    print(json.dumps(rcs, indent=2))
    print(f"wrote {agg}", flush=True)
    # 1 = a measured no-go; 14 = a crash/insufficiency/missing json.
    return 1 if 1 in rcs.values() else (14 if 14 in rcs.values() else 0)


if __name__ == "__main__":
    sys.exit(main())
