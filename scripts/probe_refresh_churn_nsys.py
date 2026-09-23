#!/usr/bin/env python3
"""#805 follow-up: why is the sparse REFRESH tick 4.4x a graph tick on sm70.

One short device window answers the two discriminators 94 approved before
choosing between a longer refresh interval (lever B) and asynchronous refresh
(lever A'):

  1. SELECTION CHURN. At each refresh the eager tick re-selects top-k pages per
     source group. Record, between two ADJACENT refreshes per group:
       replaced = |prev Δ cur|, jaccard = |prev ∩ cur| / |prev ∪ cur|.
     If the set barely moves over 8 decode ticks, SPARSE_REFRESH_TICKS 8→16/32
     is nearly free; large churn means the staleness quality cost is real.
     Read from SparseRuntime.last_selected (the engine's own per-tick chosen
     sets) on eager sparse ticks only; this script changes no engine behavior.

  2. REFRESH-ONLY NSYS FRAME. cudaProfilerStart/Stop brackets a run that
     contains >=5 eager refresh ticks and the graph ticks between them
     (control). nsys then reports D2H copies, cudaStreamSynchronize count and
     the kernel-vs-idle split; this script additionally counts SSD bytes read
     inside the frame (ColdSsdFile.read has no byte counter, only ssd_ms, so
     the driver wraps read and sums the returned blob bytes) and emits one
     NVTX range per tick tagged with its kind so the frame is readable.

The frame's tick kinds are printed on every platform, so --smoke (CPU tiny,
CpuSparseGraph seam) verifies the framing actually brackets BOTH kinds before
the device run. --smoke also runs the churn math negative controls: identical
sets must report zero churn, a perturbed set positive churn.

Usage:
  CPU:  python probe_refresh_churn_nsys.py --smoke
  dev:  python probe_refresh_churn_nsys.py --model qwen38-27b --source $SRC \
            --draft $DRAFT --prompts <37.6k jsonl> --expect-tree $SHA
        then under: nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi …
"""

from __future__ import annotations

import argparse
import json
import sys


# --------------------------------------------------------------------------- #
# churn math — pure, negative-control tested on CPU
# --------------------------------------------------------------------------- #
def churn_stats(prev: dict, cur: dict) -> dict:
    """Per-group churn between two refresh selections. prev/cur map
    group -> iterable of chosen logical page ids."""
    out = {}
    for g in sorted(set(prev) | set(cur)):
        a, b = set(prev.get(g, ())), set(cur.get(g, ()))
        inter, union = len(a & b), len(a | b)
        out[g] = {"prev": len(a), "cur": len(b),
                  "replaced": len(a ^ b),
                  "jaccard": round(inter / union, 4) if union else 1.0}
    return out


def _churn_self_check():
    # Fixed selection: churn must be exactly zero.
    z = churn_stats({0: [1, 2, 3], 1: [7]}, {0: [1, 2, 3], 1: [7]})
    assert all(v["replaced"] == 0 and v["jaccard"] == 1.0 for v in z.values()), z
    # Perturb one page: symmetric difference 2, jaccard strictly below 1.
    p = churn_stats({0: [1, 2, 3]}, {0: [1, 2, 9]})
    assert p[0]["replaced"] == 2 and 0.0 < p[0]["jaccard"] < 1.0, p
    # Whole set replaced.
    f = churn_stats({0: [1]}, {0: [2]})
    assert f[0]["replaced"] == 2 and f[0]["jaccard"] == 0.0, f


# --------------------------------------------------------------------------- #
# SSD read-byte meter (ColdSsdFile only exposes ssd_ms, not bytes)
# --------------------------------------------------------------------------- #
def install_ssd_byte_meter(e):
    """Wrap every ColdSsdFile reachable from the cold tiers; return
    (get_state, set_in_frame). Sums the byte width of the blobs each read
    returns — the device-side cost this window cares about. n_targets>0 proves
    the meter actually attached (a zero byte total with zero targets means the
    meter missed, not that no disk read happened)."""
    state = {"total": 0, "frame": 0, "in_frame": False, "n_targets": 0,
             "n_reads": 0}
    cold = getattr(getattr(e, "_kv", None), "cold", None)
    targets = []
    for attr in ("_ssd",):
        f = getattr(cold, attr, None)
        if f is not None and hasattr(f, "read"):
            targets.append(f)
    for f in getattr(cold, "_shared_ssds", {}).values():
        if hasattr(f, "read"):
            targets.append(f)
    state["n_targets"] = len(targets)
    for f in targets:
        raw = f.read

        def wrapped(key, pin, raw=raw):
            blob = raw(key, pin)
            n = sum(t.element_size() * t.numel() for t in blob.values())
            state["total"] += n
            state["n_reads"] += 1
            if state["in_frame"]:
                state["frame"] += n
            return blob

        f.read = wrapped
    return (lambda: dict(state)), lambda v: state.__setitem__("in_frame", v)


# --------------------------------------------------------------------------- #
# instrumented run
# --------------------------------------------------------------------------- #
def run_window(e, ids, max_new, want_refresh, smoke=False):
    """One prompt. Churn on every eager sparse tick (refresh selections);
    profiler frame starts after warmup and stops after `want_refresh` refreshes
    are seen inside it. Returns a report dict."""
    import torch
    from probe_serve_sm70_w2048 import _sampling

    tm = e._step_timing
    get_ssd, set_frame = install_ssd_byte_meter(e)
    cudart = torch.cuda.cudart() if torch.cuda.is_available() else None
    have_nvtx = torch.cuda.is_available() and hasattr(torch.cuda, "nvtx")

    rid = e.submit(list(ids), _sampling(max_new))
    warmup = 0 if smoke else 24
    framed = False
    frame_done = False
    refreshes_in_frame = 0
    total_refresh = 0
    frame_ticks = []
    churn_between = []
    pred_mismatch = []
    prev_chosen = {}

    for step_i in range(200000):
        live = any(r.req_id == rid for r in e._running)
        # Predict the kind BEFORE the step for the NVTX label: a pure-decode
        # tick whose counter would hit R declines the graph (run_decode_graph
        # peeks the same counter). Confirmed against fwd_path after the step.
        sr = e._sparse.ticks_since_refresh if e._sparse is not None else 0
        pred_refresh = sr + 1 >= 8
        if not framed and step_i >= warmup and live:
            framed = True
            set_frame(True)
            if cudart is not None:
                cudart.cudaProfilerStart()
            print("[nsys-frame] START", flush=True)
        if framed and not frame_done and have_nvtx:
            torch.cuda.nvtx.range_push(
                f"tick_{'refresh' if pred_refresh else 'graph'}")
        f0 = e._decode_forwards
        e.step()
        if framed and not frame_done and have_nvtx:
            torch.cuda.nvtx.range_pop()
        df = e._decode_forwards - f0
        alive = any(r.req_id == rid for r in e._running)
        if live and df and tm is not None and tm.fwd_sparse:
            kind = "refresh" if tm.fwd_path != "graph" else "graph"
            if kind == "refresh":
                total_refresh += 1
                chosen = {}
                ls = getattr(e._sparse, "last_selected", {}).get(rid, {})
                for g, (_cand, picked) in ls.items():
                    chosen[g] = list(picked)
                if prev_chosen and chosen:
                    churn_between.append(churn_stats(prev_chosen, chosen))
                if chosen:
                    prev_chosen = chosen
            if framed and not frame_done:
                frame_ticks.append(kind)
                # Label integrity: on a PURE decode tick the counter prediction
                # must equal the observed path (a prefill/mixed tick is eager
                # regardless of the counter, so it is excluded). A mismatch
                # means the NVTX tick_* label does not match the bucketed kind
                # and nsys grouping would be mislabeled.
                if tm.phase_pre == 0:
                    pred_kind = "refresh" if pred_refresh else "graph"
                    if pred_kind != kind:
                        pred_mismatch.append(
                            {"tick": len(frame_ticks) - 1,
                             "pred": pred_kind, "observed": kind,
                             "ticks_since_refresh": sr})
                print(f"[nsys-frame] tick {len(frame_ticks) - 1}: {kind} "
                      f"(pred {'refresh' if pred_refresh else 'graph'})",
                      flush=True)
                if kind == "refresh":
                    refreshes_in_frame += 1
                if refreshes_in_frame >= want_refresh:
                    frame_done = True
                    if cudart is not None:
                        cudart.cudaProfilerStop()
                    set_frame(False)
                    print(f"[nsys-frame] STOP ticks={len(frame_ticks)} "
                          f"refresh={refreshes_in_frame}; draining request",
                          flush=True)
        if not alive:
            break

    if framed and not frame_done:
        # Request ended before the frame filled: stop cleanly anyway.
        if cudart is not None:
            cudart.cudaProfilerStop()
        set_frame(False)
        print(f"[nsys-frame] STOP (early end) ticks={len(frame_ticks)} "
              f"refresh={refreshes_in_frame}", flush=True)

    out = list(e.poll().get(rid, []))
    return {"n_out": len(out),
            "total_refresh_ticks": total_refresh,
            "frame_tick_kinds": frame_ticks,
            "frame_counts": {"graph": frame_ticks.count("graph"),
                             "refresh": frame_ticks.count("refresh")},
            "refreshes_in_frame": refreshes_in_frame,
            "churn_between_adjacent_refreshes": churn_between,
            "pred_observed_mismatches": pred_mismatch,
            "ssd_bytes_read": get_ssd(),
            # How to read this against the nsys report. The measured wall gap
            # is refresh 203ms vs graph 46ms (stage0 p50), and ~128ms of it
            # sits in the eager sparse model.forward envelope. nsys decides
            # kernel-vs-idle: if the two tick kinds have similar GPU kernel
            # time but the refresh tick shows far more stream-idle/D2H/sync,
            # lever A' (move selection reconciliation/promotion off the tick)
            # removes it; if refresh kernel time itself is higher, lever B (a
            # longer refresh interval) is the cheaper answer. churn decides
            # whether B's staleness is free (sets stable -> Jaccard ~1).
            "interpretation": "refresh-graph gap ~157ms wall, ~128ms in "
                              "eager model envelope; compare nsys kernel vs "
                              "idle to choose A'(idle/sync-bound) vs B(kernel)"}


def gate_verdict(rep, want_refresh):
    """Pure device-window gate. Returns (rc, note). rc14 means the result
    cannot certify the question: a mislabeled nsys frame, an SSD meter that did
    not attach or never fired, or too few refreshes in frame. Kept separate
    from main so the two self-proof gates get real negative controls."""
    if rep["pred_observed_mismatches"]:
        return 14, "pred/observed tick-kind mismatch; nsys labels invalid"
    ssd = rep["ssd_bytes_read"]
    if ssd["n_targets"] == 0:
        return 14, "SSD meter unattached"
    if ssd["total"] == 0 or ssd["n_reads"] == 0:
        return 14, "SSD meter attached but zero reads"
    if rep["refreshes_in_frame"] < want_refresh:
        return 14, "not enough refreshes in frame"
    return 0, ""


def _gate_self_check():
    """Negative controls for the two self-proof gates against synthetic reps:
    a mismatch and an unattached/zero-read meter MUST each yield rc14; a good
    rep must yield 0. Guards against a gate that cannot fire."""
    good = {"pred_observed_mismatches": [],
            "ssd_bytes_read": {"n_targets": 1, "n_reads": 5, "total": 100,
                               "frame": 100},
            "refreshes_in_frame": 5}
    assert gate_verdict(good, 5) == (0, ""), gate_verdict(good, 5)
    bad_label = dict(good, pred_observed_mismatches=[{"tick": 7}])
    assert gate_verdict(bad_label, 5)[0] == 14
    no_target = dict(good, ssd_bytes_read={**good["ssd_bytes_read"],
                                           "n_targets": 0, "n_reads": 0,
                                           "total": 0})
    assert gate_verdict(no_target, 5)[0] == 14
    zero_read = dict(good, ssd_bytes_read={**good["ssd_bytes_read"],
                                           "n_targets": 1, "n_reads": 0,
                                           "total": 0})
    assert gate_verdict(zero_read, 5)[0] == 14
    few = dict(good, refreshes_in_frame=4)
    assert gate_verdict(few, 5)[0] == 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--source", default="")
    ap.add_argument("--draft", default="")
    ap.add_argument("--prompts", default="")
    ap.add_argument("--expect-tree", default="")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--want-refresh", type=int, default=5)
    ap.add_argument("--out", default="refresh_churn_nsys.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # run_window reads fwd_path/fwd_sparse from the engine's step timing; it is
    # constructed from env at Engine init, so set it before any build.
    import os

    os.environ.setdefault("TILERL_STEP_TIMING", "1")
    os.environ.setdefault("TILERL_STEP_TIMING_SLOW_MS", "0")

    _churn_self_check()
    _gate_self_check()
    print("[self-check] churn math + window gates: OK", flush=True)

    if args.smoke:
        from probe_serve_sm70_w2048 import build_smoke_engine

        e, _be, _cfg = build_smoke_engine("graph_w2048")
        # 400 tokens: long enough that pages fall outside the 8-page own window
        # and the bounds scorer picks real candidates (a 96-token prompt has
        # zero candidates, so churn would only ever see empty sets).
        ids = [7 + (i % 300) for i in range(400)]
        rep = run_window(e, ids, 80, want_refresh=2, smoke=True)
        e.shutdown()
        # Framing control: the CPU frame must contain BOTH tick kinds — a frame
        # that bracketed only graph ticks would prove nothing on device.
        if rep["frame_counts"]["refresh"] < 1 or rep["frame_counts"]["graph"] < 1:
            print(f"SMOKE FAIL: frame missing a tick kind: {rep['frame_counts']}",
                  file=sys.stderr)
            return 14
        print(f"[smoke] frame kinds = {rep['frame_counts']}", flush=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"[smoke] wrote {args.out}", flush=True)
        return 0

    # Device window: graph arm build, one real long prompt.
    import subprocess

    if not args.expect_tree:
        print("--expect-tree required", file=sys.stderr)
        return 14
    sha = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if sha != args.expect_tree[:8]:
        print(f"tree {sha} != {args.expect_tree[:8]}", file=sys.stderr)
        return 14

    from probe_serve_sm70_w2048 import build_arm_engine, load_prompts

    from tilerl.cli import _qwen38_tokenizer

    e, be, _cfg = build_arm_engine(args.model, args.source, args.draft,
                                   "graph_w2048")
    tok = _qwen38_tokenizer()
    prompts = load_prompts(args.prompts, tok, 1, 20000, 40000)
    try:
        rep = run_window(e, prompts[0], args.max_new_tokens, args.want_refresh)
    finally:
        e.shutdown()
        import torch

        torch.cuda.synchronize()
    def finish(rc, note=""):
        if note:
            rep["gate_note"] = note
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"wrote {args.out}: {rep['frame_counts']} "
              f"ssd={rep['ssd_bytes_read']} {note}".rstrip(), flush=True)
        return rc

    rc, note = gate_verdict(rep, args.want_refresh)
    if rc == 14:
        if "mismatch" in note:
            print(f"INSUFFICIENT: {len(rep['pred_observed_mismatches'])} "
                  f"pred-vs-observed tick mismatches: "
                  f"{rep['pred_observed_mismatches'][:5]}", file=sys.stderr)
        elif note == "SSD meter unattached":
            print("INSUFFICIENT: SSD byte meter attached to ZERO ColdSsdFile "
                  "(cold._ssd / _shared_ssds both absent); configure the spill "
                  "tier (H2_COLD_SSD / H2_COLD_SSD_BYTES). A zero-byte result "
                  "from an unattached meter is not a reading.", file=sys.stderr)
        elif "zero reads" in note:
            ssd0 = rep["ssd_bytes_read"]
            print(f"INSUFFICIENT: SSD meter attached to {ssd0['n_targets']} "
                  f"file(s) but observed 0 reads / 0 bytes over the whole run; "
                  f"cannot certify refresh SSD cost (confirm the prompt spills "
                  f"past the host tier)", file=sys.stderr)
        else:
            print(f"INSUFFICIENT: framed only {rep['refreshes_in_frame']} "
                  f"refreshes (need {args.want_refresh}); raise --max-new-tokens",
                  file=sys.stderr)
        return finish(14, note)
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
