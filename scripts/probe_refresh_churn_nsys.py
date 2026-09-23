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
import os
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
    """Meter EVERY ColdSsdFile regardless of when it is created. The spill file
    is lazy: HostKvPages._ssd is None until the first page spills past the host
    budget (kv_tiers.py creates it inside hold/take), so patching instances at
    engine build attaches to nothing and silently reads 0. Patch the CLASS read
    instead; an instance created later is metered on its first call. Returns
    (get_state, set_in_frame); n_instances>0 proves attachment."""
    from tilerl import kv_tiers

    state = {"total": 0, "frame": 0, "in_frame": False,
             "n_instances": 0, "n_reads": 0,
             "n_instances_at_window_start": 0}
    cls = kv_tiers.ColdSsdFile
    orig_read = cls.read

    def metered_read(self, key, pin):
        blob = orig_read(self, key, pin)
        n = sum(t.element_size() * t.numel() for t in blob.values())
        state["total"] += n
        state["n_reads"] += 1
        if state["in_frame"]:
            state["frame"] += n
        return blob

    cls.read = metered_read
    # Count instances whenever one is constructed (covers lazy creation).
    orig_init = cls.__init__

    def metered_init(self, *a, **k):
        orig_init(self, *a, **k)
        state["n_instances"] += 1

    cls.__init__ = metered_init
    state["n_instances_at_window_start"] = state["n_instances"]

    def get_state():
        d = dict(state)
        d.pop("in_frame", None)
        d["meter_attached"] = True
        return d

    return get_state, lambda v: state.__setitem__("in_frame", v)


# --------------------------------------------------------------------------- #
# task 2: nsys-free refresh-tick phase breakdown
# --------------------------------------------------------------------------- #
class PhaseMeter:
    """Per-tick sub-phase meter for the EAGER refresh tick, no nsys. CUDA
    events time GPU spans (quest scoring, select_pages, promote, shared
    promote, evict, the whole eager model.forward); wall time times host
    blocking points (torch.cuda.synchronize, .item/.tolist/.cpu readbacks).
    Counts every host readback — the D2H/sync census nsys would give as raw
    memcpy counts — without changing engine behavior. Attaches by patching
    module/instance/class functions; restore() undoes every patch.

    GPU spans nest/overlap (select runs inside model.forward), so spans are
    reported, not summed into a total; the tick total is the step wall."""

    def __init__(self, e):
        import torch

        self.t = torch
        self.e = e
        self.cuda = torch.cuda.is_available()
        self.active = False
        self.kind = None
        self.tick_i = -1
        self.ticks = {}  # tick_i -> {kind, spans:{...ms}, walls, counts}
        self._events = []  # (tick_i, name, start, end)
        self._cur = None
        self.patches = []

    def _rec(self):
        d = self.ticks.setdefault(self.tick_i, {
            "kind": self.kind,
            "spans_ms": {}, "wall_ms": {},
            "n_item": 0, "n_tolist": 0, "n_cpu": 0, "n_synchronize": 0})
        return d

    def _gpu_span(self, name, fn, *a, **k):
        if not (self.active and self.cuda):
            return fn(*a, **k)
        # A graph's first tick per bucket key calls model.forward INSIDE
        # torch.cuda.graph capture, yet the engine still labels that tick
        # fwd_path=graph. Timing it would put a capture-length (seconds) span
        # in the graph group's eager_trunk p50 — and a replay never calls
        # forward at all, so the median would be over captures only. Skip
        # recording while a stream is capturing. (Recording events during
        # capture is otherwise safe; this is sampling, not capture safety.)
        if self.t.cuda.is_current_stream_capturing():
            return fn(*a, **k)
        s, z = self.t.cuda.Event(enable_timing=True), self.t.cuda.Event(enable_timing=True)
        s.record()
        try:
            return fn(*a, **k)
        finally:
            z.record()
            self._events.append((self.tick_i, name, s, z))

    def _wall(self, name, fn, *a, **k):
        if not self.active:
            return fn(*a, **k)
        import time

        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            d = self._rec()
            d["wall_ms"][name] = d["wall_ms"].get(name, 0.0) + (
                time.perf_counter() - t0) * 1000.0

    def install(self):
        import time

        t = self.t
        e = self.e
        # --- module/instance GPU spans ---
        def patch(obj, attr, name, gpu=True):
            orig = getattr(obj, attr)

            def wrapped(*a, **k):
                if not self.active:
                    return orig(*a, **k)
                if gpu and self.cuda:
                    return self._gpu_span(name, orig, *a, **k)
                return self._wall(name, orig, *a, **k)

            setattr(obj, attr, wrapped)
            self.patches.append((obj, attr, orig))

        from tilerl import sparse_engine as se

        patch(se, "quest_scores", "select_quest_score", gpu=True)
        patch(e._backend, "select_pages", "select_pages", gpu=True)
        kv = e._kv
        if hasattr(kv, "promote_keyed"):
            patch(kv, "promote_keyed", "promote_keyed", gpu=True)
        if hasattr(kv, "shared_promote"):
            # shared_promote itself ends in a per-page cuda.synchronize; the
            # event span includes that host wait on the stream timeline.
            patch(kv, "shared_promote", "shared_promote_sync", gpu=True)
        patch(e._sparse, "evict_victim", "evict_victim", gpu=False)
        patch(e._model, "forward", "eager_trunk_forward", gpu=True)

        # --- host blocking points / readback census ---
        sync0 = t.cuda.synchronize

        def sync_wrapped(*a, **k):
            t0 = time.perf_counter()
            try:
                return sync0(*a, **k)
            finally:
                if self.active:
                    d = self._rec()
                    d["n_synchronize"] += 1
                    d["wall_ms"]["cuda.synchronize"] = d["wall_ms"].get(
                        "cuda.synchronize", 0.0) + (time.perf_counter() - t0) * 1000

        t.cuda.synchronize = sync_wrapped
        self.patches.append((t.cuda, "synchronize", sync0))

        def count_patch(cls, attr, key):
            orig = getattr(cls, attr)

            def wrapped(selfx, *a, **k):
                if self.active:
                    self._rec()[key] += 1
                return orig(selfx, *a, **k)

            setattr(cls, attr, wrapped)
            self.patches.append((cls, attr, orig))

        if self.cuda:
            count_patch(t.Tensor, "item", "n_item")
            count_patch(t.Tensor, "tolist", "n_tolist")
            count_patch(t.Tensor, "cpu", "n_cpu")
        return self

    def begin_tick(self, kind, tick_i):
        import time

        self.kind, self.tick_i, self.active = kind, tick_i, True
        self._t0 = time.perf_counter()

    def end_tick(self):
        import time

        if self.active:
            d = self._rec()
            d["wall_ms"]["step_total"] = (time.perf_counter() - self._t0) * 1000
        self.active = False

    def resolve(self):
        """Sum recorded GPU event spans per tick, once; clears the event log."""
        evs, self._events = self._events, []
        for tick_i, name, s, z in evs:
            d = self.ticks.get(tick_i)
            if d is None:
                continue
            d["spans_ms"][name] = d["spans_ms"].get(name, 0.0) + s.elapsed_time(z)
        return self.ticks

    def restore(self):
        for obj, attr, orig in reversed(self.patches):
            setattr(obj, attr, orig)
        self.patches = []


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
    phase_meter = None
    if os.environ.get("TILERL_REFRESH_PHASES"):
        phase_meter = PhaseMeter(e).install()

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
        # Phase meter brackets every sparse decode tick (not just the nsys
        # frame) by the predicted kind; verified against fwd_path below.
        if live and phase_meter is not None:
            live_row = next((x for x in e._running if x.req_id == rid), None)
            if live_row is not None and live_row.phase == 2:  # _PHASE_DECODE
                phase_meter.begin_tick("refresh" if pred_refresh else "graph",
                                       step_i)
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
        if phase_meter is not None and phase_meter.active:
            phase_meter.end_tick()
        if framed and not frame_done and have_nvtx:
            torch.cuda.nvtx.range_pop()
        df = e._decode_forwards - f0
        alive = any(r.req_id == rid for r in e._running)
        if live and df and tm is not None and tm.fwd_sparse:
            kind = "refresh" if tm.fwd_path != "graph" else "graph"
            # Label integrity is PROCESS-WIDE (the phase meter groups every
            # decode tick, not just the frame): on a PURE decode tick the
            # counter prediction must equal the observed path (prefill/mixed
            # is eager regardless of the counter, excluded by phase_pre).
            if tm.phase_pre == 0:
                pred_kind = "refresh" if pred_refresh else "graph"
                if pred_kind != kind:
                    pred_mismatch.append(
                        {"step": step_i, "pred": pred_kind,
                         "observed": kind, "ticks_since_refresh": sr})
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
    phase_summary = None
    if phase_meter is not None:
        import statistics

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # all recorded events must be queryable
        ticks = phase_meter.resolve()
        phase_meter.restore()

        def agg(kind):
            rows = [d for d in ticks.values() if d["kind"] == kind]
            if not rows:
                return None
            keys = set()
            for d in rows:
                keys |= set(d["spans_ms"]) | set(d["wall_ms"])
            # Per-key {p50, n}: a median over 1 tick must not look like one
            # over 70 (the capture-tick grouping bug read exactly that way).
            per_key = {}
            for k in sorted(keys):
                vals = []
                for d in rows:
                    if k in d["spans_ms"]:
                        vals.append(d["spans_ms"][k])
                    elif k in d["wall_ms"]:
                        vals.append(d["wall_ms"][k])
                if vals:
                    per_key[k] = {"p50_ms": round(statistics.median(vals), 3),
                                  "n": len(vals)}
            counts = {k: sum(d[k] for d in rows)
                      for k in ("n_item", "n_tolist", "n_cpu", "n_synchronize")}
            return {"n_ticks": len(rows), "per_key": per_key,
                    "total_counts": counts}

        phase_summary = {"refresh": agg("refresh"), "graph": agg("graph")}
        # Capture health, surfaced before the gate: a failed sparse capture
        # sets graph_on=False and every tick would then be mislabeled refresh.
        sp = e._sparse
        phase_summary["graph_on"] = bool(getattr(sp, "graph_on", False))
        phase_summary["n_graphs_captured"] = len(getattr(sp, "graphs", {}))
    return {"n_out": len(out),
            "total_refresh_ticks": total_refresh,
            "frame_tick_kinds": frame_ticks,
            "frame_counts": {"graph": frame_ticks.count("graph"),
                             "refresh": frame_ticks.count("refresh")},
            "refreshes_in_frame": refreshes_in_frame,
            "churn_between_adjacent_refreshes": churn_between,
            "pred_observed_mismatches": pred_mismatch,
            "ssd_bytes_read": get_ssd(),
            "refresh_phase_breakdown": phase_summary,
            # How to read: refresh p50 203ms vs graph 46ms (stage0), ~128ms
            # in the eager sparse forward envelope. The breakdown attributes it
            # without nsys (nsys 2022.4.2.1 export is broken on the device):
            # GPU-event spans for quest scoring/select_pages/promote/shared
            # promote/eager trunk, wall time for synchronize/evict, and raw
            # counts of .item/.tolist/.cpu readbacks. A large eager_trunk span
            # with many readbacks and synchronize walls => idle/sync-bound =>
            # 1-tick-delay async refresh (A') removes it; eager_trunk GPU time
            # genuinely higher => longer refresh interval (B). churn Jaccard
            # 0.438 means B/R=16 staleness is not free.
            "interpretation": "spans_ms are GPU-event time (nested, do not "
                              "sum); wall_ms host blocking; see refresh_phase_"
                              "breakdown to choose A' vs B"}


def gate_verdict(rep, want_refresh):
    """Pure device-window gate. Returns (rc, note). rc14 means the result
    cannot certify the question: a mislabeled nsys frame, an SSD meter that did
    not attach or never fired, or too few refreshes in frame. Kept separate
    from main so the two self-proof gates get real negative controls."""
    if rep["pred_observed_mismatches"]:
        return 14, "pred/observed tick-kind mismatch; nsys labels invalid"
    ssd = rep["ssd_bytes_read"]
    if not ssd.get("meter_attached"):
        return 14, "SSD meter class patch missing"
    if ssd["n_instances"] == 0:
        # Class is patched but no spill file was ever lazily created: the run
        # never spilled past the host budget, so refresh-from-SSD is untested.
        return 14, "no ColdSsdFile created (prompt never spilled)"
    if ssd["total"] == 0 or ssd["n_reads"] == 0:
        # File exists but the wrapped read never fired: at 32k with the spill
        # tier the refresh tick promotes from SSD (stage0 showed ssd_mmap
        # ticks), so zero here means the meter is not on the call path. Loud.
        return 14, "ColdSsdFile exists but zero reads (meter miss?)"
    pb = rep.get("refresh_phase_breakdown")
    if pb is None:
        # Window ran without TILERL_REFRESH_PHASES=1: the A'-vs-B question has
        # no data even though the churn/frame checks pass.
        return 14, "refresh_phase_breakdown missing (set TILERL_REFRESH_PHASES=1)"
    if not pb.get("graph_on") or pb.get("n_graphs_captured", 0) == 0:
        # A failed capture sets graph_on=False; every tick then classifies
        # refresh and the frame fills with no graph control — green, no answer.
        return 14, "sparse graph never captured (graph_on/captured=0)"
    if pb.get("refresh") is None or pb.get("graph") is None:
        return 14, "phase breakdown missing a tick group"
    if pb["graph"]["n_ticks"] == 0:
        # Frame held only refresh ticks; there is no graph control to diff.
        return 14, "zero graph ticks in breakdown"
    if pb["refresh"]["n_ticks"] == 0:
        return 14, "zero refresh ticks in breakdown"
    if rep["refreshes_in_frame"] < want_refresh:
        return 14, "not enough refreshes in frame"
    return 0, ""


def _gate_self_check():
    """Negative controls against synthetic reps: every failure mode MUST yield
    rc14 and a good rep 0. Guards against a gate that cannot fire."""
    def grp(n):
        return {"n_ticks": n, "per_key": {}, "total_counts": {}} if n else None

    pb_ok = {"graph_on": True, "n_graphs_captured": 3,
             "refresh": grp(9), "graph": grp(70)}
    good = {"pred_observed_mismatches": [],
            "ssd_bytes_read": {"meter_attached": True, "n_instances": 1,
                               "n_reads": 5, "total": 100, "frame": 100},
            "refreshes_in_frame": 5,
            "refresh_phase_breakdown": pb_ok}
    assert gate_verdict(good, 5) == (0, ""), gate_verdict(good, 5)
    bad_label = dict(good, pred_observed_mismatches=[{"tick": 7}])
    assert gate_verdict(bad_label, 5)[0] == 14
    no_patch = dict(good, ssd_bytes_read={**good["ssd_bytes_read"],
                                          "meter_attached": False})
    assert gate_verdict(no_patch, 5)[0] == 14
    no_file = dict(good, ssd_bytes_read={**good["ssd_bytes_read"],
                                         "n_instances": 0, "n_reads": 0,
                                         "total": 0})
    assert gate_verdict(no_file, 5)[0] == 14
    zero_read = dict(good, ssd_bytes_read={**good["ssd_bytes_read"],
                                           "n_reads": 0, "total": 0})
    assert gate_verdict(zero_read, 5)[0] == 14
    few = dict(good, refreshes_in_frame=4)
    assert gate_verdict(few, 5)[0] == 14
    no_breakdown = dict(good)
    no_breakdown.pop("refresh_phase_breakdown")
    assert gate_verdict(no_breakdown, 5)[0] == 14
    graph_off = dict(good, refresh_phase_breakdown={**pb_ok, "graph_on": False})
    assert gate_verdict(graph_off, 5)[0] == 14
    no_capture = dict(good, refresh_phase_breakdown={**pb_ok,
                                                     "n_graphs_captured": 0})
    assert gate_verdict(no_capture, 5)[0] == 14
    graph_empty = dict(good, refresh_phase_breakdown={
        **pb_ok, "graph": grp(0)})
    assert gate_verdict(graph_empty, 5)[0] == 14
    no_refresh = dict(good, refresh_phase_breakdown={
        **pb_ok, "refresh": {"n_ticks": 0, "per_key": {}, "total_counts": {}}})
    assert gate_verdict(no_refresh, 5)[0] == 14


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
        if os.environ.get("TILERL_REFRESH_PHASES"):
            pb = rep.get("refresh_phase_breakdown")
            assert pb is not None and pb.get("refresh") and pb.get("graph"), pb
            print(f"[smoke] phase groups: "
                  f"refresh n={pb['refresh']['n_ticks']} "
                  f"graph n={pb['graph']['n_ticks']} "
                  f"graph_on={pb['graph_on']} captured={pb['n_graphs_captured']}",
                  flush=True)
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
        elif note == "SSD meter class patch missing":
            print("INSUFFICIENT: ColdSsdFile.read class patch not installed",
                  file=sys.stderr)
        elif "never spilled" in note:
            print("INSUFFICIENT: no ColdSsdFile was created — the prompt never "
                  "spilled past the host tier; refresh-from-SSD untested "
                  "(set H2_COLD_SSD / H2_COLD_SSD_BYTES and a smaller host "
                  "budget)", file=sys.stderr)
        elif "zero reads" in note:
            ssd0 = rep["ssd_bytes_read"]
            print(f"INSUFFICIENT: ColdSsdFile created x{ssd0['n_instances']} but "
                  f"the patched read fired 0 times / 0 bytes; refresh promotes "
                  f"from SSD at 32k (stage0 ssd_mmap), so the meter is missing "
                  f"the call path", file=sys.stderr)
        elif "phase_breakdown missing" in note:
            print("INSUFFICIENT: no refresh phase breakdown — rerun with "
                  "TILERL_REFRESH_PHASES=1; the A'-vs-B attribution is the "
                  "window's question", file=sys.stderr)
        elif "never captured" in note:
            pb0 = rep.get("refresh_phase_breakdown", {})
            print(f"INSUFFICIENT: sparse graph not captured (graph_on="
                  f"{pb0.get('graph_on')}, n_graphs={pb0.get('n_graphs_captured')}"
                  f"); every tick would mislabel refresh and the frame has no "
                  f"graph control", file=sys.stderr)
        elif "tick group" in note or "graph ticks" in note \
                or "refresh ticks" in note:
            print(f"INSUFFICIENT: phase breakdown lacks a valid graph/refresh "
                  f"group: {rep.get('refresh_phase_breakdown')}", file=sys.stderr)
        else:
            print(f"INSUFFICIENT: framed only {rep['refreshes_in_frame']} "
                  f"refreshes (need {args.want_refresh}); raise --max-new-tokens",
                  file=sys.stderr)
        return finish(14, note)
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
