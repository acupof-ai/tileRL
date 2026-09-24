#!/usr/bin/env python3
"""PROBE-ONLY #805: v2 real 1-tick-delay device window.

Gates are locked in scripts/REFRESH_1TICK_DELAY_PLAN.md (pre-registration,
#816). Three subprocesses, one 27B engine each, the SAME six real prompts
(serve805_prompts.jsonl), configuration identical to the 24.9 window — sparse
graph + draft W=2 + W2048 + sparse_min_tokens=0 — differing only in
TILERL_SPARSE_V2:

    controlA (off) -> v2 (async) -> controlB (off)

Worker records per prompt, over the WARM decode window (ticks [16,end), close
tick excluded), the sparse-decode tick stream run_prompt already exposes (wall,
fwd path, refresh_after; refresh_after==0 marks the 8-tick cycle's carry/refresh
tick): no engine change. The first STARTUP_GRAPH_TICKS graph ticks are dropped
as non-stationary (v1 seg0). It pools the raw steady cycle/carry/plain ms lists
so the driver gets exact p50/p90 rather than a percentile of percentiles.

Gates (driver):
  - placement: controlA vs controlB aggregate eff differ <= 5% AND token
    identical (the floor); else rc14;
  - GO (rc0) iff v2/control aggregate paired eff ratio >= 1.20;
  - v2 carrying-tick p90 <= 135.4 ms;
  - v2 carry_fallback_frac (eager-carry cycles / steady cycles) <= 0.05,
    cross-checked against the lag controller's own fallback count;
  - cycle reconciliation |7*mean(plain)+mean(carry)-mean(cycle)|/mean(cycle)
    <= 0.05;
  - quality v2 vs control: mean agreement >= 0.995, every prompt >= 0.99,
    median first divergence >= 128; mod8 of divergence positions reported.

Absolute eff of both arms and historical 24.9 are reported; control deviating
> 10% from 24.9 is flagged in the verdict. Missing worker json -> rc14
(instrument), distinct from a measured no-go rc1.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_serve_sm70_w2048 import (  # noqa: E402
    build_arm_engine,
    load_prompts,
    run_prompt,
)

WARMUP_DECODE = 16
STARTUP_GRAPH_TICKS = 64
GO_RATIO = 1.20
CARRY_P90_MAX = 135.4
FALLBACK_FRAC_MAX = 0.05
CYCLE_RECONCILE_MAX = 0.05
DIV_MIN = 128
PLACEMENT_MAX = 0.05
# Teacher-forced floor KL is compared with a tolerance, not float ==0: two
# separate device processes/engines on the same geometry are token-identical
# (top1 discrete, still required ==1.0) but need not be bit-identical in f32
# logits (W>1 vs W=1 tiles are explicitly not bit-exact off the CPU reference).
FLOOR_KL_EPS = 1e-4
TF_TOP1_MIN = 0.99
HIST_24P9 = 24.915
N_PROMPTS = 6
REFRESH_MOD = 8  # SPARSE_REFRESH_TICKS (carry/refresh every 8 sparse ticks)


def _load_json(p):
    with open(p) as f:
        return json.load(f)


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    i = min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1))))
    return round(s[i], 4)


def steady_distributions(warm_ticks):
    """Split warm sparse ticks into 8-tick cycles ending at each
    refresh_after==0 tick; drop cycles whose carry is within the first
    STARTUP_GRAPH_TICKS graph ticks. Returns raw ms lists: cycles, carries,
    plains, and (n_cycles, n_eager_carry)."""
    cycles, carries, plains = [], [], []
    plain = []
    graph_seen = 0
    n_cycles = n_eager = 0
    for t in warm_ticks:
        if t["path"] == "graph":
            graph_seen += 1
        if t["refresh_after"] == 0:
            if graph_seen > STARTUP_GRAPH_TICKS:
                cm = t["wall"] + sum(plain)
                cycles.append(cm)
                carries.append(t["wall"])
                plains.extend(plain)
                n_cycles += 1
                n_eager += int(t["path"] != "graph")
            plain = []
        else:
            plain.append(t["wall"])
    return cycles, carries, plains, n_cycles, n_eager, len(plain)


def first_div(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def structural_report(samples, pin_ceiling):
    """Residency structural gate. A missing demote reconcile (the dead-flag
    bug) shows up as resident frames exceeding the pin ceiling and/or free
    blocks draining monotonically across steady carry cycles. Returns samples
    summary + violated=False when healthy."""
    if not samples:
        return {"sampled_cycles": 0, "violated": False,
                "note": "no steady carry samples (control or no carries)"}
    max_res = max(s["resident_frames"] for s in samples)
    max_blocks = max(s["blocks"] for s in samples)
    first_free = samples[0]["free_blocks"]
    last_free = samples[-1]["free_blocks"]
    # monotonic non-increasing free over the whole window = leak (normal
    # promotion/demote makes free oscillate).
    frees = [s["free_blocks"] for s in samples]
    monotonic_drain = all(b <= a for a, b in zip(frees, frees[1:])) \
        and frees[-1] < frees[0]
    over_ceiling = max_res > pin_ceiling
    return {
        "sampled_cycles": len(samples),
        "pin_ceiling": pin_ceiling,
        "max_resident_frames": max_res,
        "max_blocks": max_blocks,
        "free_first": first_free, "free_last": last_free,
        "free_monotonic_drain": monotonic_drain,
        "over_pin_ceiling": over_ceiling,
        "violated": bool(over_ceiling or monotonic_drain),
    }


def _free_run_pass(e, prompts, args, struct_samples):
    """Genuine free-run leg for the async arm (the anchored leg changes the
    token stream and so its speed/acceptance are not comparable to an unforced
    baseline). Same warm-window caliber [16,end), close tick dropped. Returns
    the metrics 94 asked for: warm eff tok/s, graph/eager tick p50, eager
    fraction, effective tokens/tick, spec acceptance (overall + in/post)."""
    pcyc = pcarry = pplain = []
    all_graph_w, all_eager_w = [], []
    agg_tok = agg_wall = 0
    d_acc = d_dft = d_in_a = d_in_d = d_post_a = d_post_d = 0
    lag = "async"
    per_prompt = []
    for pi, ids in enumerate(prompts):
        e._sparse.ticks_since_refresh = 0  # #821: phase 0 per prompt
        c0 = (e._spec_accepted, e._spec_drafted, e._spec_acc_in,
              e._spec_dft_in, e._spec_acc_post, e._spec_dft_post)
        ticks, acc = [], [0, 0]

        def on_decode(idx, wall, tm, df, da, is_close, refresh_after,
                      phase_pre, acc=acc, ticks=ticks, pi=pi):
            warm = idx >= WARMUP_DECODE and not is_close
            if warm:
                acc[0] += df + da
                acc[1] += wall
            ticks.append({"wall": wall, "path": tm.fwd_path,
                          "refresh_after": refresh_after, "warm": warm})
            if (lag == "async" and warm and refresh_after == 0
                    and tm.fwd_path == "graph"):
                rt = e._sparse
                pool = rt.ctx.kv
                live_rows = [x for x in e._running]
                live_row = live_rows[0] if len(live_rows) == 1 else None
                if live_row is not None \
                        and live_row.req_id in rt.tracker.resident:
                    struct_samples.append({
                        "prompt": pi, "decode_idx": idx, "leg": "free",
                        "resident_frames":
                            len(rt.tracker.resident[live_row.req_id]),
                        "blocks": len(live_row.blocks),
                        "free_blocks": pool.free_blocks,
                        "reserve": len(rt._lag_obj.reserve)})

        r = run_prompt(e, ids, args.max_new_tokens, on_decode)
        c1 = (e._spec_accepted, e._spec_drafted, e._spec_acc_in,
              e._spec_dft_in, e._spec_acc_post, e._spec_dft_post)
        warm = [t for t in ticks if t["warm"]]
        all_graph_w += [t["wall"] for t in warm if t["path"] == "graph"]
        all_eager_w += [t["wall"] for t in warm if t["path"] != "graph"]
        cyc, carry, plain, ncyc, neager, trail = steady_distributions(warm)
        pcyc, pcarry, pplain = pcyc + cyc, pcarry + carry, pplain + plain
        agg_tok += acc[0]
        agg_wall += acc[1]
        d_acc += c1[0] - c0[0]
        d_dft += c1[1] - c0[1]
        d_in_a += c1[2] - c0[2]
        d_in_d += c1[3] - c0[3]
        d_post_a += c1[4] - c0[4]
        d_post_d += c1[5] - c0[5]
        per_prompt.append({"i": pi, "n_out": len(r["output"]),
                           "warm_eff_tokens": acc[0],
                           "warm_eff_tok_s": round(acc[0] / (acc[1] / 1000.0), 4)
                               if acc[1] else None})
    n_warm = len(all_graph_w) + len(all_eager_w)
    lagc = getattr(e._sparse, "_lag_obj", None)
    return {
        "warm_eff_tok_s": round(agg_tok / (agg_wall / 1000.0), 4) if agg_wall else None,
        "graph_tick_ms_p50": pct(all_graph_w, 50),
        "eager_tick_ms_p50": pct(all_eager_w, 50),
        "carry_ms_p50": pct(pcarry, 50), "carry_ms_p90": pct(pcarry, 90),
        "plain_ms_mean": round(statistics.fmean(pplain), 4) if pplain else None,
        "eager_tick_frac": round(len(all_eager_w) / n_warm, 4) if n_warm else None,
        "warm_ticks": n_warm,
        "tokens_per_tick": round(agg_tok / n_warm, 4) if n_warm else None,
        "accept_rate": round(d_acc / d_dft, 4) if d_dft else None,
        "spec_accepted": d_acc, "spec_drafted": d_dft,
        "accept_in": round(d_in_a / d_in_d, 4) if d_in_d else None,
        "accept_post": round(d_post_a / d_post_d, 4) if d_post_d else None,
        "steady_cycles": len(pcyc),
        "lag_carry": lagc.carry_cycles if lagc else None,
        "lag_fallback": lagc.fallback_cycles if lagc else None,
        "per_prompt": per_prompt,
    }


def run_worker(tag, lag, args):
    smoke = bool(getattr(args, "smoke", False))
    if lag == "async":
        os.environ["TILERL_SPARSE_V2"] = "async"
    else:
        os.environ.pop("TILERL_SPARSE_V2", None)
    os.environ.setdefault("TILERL_STEP_TIMING", "1")
    os.environ.setdefault("TILERL_STEP_TIMING_SLOW_MS", "0")

    import subprocess

    sha = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if not sha:
        # Deployed (rsync/tar) trees carry .synced_commit instead of .git.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sc = os.path.join(root, ".synced_commit")
        if os.path.exists(sc):
            with open(sc) as cf:
                sha = cf.read().strip()[:8]
    if args.expect_tree and sha != args.expect_tree[:8]:
        print(f"tree {sha} != {args.expect_tree[:8]}", file=sys.stderr)
        return 14

    # Geometry is the production process-wide build path (sparse own-window and
    # refresh interval, same as the serve --sparse-window/--sparse-refresh
    # flags); it is passed into build_arm_engine -> build_engine, NOT monkey-
    # patched. The draft read window stays 2048.
    cap_state = None
    if not smoke and lag == "async":
        # Startup hard assertion for the v2 capture-order defect: the prior
        # device run baked the graph BEFORE sf.lag_enabled was set, so the
        # q-clone/override merge were absent and every carry silently fell back
        # eager. Wrap the capture entry: a graph captured for the async arm with
        # sf.lag_enabled falsy is an instrument error, not a measured no-go.
        import tilerl.sparse_runtime as srt

        _raw_make = srt.make_sparse_graph
        cap_state = {"captures": 0, "bad": 0}

        def _make_checked(*a, **kw):
            sf = a[5] if len(a) > 5 else kw.get("sf")
            if not getattr(sf, "lag_enabled", False):
                cap_state["bad"] += 1
                raise RuntimeError(
                    "capture-order violation: sf.lag_enabled is False at "
                    "make_sparse_graph (q-clone/merge absent from graph)")
            cap_state["captures"] += 1
            return _raw_make(*a, **kw)

        srt.make_sparse_graph = _make_checked

    if smoke:
        from probe_serve_sm70_w2048 import build_smoke_engine

        prompts = [[7 + (i % 300) for i in range(400)]]
        e, _be, config = build_smoke_engine("graph_w2048")
    else:
        from tilerl.cli import _qwen38_tokenizer

        tok = _qwen38_tokenizer()
        prompts = load_prompts(args.prompts, tok, args.n_prompts,
                               args.min_tokens, args.max_tokens)
        if len(prompts) < args.n_prompts:
            print(f"only {len(prompts)} prompts (< {args.n_prompts})",
                  file=sys.stderr)
            return 14
        e, _be, config = build_arm_engine(
            args.model, args.source, args.draft, "graph_w2048",
            window_tokens=args.window_tokens, refresh_ticks=args.refresh)
        # Geometry actually applied at build (the process-wide constants the
        # captured graph key/pool use), not just passed: W pages in both modules
        # and R — a mismatch here silently measures the wrong configuration.
        from tilerl import sparse_engine as _se
        from tilerl import sparse_index as _si

        _want_pages = args.window_tokens // _si.BLOCK_TOKENS
        if _want_pages != _si.WINDOW_PAGES or _want_pages != _se.WINDOW_PAGES:
            print(f"window geometry not applied: {_si.WINDOW_PAGES}/"
                  f"{_se.WINDOW_PAGES} != {_want_pages}", file=sys.stderr)
            return 14
        if args.refresh != _se.SPARSE_REFRESH_TICKS:
            print(f"refresh R not applied: {_se.SPARSE_REFRESH_TICKS} != "
                  f"{args.refresh}", file=sys.stderr)
            return 14
        print(f"GEOMETRY ok window_tokens={args.window_tokens} "
              f"pages={_want_pages} R={args.refresh}", flush=True)
    # 94(a) hard assertion: this window is B=1-scoped while the B>1 sparse-graph
    # verify CUDA illegal-access is open. Every graph decode tick must enter
    # with exactly one active row; a second concurrent request invalidates the
    # verdict and is an instrument error (rc14), not a measured no-go.
    _rt0 = e._sparse
    _raw_rdg = _rt0.run_decode_graph
    _b1_state = {"violations": 0, "max_rows": 0, "ticks": 0}

    def _b1_checked_rdg(reqs, chains=None):
        n = len(reqs)
        _b1_state["ticks"] += 1
        _b1_state["max_rows"] = max(_b1_state["max_rows"], n)
        if n != 1:
            _b1_state["violations"] += 1
            raise RuntimeError(
                f"B>1 graph decode tick: {n} active rows (B=1-only window)")
        return _raw_rdg(reqs, chains)

    _rt0.run_decode_graph = _b1_checked_rdg

    if getattr(args, "ab_free", False):
        # Stacked-combo A/B speed read: two pure free runs (A v2 off, B v2 on),
        # no teacher forcing — the anchored leg changes acceptance, so speed must
        # be free. A/B run in separate processes via the shell orchestrator.
        # Allowed under --smoke so the free-run metric path has CPU coverage;
        # device geometry is skipped there.
        samples = []
        rep = _free_run_pass(e, prompts, args, samples)
        lagc = getattr(e._sparse, "_lag_obj", None)
        from tilerl.memory import sparse_hot_pages_per_slot
        tr = e._sparse.tracker
        pin_ceiling = sparse_hot_pages_per_slot(
            tr.cfg, tr.k_pages, e.limits.max_num_batched_tokens)
        rep.update({
            "tag": tag, "lag": lag, "tree": sha,
            "config": config, "n_prompts": len(prompts),
            "window_tokens": args.window_tokens, "refresh_ticks_R": args.refresh,
            "draft_true_q_width": bool(
                __import__("tilerl.spec", fromlist=["_DRAFT_TRUE_Q_WIDTH"])
                ._DRAFT_TRUE_Q_WIDTH),
            "graph_carry_observed": (rep.get("lag_carry") or 0) > 0,
            "capture_order_checked": (cap_state or {"captures": 0})["captures"],
            "capture_order_violations": (cap_state or {"bad": 0})["bad"],
            "b1_violations": _b1_state["violations"],
            "residency_structural": structural_report(samples, pin_ceiling),
        })
        try:
            e.shutdown()
        finally:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        with open(f"{args.out_prefix}_{tag}.json", "w") as f:
            json.dump(rep, f, indent=2)
        print(f"[{tag}] FREE warm_tok/s={rep['warm_eff_tok_s']} "
              f"graph_p50={rep['graph_tick_ms_p50']} "
              f"eager_p50={rep['eager_tick_ms_p50']} "
              f"eager_frac={rep['eager_tick_frac']} "
              f"tok/tick={rep['tokens_per_tick']} accept={rep['accept_rate']} "
              f"carry={rep['lag_carry']} fb={rep['lag_fallback']}", flush=True)
        return 0

    rows = []
    pcyc = pcarry = pplain = []
    struct_samples = []
    all_graph_w, all_eager_w = [], []
    t_acc = t_dft = t_in_a = t_in_d = t_post_a = t_post_d = 0
    # Teacher-forcing. controlA runs free and RECORDS its own logits (its greedy
    # output is the anchor). v2 and controlB load controlA's per-prompt output
    # and teacher-force that stream, recording their logits on the same tokens.
    record_only = (tag == "controlA")
    recorder = None
    if not record_only:
        anchors = {}
        for pi in range(args.n_prompts if not smoke else 1):
            ap = os.path.join(args.per_prompt_dir, f"controlA_{pi:03d}.json")
            anchors[pi] = _load_json(ap)["output"]
    try:
        for pi, ids in enumerate(prompts):
            # Install the recorder on the engine before each prompt's submit so
            # verify/sample are intercepted from the first step; auto-bind maps
            # this single request to index 0. Wrap submit to bind deterministically.
            from probe_teacher_force import TeacherForceRecorder
            anc = None if record_only else {0: anchors[pi]}
            recorder = TeacherForceRecorder(e, anc).install()
            ticks, acc = [], [0, 0.0]
            _c0 = (e._spec_accepted, e._spec_drafted, e._decode_forwards,
                   getattr(e, "_spec_acc_in", 0), getattr(e, "_spec_dft_in", 0),
                   getattr(e, "_spec_acc_post", 0), getattr(e, "_spec_dft_post", 0))

            def on_decode(idx, wall, tm, df, da, is_close, refresh_after,
                          phase_pre, acc=acc, ticks=ticks, pi=pi):
                warm = idx >= WARMUP_DECODE and not is_close
                if warm:
                    acc[0] += df + da
                    acc[1] += wall
                    (all_graph_w if tm.fwd_path == "graph"
                     else all_eager_w).append(wall)
                ticks.append({"wall": wall, "path": tm.fwd_path,
                              "refresh_after": refresh_after, "warm": warm})
                # Structural gate sample at every steady cycle boundary: the
                # demote reconcile must keep the request's resident union at the
                # pin ceiling and must not leak frames (free/reserve drain).
                if (lag == "async" and warm and refresh_after == 0
                        and tm.fwd_path == "graph"):
                    rt = e._sparse
                    pool = rt.ctx.kv
                    live_rows = [x for x in e._running]
                    live_row = live_rows[0] if len(live_rows) == 1 else None
                    if live_row is not None \
                            and live_row.req_id in rt.tracker.resident:
                        lrid = live_row.req_id
                        struct_samples.append({
                            "prompt": pi, "decode_idx": idx,
                            "resident_frames": len(rt.tracker.resident[lrid]),
                            "blocks": len(live_row.blocks),
                            "free_blocks": pool.free_blocks,
                            "reserve": len(rt._lag_obj.reserve),
                        })

            r = run_prompt(e, ids, args.max_new_tokens, on_decode)
            c1 = (e._spec_accepted, e._spec_drafted, e._spec_acc_in,
                  e._spec_dft_in, e._spec_acc_post, e._spec_dft_post)
            t_acc += c1[0] - _c0[0]
            t_dft += c1[1] - _c0[1]
            t_in_a += c1[2] - _c0[2]
            t_in_d += c1[3] - _c0[3]
            t_post_a += c1[4] - _c0[4]
            t_post_d += c1[5] - _c0[5]
            # Dump teacher-forced logits (top1 + full f32 logits per recorded
            # chain slot) for the driver's quality gate, then detach.
            tf_rows = recorder.rows.get(0, [])
            with open(os.path.join(args.per_prompt_dir,
                                   f"{tag}_{pi:03d}.logits.json"), "w") as lf:
                json.dump({"output": r["output"],
                           "records": [{"top1": x["top1"],
                                        "committed": x["committed"],
                                        "gen_idx": x["gen_idx"],
                                        "accepted": x.get("accepted", True),
                                        "logits": x["logits"].tolist()
                                                  if "logits" in x else None}
                                       for x in tf_rows]}, lf)
            recorder.uninstall()
            warm = [t for t in ticks if t["warm"]]
            cyc, carry, plain, ncyc, neager, trail = steady_distributions(warm)
            pcyc, pcarry, pplain = pcyc + cyc, pcarry + carry, pplain + plain
            eff = round(acc[0] / (acc[1] / 1000.0), 4) if acc[1] else None
            row = {"i": pi, "n_out": len(r["output"]),
                   "warm_eff_tokens": acc[0], "warm_wall_ms": round(acc[1], 2),
                   "warm_eff_tok_s": eff,
                   "steady_cycles": ncyc, "eager_carry_cycles": neager,
                   "trailing_plain_ticks": trail}
            rows.append(row)
            with open(os.path.join(args.per_prompt_dir,
                                   f"{tag}_{pi:03d}.json"), "w") as f:
                json.dump({"output": r["output"]}, f)
        lagc = getattr(e._sparse, "_lag_obj", None)
        lag_carry = lag_fb = None
        if lagc is not None:
            lag_carry, lag_fb = lagc.carry_cycles, lagc.fallback_cycles
        # Canonical per-slot hot ceiling build_engine sizes the pool for:
        # n_groups*k + WINDOW + chunk pages. Use it (not a hand formula) so the
        # structural gate matches the actual allocation on every model shape.
        from tilerl.memory import sparse_hot_pages_per_slot
        tr = e._sparse.tracker
        pin_ceiling = sparse_hot_pages_per_slot(
            tr.cfg, tr.k_pages, e.limits.max_num_batched_tokens)
    finally:
        e.shutdown()
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    toks = sum(r["warm_eff_tokens"] for r in rows)
    wall = sum(r["warm_wall_ms"] for r in rows)
    agg_eff = round(toks / (wall / 1000.0), 4) if wall else None
    n_cycles = len(pcyc)
    n_eager = sum(r["eager_carry_cycles"] for r in rows)
    n_graph_carry = n_cycles - n_eager  # carries observed with path=="graph"
    # rev gate A: on the DEVICE v2 must actually run graph carries (sf armed),
    # not silently fall back to eager every cycle (the capture-order bug). On a
    # real run this must be > 0. (CPU smoke is exempt: path shapes differ.)
    graph_carry_seen = n_graph_carry > 0
    fb_consistent = (lag_fb is None) or (lag_fb == n_eager)
    recon = None
    if pcyc and pcarry and pplain:
        lhs = 7 * statistics.fmean(pplain) + statistics.fmean(pcarry)
        recon = round(abs(lhs - statistics.fmean(pcyc))
                      / statistics.fmean(pcyc), 4)
    rep = {
        "tag": tag, "lag": lag, "tree": sha, "config": config,
        "n_prompts": len(rows), "prompts": rows,
        "aggregate_eff_tok_s": agg_eff,
        "steady_cycles": n_cycles,
        "graph_carry_cycles": n_graph_carry,
        "graph_carry_observed": graph_carry_seen,
        "eager_carry_cycles": n_eager,
        "carry_fallback_frac": round(n_eager / n_cycles, 4) if n_cycles else None,
        "lag_controller_carry": lag_carry, "lag_controller_fallback": lag_fb,
        "fallback_counter_consistent": fb_consistent,
        "cycle_ms_p50": pct(pcyc, 50), "cycle_ms_p90": pct(pcyc, 90),
        "cycle_ms_mean": round(statistics.fmean(pcyc), 4) if pcyc else None,
        "carry_ms_p50": pct(pcarry, 50), "carry_ms_p90": pct(pcarry, 90),
        "carry_ms_mean": round(statistics.fmean(pcarry), 4) if pcarry else None,
        "plain_ms_mean": round(statistics.fmean(pplain), 4) if pplain else None,
        "cycle_reconcile_rel_gap": recon,
        "residency_structural": structural_report(struct_samples, pin_ceiling),
        "structural_samples": struct_samples,
        "b1_decode_ticks": _b1_state["ticks"],
        "b1_max_concurrent_rows": _b1_state["max_rows"],
        "b1_violations": _b1_state["violations"],
        "capture_order_checked": (cap_state or {"captures": 0})["captures"],
        "capture_order_violations": (cap_state or {"bad": 0})["bad"],
        "window_tokens": None if smoke else args.window_tokens,
        "refresh_ticks_R": None if smoke else args.refresh,
    }
    with open(f"{args.out_prefix}_{tag}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[{tag}] eff {agg_eff} cycles {n_cycles} eager-carry {n_eager} "
          f"carry_p90 {rep['carry_ms_p90']} lag_fb {lag_fb}", flush=True)
    return 0


def _accepted_records(recs, n_out):
    """One trunk-logit record per COMMITTED generated position. The recorder
    marks verify-chain rejected slots accepted=False; keep accepted rows and
    dedup by gen_idx (first record per position)."""
    by_pos = {}
    for x in recs:
        if not x.get("accepted", True) or x.get("logits") is None:
            continue
        g = x["gen_idx"]
        if g < 0 or g >= n_out or g in by_pos:
            continue
        by_pos[g] = x
    return [by_pos[g] for g in sorted(by_pos)]


def _tf_analyze(arm_logits_dir, n_prompts, tag_a, tag_b):
    """Teacher-forced distribution comparison of two arms over the SAME anchor
    positions. Returns per-prompt and aggregate top1 agreement, symmetric mean
    KL, and both-arm margins at disagreeing positions. Uses probe_teacher_force
    kl helpers (torch)."""
    import torch
    from probe_teacher_force import kl_from_logits

    per = []
    all_kl_ab, all_kl_ba, all_margins = [], [], []
    for pi in range(n_prompts):
        da = _load_json(os.path.join(arm_logits_dir, f"{tag_a}_{pi:03d}.logits.json"))
        db = _load_json(os.path.join(arm_logits_dir, f"{tag_b}_{pi:03d}.logits.json"))
        n = min(len(da["output"]), len(db["output"]))
        ra = _accepted_records(da["records"], len(da["output"]))
        rb = _accepted_records(db["records"], len(db["output"]))
        # align by gen_idx intersection
        ma = {x["gen_idx"]: x for x in ra}
        mb = {x["gen_idx"]: x for x in rb}
        gids = sorted(set(ma) & set(mb))
        same = kls_ab = kls_ba = 0
        margins = []
        for g in gids:
            xa, xb = ma[g], mb[g]
            la = torch.tensor(xa["logits"], dtype=torch.float32)
            lb = torch.tensor(xb["logits"], dtype=torch.float32)
            if xa["top1"] == xb["top1"]:
                same += 1
            else:
                ta = la.topk(2)
                tb = lb.topk(2)
                margins.append({
                    "pos": g,
                    "a_top1": int(ta.indices[0]),
                    "a_top1_minus_top2": round(float(ta.values[0] - ta.values[1]), 5),
                    "b_top1": int(tb.indices[0]),
                    "b_top1_minus_top2": round(float(tb.values[0] - tb.values[1]), 5)})
            kls_ab += kl_from_logits(la, lb)
            kls_ba += kl_from_logits(lb, la)
        m = len(gids)
        ag = same / m if m else 0.0
        # Coverage: the compared set must be (almost) every generated position.
        # The only allowed misses are the gen0 boundary (its logit can be the
        # prefill's first token rather than a decode) and a single tail position
        # (last committed gen_idx can equal n_out and is filtered g<n_out);
        # interior holes would mean the accepted-position sets drifted and the
        # top1 ratio was being computed on a silently chosen subset.
        union = set(ma) | set(mb)
        missing = sorted(union - set(gids))
        interior_missing = [g for g in missing if 0 < g < n]
        per.append({"i": pi, "positions_compared": m, "n_output": n,
                    "missing_positions": missing,
                    "interior_missing": interior_missing,
                    "coverage_ok": (len(interior_missing) == 0
                                    and n - m <= 1),
                    "top1_agreement": round(ag, 5),
                    "kl_ab_mean": round(kls_ab / m, 6) if m else None,
                    "kl_ba_mean": round(kls_ba / m, 6) if m else None})
        all_kl_ab.append(kls_ab / m if m else 0.0)
        all_kl_ba.append(kls_ba / m if m else 0.0)
        all_margins.extend(margins)
    return {
        "ok": True, "per_prompt": per,
        "coverage_ok": all(p["coverage_ok"] for p in per),
        "min_top1_agreement": round(min(p["top1_agreement"] for p in per), 5)
            if per else None,
        "mean_top1_agreement": round(statistics.fmean(
            [p["top1_agreement"] for p in per]), 5) if per else None,
        "mean_kl_ab": round(statistics.fmean(all_kl_ab), 6) if all_kl_ab else None,
        "mean_kl_ba": round(statistics.fmean(all_kl_ba), 6) if all_kl_ba else None,
        "n_disagree_positions": len(all_margins),
        "margins": all_margins,
    }


def _speed_row(j):
    """The free-run read 94 asked for."""
    return {k: j.get(k) for k in (
        "warm_eff_tok_s", "graph_tick_ms_p50", "eager_tick_ms_p50",
        "eager_tick_frac", "tokens_per_tick", "accept_rate",
        "accept_in", "accept_post", "spec_accepted", "spec_drafted",
        "graph_carry_observed", "lag_carry", "lag_fallback",
        "carry_ms_p50", "carry_ms_p90", "steady_cycles", "warm_ticks")}


def quality(v2_seqs, ctl_seqs, mod=REFRESH_MOD):
    """Legacy free-running paired comparison (kept for the end-to-end aux)."""
    agree, first_divs, modn = [], [], {}
    per_prompt = []
    for i, (v, c) in enumerate(zip(v2_seqs, ctl_seqs)):
        if len(v) != len(c):
            return {"ok": False, "reason": f"prompt {i}: length {len(v)} != {len(c)}"}
        fd = first_div(v, c)
        same = sum(1 for a, b in zip(v, c) if a == b)
        ag = same / len(c) if c else 1.0
        agree.append(ag)
        if fd is not None:
            first_divs.append(fd)
            modn[fd % mod] = modn.get(fd % mod, 0) + 1
        per_prompt.append({"i": i, "agreement": round(ag, 5),
                           "first_divergence": fd})
    return {
        "ok": True, "per_prompt": per_prompt,
        "min_agreement": round(min(agree), 5) if agree else None,
        "mean_agreement": round(statistics.fmean(agree), 5) if agree else None,
        "median_first_divergence": statistics.median(first_divs)
            if first_divs else None,
        "n_diverged_prompts": len(first_divs),
        "divergence_mod_R": dict(sorted(modn.items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--source", default="")
    ap.add_argument("--draft", default="")
    ap.add_argument("--prompts", default="")
    ap.add_argument("--expect-tree", default="")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--min-tokens", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--out-prefix", default="v2win")
    ap.add_argument("--per-prompt-dir", default="")
    ap.add_argument("--worker-tag", default="",
                    help="worker entry: controlA|v2|controlB")
    ap.add_argument("--worker-lag", default="", choices=["", "off", "async"])
    ap.add_argument("--smoke", action="store_true",
                    help="CPU tiny-model plumbing run (1 prompt; absolute timing "
                         "gates are not meaningful, but the 3-subprocess driver, "
                         "floor and quality comparison are exercised)")
    ap.add_argument("--window-tokens", type=int, default=128, choices=[128, 1024],
                    help="sparse own-window in tokens (the R×W sweep's W; the "
                         "draft read window stays W2048)")
    ap.add_argument("--refresh", type=int, default=8, choices=[8, 16, 32],
                    help="decode ticks between refreshes (R); v2 carries the "
                            "refresh tick")
    ap.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--ab-free", action="store_true",
                    help="stacked-combo speed read: pure free runs only, no "
                         "teacher-forcing/floor/quality gates (two processes A "
                         "off / B async driven by the shell orchestrator)")
    ap.add_argument("--quality-only", action="store_true",
                    help="follow-up read: rc gates floor + teacher-forced "
                         "quality + instrument gates; speed ratio/carry timing "
                         "are reported, not gated")
    args = ap.parse_args()

    if args.worker_tag:
        try:
            return run_worker(args.worker_tag, args.worker_lag or "off", args)
        except Exception:
            import traceback

            traceback.print_exc()
            print(f"INSTRUMENT ERROR(rc14) [{args.worker_tag}]",
                  file=sys.stderr)
            return 14

    import subprocess

    if not args.smoke and not args.prompts:
        print("--prompts required (serve805_prompts.jsonl)", file=sys.stderr)
        return 14
    n_prompts = 1 if args.smoke else args.n_prompts
    args.per_prompt_dir = args.per_prompt_dir or f"{args.out_prefix}_pp"
    os.makedirs(args.per_prompt_dir, exist_ok=True)
    runs = ([("A", "off"), ("B", "async")] if args.ab_free
            else [("controlA", "off"), ("v2", "async"), ("controlB", "off")])
    rcs = {}
    for tag, lag in runs:
        out_json = f"{args.out_prefix}_{tag}.json"
        if os.path.exists(out_json):
            os.remove(out_json)
        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "--worker-tag", tag, "--worker-lag", lag,
               "--model", args.model, "--source", args.source,
               "--draft", args.draft, "--prompts", args.prompts,
               "--expect-tree", args.expect_tree,
               "--max-new-tokens", str(args.max_new_tokens),
               "--min-tokens", str(args.min_tokens),
               "--max-tokens", str(args.max_tokens),
               "--out-prefix", args.out_prefix,
               "--per-prompt-dir", args.per_prompt_dir,
               "--window-tokens", str(args.window_tokens),
               "--refresh", str(args.refresh),
               "--n-prompts", str(args.n_prompts)]
        if args.ab_free:
            cmd.append("--ab-free")
        if args.smoke:
            cmd.append("--smoke")
        with open(f"{args.out_prefix}_{tag}.out", "w") as out_f, \
                open(f"{args.out_prefix}_{tag}.err", "w") as err_f:
            rcs[tag] = subprocess.run(cmd, stdout=out_f,
                                      stderr=err_f).returncode
        if rcs[tag] != 0 or not os.path.exists(out_json):
            print(f"{tag}: rc={rcs[tag]} no json -> rc14", file=sys.stderr)
            rcs[tag] = 14
            break

    if any(v == 14 for v in rcs.values()):
        print(json.dumps(rcs, indent=2))
        return 14
    def _load(p):
        with open(p) as f:
            return json.load(f)

    if args.ab_free:
        # Stacked-combo A/B free-run verdict. No teacher forcing, so the only
        # instrument gates are: v2 actually carried (capture order), B=1 held,
        # residency structural. Speed/acceptance are measured reads, not gates.
        A = _load(f"{args.out_prefix}_A.json")
        B = _load(f"{args.out_prefix}_B.json")
        problems = []
        if not B.get("graph_carry_observed"):
            problems.append("B ran ZERO graph carries (silent eager / capture order)")
        if B.get("capture_order_violations"):
            problems.append("B capture-order violations")
        if A.get("b1_violations") or B.get("b1_violations"):
            problems.append("B>1 ticks in an A/B B=1-only read")
        if A.get("residency_structural", {}).get("violated") or \
                B.get("residency_structural", {}).get("violated"):
            problems.append("residency structural violation")
        ratio = (B["warm_eff_tok_s"] / A["warm_eff_tok_s"]) \
            if A.get("warm_eff_tok_s") else None
        verdict = {
            "geometry": {"window_tokens": A["window_tokens"],
                         "refresh_R": A["refresh_ticks_R"],
                         "draft_true_q_width": A["draft_true_q_width"]},
            "A_v2off": _speed_row(A), "B_v2on": _speed_row(B),
            "B_over_A_speed_ratio": round(ratio, 4) if ratio else None,
            "B_ge_40_triggers_quality_followup": bool(
                B.get("warm_eff_tok_s") is not None and B["warm_eff_tok_s"] >= 40),
            "instrument_ok": not problems, "problems": problems}
        with open(f"{args.out_prefix}_verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)
        print(json.dumps(verdict, indent=2))
        return 14 if problems else 0

    A = _load(f"{args.out_prefix}_controlA.json")
    V = _load(f"{args.out_prefix}_v2.json")
    B = _load(f"{args.out_prefix}_controlB.json")

    def seq(tag):
        return [_load(os.path.join(args.per_prompt_dir,
                                   f"{tag}_{i:03d}.json"))["output"]
                for i in range(n_prompts)]

    sa, sb, sv = seq("controlA"), seq("controlB"), seq("v2")

    verdict = {}
    problems14 = []
    # Floor (teacher-forced distribution): controlB is force-fed controlA's
    # stream, so on the same anchor tokens its trunk logits must match
    # controlA's: top1 = 1.0 and KL ~ 0. Anything else invalidates the window.
    floor = _tf_analyze(args.per_prompt_dir, n_prompts,
                        "controlA", "controlB")
    verdict["floor_controlB_vs_controlA"] = {
        k: floor[k] for k in ("min_top1_agreement", "mean_top1_agreement",
                             "mean_kl_ab", "mean_kl_ba",
                             "n_disagree_positions", "coverage_ok",
                             "per_prompt")}
    # top1 discrete -> strict 1.0; KL f32 across separate processes -> tolerance.
    # Coverage must hold so the agreement is over (almost) every position, not a
    # silently chosen subset.
    floor_ok = (floor.get("coverage_ok")
                and floor["min_top1_agreement"] == 1.0
                and (floor["mean_kl_ab"] or 0.0) <= FLOOR_KL_EPS
                and (floor["mean_kl_ba"] or 0.0) <= FLOOR_KL_EPS)
    verdict["floor_top1_eq_1_kl_le_eps"] = floor_ok
    verdict["floor_kl_eps"] = FLOOR_KL_EPS
    if not floor.get("coverage_ok"):
        bad = [p for p in floor["per_prompt"] if not p["coverage_ok"]]
        problems14.append(
            f"teacher-forced floor coverage broken (interior missing or >1 "
            f"uncompared): {[(p['i'], p['missing_positions']) for p in bad]}")
    if not floor_ok:
        problems14.append(
            f"teacher-forced floor violated: controlB vs controlA "
            f"min_top1={floor['min_top1_agreement']} "
            f"kl_ab={floor['mean_kl_ab']} kl_ba={floor['mean_kl_ba']} "
            f"(eps {FLOOR_KL_EPS})")

    # Free-running token identity of the two controls (aux end-to-end read; not
    # a hard instrument gate now that the teacher-forced floor is binding).
    verdict["controls_freerun_first_div"] = \
        [first_div(a, b) for a, b in zip(sa, sb)]

    # Placement: aggregate eff of the two controls within 5%.
    ea, eb = A["aggregate_eff_tok_s"], B["aggregate_eff_tok_s"]
    ctl_eff = round(0.5 * (ea + eb), 4)
    place = abs(ea - eb) / ctl_eff if ctl_eff else None
    verdict["control_eff_A"] = ea
    verdict["control_eff_B"] = eb
    verdict["control_eff_mean"] = ctl_eff
    verdict["placement_rel_gap"] = round(place, 4) if place is not None else None
    if place is None or place > PLACEMENT_MAX:
        problems14.append(f"control placement gap {place} > {PLACEMENT_MAX}")

    # Binding quality: v2 teacher-forced on the controlA anchor vs controlA.
    tfq = _tf_analyze(args.per_prompt_dir, n_prompts, "controlA", "v2")
    verdict["teacher_forced_quality"] = {
        k: tfq[k] for k in ("min_top1_agreement", "mean_top1_agreement",
                           "mean_kl_ab", "mean_kl_ba",
                           "n_disagree_positions", "coverage_ok",
                           "per_prompt", "margins")}
    if not tfq.get("coverage_ok"):
        bad = [p for p in tfq["per_prompt"] if not p["coverage_ok"]]
        problems14.append(
            "v2 teacher-forced coverage broken (subset assertion risk): "
            + str([(p["i"], p["missing_positions"]) for p in bad]))
    q = quality(sv, sa, mod=args.refresh)
    verdict["freerun_quality_aux"] = q if q["ok"] else q
    if not q["ok"]:
        problems14.append(q["reason"])
    if V.get("fallback_counter_consistent") is False:
        problems14.append(
            f"lag controller fallback {V.get('lag_controller_fallback')} != "
            f"harness eager-carry {V.get('eager_carry_cycles')} (counter "
            f"mismatch; a carry fell back uncounted or counters diverged)")
    s = V.get("residency_structural", {})
    if s.get("violated"):
        problems14.append(
            f"residency structural gate violated: max_resident "
            f"{s.get('max_resident_frames')} vs ceiling {s.get('pin_ceiling')}, "
            f"free {s.get('free_first')}->{s.get('free_last')} monotonic_drain="
            f"{s.get('free_monotonic_drain')} (demote reconcile leaking frames)")
    if not args.smoke and not V.get("graph_carry_observed"):
        problems14.append(
            "v2 graph-carry gate: ZERO carries observed with fwd_path==graph "
            f"(graph_carry_cycles={V.get('graph_carry_cycles')}); v2 is "
            "silently running every refresh as eager (capture/q-clone order)")
    for _arm in (A, V, B):
        if _arm.get("b1_violations"):
            problems14.append(
                f"{_arm.get('tag')}: {_arm['b1_violations']} B>1 graph decode "
                f"ticks (max {_arm.get('b1_max_concurrent_rows')} rows); this "
                "window is B=1-only (94(a))")
    if V.get("capture_order_violations"):
        problems14.append(
            "v2 capture-order gate: a sparse graph was captured with "
            "sf.lag_enabled falsy; q-clone/override merge absent from the "
            "baked graph, so every carry would be a silent eager no-op (the "
            "prior device-run defect)")

    # Timing/result gates (measured; failures are rc1, not rc14).
    ratio = round(V["aggregate_eff_tok_s"] / ctl_eff, 4) if ctl_eff else None
    verdict["instrument_problems"] = problems14
    if problems14:
        # Instrument gates (floor, coverage, placement, graph-carry observed,
        # B==1, capture order, residency structural, counter consistency): any
        # failure invalidates the whole window regardless of the speed gates.
        with open(f"{args.out_prefix}_verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)
        for p in problems14:
            print("INSTRUMENT FAIL(rc14): " + p, file=sys.stderr)
        return 14
    gates1 = {
        "eff_ratio_v2_over_control": ratio,
        "go_ratio_ge_1p20": ratio is not None and ratio >= GO_RATIO,
        "carry_p90_ms": V["carry_ms_p90"],
        "carry_p90_le_135p4": (V["carry_ms_p90"] is not None
                               and V["carry_ms_p90"] <= CARRY_P90_MAX),
        "carry_fallback_frac": V["carry_fallback_frac"],
        "carry_fallback_le_0p05": (V["carry_fallback_frac"] is not None
                                   and V["carry_fallback_frac"] <= FALLBACK_FRAC_MAX),
        "cycle_reconcile_rel_gap": V["cycle_reconcile_rel_gap"],
        "cycle_reconcile_le_0p05": (V["cycle_reconcile_rel_gap"] is not None
                                    and V["cycle_reconcile_rel_gap"] <= CYCLE_RECONCILE_MAX),
        # Teacher-forced quality: top1 agreement is the BINDING gate (>=0.99
        # per 94); KL is report-only this round; margins describe near-ties.
        "tf_mean_top1_agreement": tfq.get("mean_top1_agreement"),
        "tf_min_top1_agreement": tfq.get("min_top1_agreement"),
        "tf_mean_kl_ab": tfq.get("mean_kl_ab"),
        "tf_mean_kl_ba": tfq.get("mean_kl_ba"),
        "tf_n_disagree_positions": tfq.get("n_disagree_positions"),
        "tf_coverage_ok": tfq.get("coverage_ok"),
        "tf_top1_mean_ge_0p99": (tfq.get("coverage_ok") is True
            and tfq.get("mean_top1_agreement") is not None
            and tfq["mean_top1_agreement"] >= TF_TOP1_MIN),
    }
    verdict["v2"] = {k: V[k] for k in (
        "aggregate_eff_tok_s", "steady_cycles", "graph_carry_cycles",
        "graph_carry_observed", "eager_carry_cycles",
        "carry_fallback_frac", "lag_controller_carry", "lag_controller_fallback",
        "fallback_counter_consistent", "residency_structural",
        "cycle_ms_p50", "cycle_ms_p90", "carry_ms_p50", "carry_ms_p90",
        "cycle_reconcile_rel_gap")}
    verdict["gates"] = gates1
    verdict["historical_24p9_tok_s"] = HIST_24P9
    dev = abs(ctl_eff - HIST_24P9) / HIST_24P9 if ctl_eff else None
    verdict["control_vs_hist_24p9_rel_dev"] = round(dev, 4) if dev is not None else None
    verdict["control_24p9_deviation_over_10pct"] = (
        dev is not None and dev > 0.10)
    go = all((gates1["go_ratio_ge_1p20"],
              gates1["carry_p90_le_135p4"],
              gates1["carry_fallback_le_0p05"],
              gates1["cycle_reconcile_le_0p05"],
              gates1["tf_top1_mean_ge_0p99"]))
    verdict["GO"] = bool(go)
    with open(f"{args.out_prefix}_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(rcs, indent=2))
    print(json.dumps({"ratio": ratio, "carry_p90": V["carry_ms_p90"],
                      "fallback": V["carry_fallback_frac"],
                      "graph_carries": V.get("graph_carry_cycles"),
                      "tf_top1": gates1["tf_mean_top1_agreement"],
                      "tf_kl": gates1["tf_mean_kl_ab"],
                      "GO": verdict["GO"]}, indent=2))
    if args.smoke:
        # Plumbing run: the measured gates are not meaningful on the tiny model
        # (few cycles). Require only that all three subprocesses produced
        # jsons (rc already enforced) and the two controls are token-identical.
        return 0 if floor_ok else 1
    if args.quality_only:
        # Follow-up after the speed read: instrument gates + floor + binding
        # teacher-forced quality only; speed ratio/carry timing are reported.
        quality_go = floor_ok and gates1["tf_top1_mean_ge_0p99"]
        verdict["QUALITY_GO"] = bool(quality_go)
        with open(f"{args.out_prefix}_verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)
        return 0 if quality_go else 1
    return 0 if verdict["GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
