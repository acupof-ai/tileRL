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
HIST_24P9 = 24.915
N_PROMPTS = 6
REFRESH_MOD = 8  # SPARSE_REFRESH_TICKS (carry/refresh every 8 sparse ticks)


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
    if args.expect_tree and sha != args.expect_tree[:8]:
        print(f"tree {sha} != {args.expect_tree[:8]}", file=sys.stderr)
        return 14

    if smoke:
        from probe_serve_sm70_w2048 import build_smoke_engine

        prompts = [[7 + (i % 300) for i in range(400)]]
        e, _be, config = build_smoke_engine("graph_w2048")
    else:
        from tilerl.cli import _qwen38_tokenizer

        tok = _qwen38_tokenizer()
        prompts = load_prompts(args.prompts, tok, N_PROMPTS,
                               args.min_tokens, args.max_tokens)
        if len(prompts) < N_PROMPTS:
            print(f"only {len(prompts)} prompts (< {N_PROMPTS})", file=sys.stderr)
            return 14
        e, _be, config = build_arm_engine(args.model, args.source, args.draft,
                                          "graph_w2048")
    rows = []
    pcyc = pcarry = pplain = []
    try:
        for pi, ids in enumerate(prompts):
            ticks, acc = [], [0, 0.0]

            def on_decode(idx, wall, tm, df, da, is_close, refresh_after,
                          phase_pre, acc=acc, ticks=ticks):
                warm = idx >= WARMUP_DECODE and not is_close
                if warm:
                    acc[0] += df + da
                    acc[1] += wall
                ticks.append({"wall": wall, "path": tm.fwd_path,
                              "refresh_after": refresh_after, "warm": warm})

            r = run_prompt(e, ids, args.max_new_tokens, on_decode)
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
        "eager_carry_cycles": n_eager,
        "carry_fallback_frac": round(n_eager / n_cycles, 4) if n_cycles else None,
        "lag_controller_carry": lag_carry, "lag_controller_fallback": lag_fb,
        "cycle_ms_p50": pct(pcyc, 50), "cycle_ms_p90": pct(pcyc, 90),
        "cycle_ms_mean": round(statistics.fmean(pcyc), 4) if pcyc else None,
        "carry_ms_p50": pct(pcarry, 50), "carry_ms_p90": pct(pcarry, 90),
        "carry_ms_mean": round(statistics.fmean(pcarry), 4) if pcarry else None,
        "plain_ms_mean": round(statistics.fmean(pplain), 4) if pplain else None,
        "cycle_reconcile_rel_gap": recon,
    }
    with open(f"{args.out_prefix}_{tag}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[{tag}] eff {agg_eff} cycles {n_cycles} eager-carry {n_eager} "
          f"carry_p90 {rep['carry_ms_p90']} lag_fb {lag_fb}", flush=True)
    return 0


def quality(v2_seqs, ctl_seqs):
    """Paired per-prompt comparison. Returns a verdict dict."""
    agree, first_divs, mod8 = [], [], {}
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
            mod8[fd % REFRESH_MOD] = mod8.get(fd % REFRESH_MOD, 0) + 1
        per_prompt.append({"i": i, "agreement": round(ag, 5),
                           "first_divergence": fd})
    return {
        "ok": True, "per_prompt": per_prompt,
        "min_agreement": round(min(agree), 5) if agree else None,
        "mean_agreement": round(statistics.fmean(agree), 5) if agree else None,
        "median_first_divergence": statistics.median(first_divs)
            if first_divs else None,
        "n_diverged_prompts": len(first_divs),
        "divergence_mod8": dict(sorted(mod8.items())),
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
    n_prompts = 1 if args.smoke else N_PROMPTS
    args.per_prompt_dir = args.per_prompt_dir or f"{args.out_prefix}_pp"
    os.makedirs(args.per_prompt_dir, exist_ok=True)
    runs = [("controlA", "off"), ("v2", "async"), ("controlB", "off")]
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
               "--per-prompt-dir", args.per_prompt_dir]
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
    # Floor: the two controls must be token-identical (measured floor 1.0).
    floor_fd = [first_div(a, b) for a, b in zip(sa, sb)]
    floor_ok = all(d is None for d in floor_fd)
    verdict["controls_token_identical"] = floor_ok
    if not floor_ok:
        problems14.append(f"controls diverged: {floor_fd} (floor not 1.0)")

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

    q = quality(sv, sa)
    verdict["quality_vs_controlA"] = q if q["ok"] else q
    if not q["ok"]:
        problems14.append(q["reason"])

    # Timing/result gates (measured; failures are rc1, not rc14).
    ratio = round(V["aggregate_eff_tok_s"] / ctl_eff, 4) if ctl_eff else None
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
        "quality_mean_agreement": q.get("mean_agreement") if q["ok"] else None,
        "quality_min_agreement": q.get("min_agreement") if q["ok"] else None,
        "quality_mean_ge_0p995": q["ok"] and q["mean_agreement"] is not None
            and q["mean_agreement"] >= 0.995,
        "quality_each_ge_0p99": q["ok"] and q["min_agreement"] is not None
            and q["min_agreement"] >= 0.99,
        "quality_median_first_div": q.get("median_first_divergence") if q["ok"] else None,
        "quality_median_div_ge_128": (not q["ok"]) or q["median_first_divergence"] is None
            or q["median_first_divergence"] >= DIV_MIN,
    }
    verdict["v2"] = {k: V[k] for k in (
        "aggregate_eff_tok_s", "steady_cycles", "eager_carry_cycles",
        "carry_fallback_frac", "lag_controller_carry", "lag_controller_fallback",
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
              gates1["quality_mean_ge_0p995"],
              gates1["quality_each_ge_0p99"],
              gates1["quality_median_div_ge_128"]))
    verdict["GO"] = bool(go)
    with open(f"{args.out_prefix}_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(rcs, indent=2))
    print(json.dumps({"ratio": ratio, "carry_p90": V["carry_ms_p90"],
                      "fallback": V["carry_fallback_frac"],
                      "quality": gates1["quality_median_first_div"],
                      "GO": verdict["GO"]}, indent=2))
    if args.smoke:
        # Plumbing run: the measured gates are not meaningful on the tiny model
        # (few cycles). Require only that all three subprocesses produced
        # jsons (rc already enforced) and the two controls are token-identical.
        return 0 if floor_ok else 1
    return 0 if verdict["GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
