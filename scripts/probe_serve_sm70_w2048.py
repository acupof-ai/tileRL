#!/usr/bin/env python3
"""#805 end-to-end serve-window probe: sm70 sparse graph+W2048 vs production eager.

Three arms, each its own subprocess (one 27B model per process; two in a
process OOM the 32G card):

  baseline        production, verbatim: depth 1 + draft window W2048 (today's
                  run_serve_prod.sh ships --draft-attn-window-tokens 2048) +
                  sparse_min_tokens=8192, which gates the captured sparse tick
                  off. The graph/ref arms change ONLY the guard + min_tokens;
                  the window is present on both sides and cancels in the diff.
  ref_eager_w2048 probe tree: sparse_min_tokens=0 + W2048 + d1, built with the
                  sparse decode graph ARMED then forced eager after build
                  (graph_on=False). Identical to the graph arm down to the
                  build call — decode_graph is the ONLY effective difference.
  graph_w2048     same build as ref, sparse graph armed and replaying.

Why the reference is a third arm and not baseline: baseline differs from graph
by sparse_min_tokens 8192 vs 0, which changes the PREFILL sparse routing, so
baseline can legitimately commit different tokens and is not a correctness
reference. (Draft W does NOT change which tokens greedy commits — speculation is
lossless under greedy verify, only how many are accepted per forward — so W is
not the reason. W is also identical across all three arms anyway: production
already runs W2048.)

Answers on REAL long prompts at ~32k with the production cold tier:
  1. CORRECTNESS — errors/2026-09-17 gate. A sparse decode graph lazily captured
     at a new cmax bucket once returned a wrong FIRST token on its first replay
     (6/6, both widths, open). Every time run_decode_graph lazily creates a graph
     for a new (B,W,cmax_bucket,own_w) key, THAT call is also its first replay;
     the recorder snapshots the tokens it commits and aligns them by decode
     position against ref_eager_w2048 at the same position on the SAME prompt
     (same temp 0, seed 0, max_new — the sequences are position-aligned). EVERY
     new-key event is gated, and the buckets are those ACTUALLY triggered — no
     hardcoded set: a real 20-40k prompt starts decode already past 512/1024
     (first-tick cmax = prompt_tokens//16 - 7), so those parity-window buckets
     can never recur here. Stage-0 sufficiency: a first replay at bucket >=
     MIN_NEW_BUCKET (4096) — the level no earlier window verified (a ~33k prompt
     transitions 2048 -> 4096 a few tokens into generation; a 40k prompt starts
     at 4096). A path=graph occupancy fraction cannot catch a wrong
     token — this is a value comparison.
     Length precondition: len(ref) == len(graph output) is asserted BEFORE any
     token compare; unequal lengths are rc14 (instrument divergence), never
     reported as a 09-17 token mismatch.
  2. CONFIG IS ARMED — graph/ref arms: _sparse_graph_on True after build and no
     "auto-disabled" warning (the real guard evidence); ref then forced eager
     and observed to take zero graph ticks. Baseline's graph-off is met by
     sparse_min_tokens=8192 in the build expression, NOT by the sm70 guard, and
     is labelled that way in its config record.
  3. THROUGHPUT (stage 1) — graph_w2048 vs THIS WINDOW's engine baseline only
     (same machine, same model load, separate subprocess; in-engine counters,
     not HTTP). Earlier-window numbers (e.g. the 09-19 read) are background,
     never a comparison column. Effective tok/s = (decode_forwards +
     spec_accepted) over summed sparse-decode-tick walls, first WARMUP_DECODE
     ticks of each prompt excluded; per-forward accept ratio, step-wall p50
     split graph/eager, close ticks reported separately. Graph-arm eager ticks
     are not fraction-gated (the every-8-tick refresh caps graph occupancy at
     8/9 by construction); instead EVERY eager sparse tick must attribute to a
     refresh or a prefill/mixed boundary, and an unattributable one is rc14.

Stages:
  --stage 0  one prompt per arm: per-bucket first-replay gate + config only.
             rc 0 green, 1 mismatch (errors/2026-09-17 still live), 14
             insufficient/instrument error.
  --stage 1  N_PROMPTS prompts per arm (default 30, hard floor 20): throughput
             plus the same gate on every prompt.

Usage (device; the driver spawns the three subprocesses — the graph arm reads
the ref arm's per-prompt reference files, so ref always precedes it):
  python probe_serve_sm70_w2048.py --model qwen38-27b \
      --source $SRC --draft $DRAFT --prompts prompts.jsonl --stage 0 \
      --expect-tree $SHA --out-prefix s0
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import warnings

N_PROMPTS = 30
N_PROMPTS_MIN = 20
WARMUP_DECODE = 16
# Buckets are NOT a hardcoded gate: first-decode cmax = prompt_tokens//16 - 7,
# so a real 20-40k prompt starts at bucket 2048 or 4096 and can never revisit
# 512/1024. Stage-0 sufficiency is arming a bucket no earlier window verified:
# parity windows covered only cmax <= 2048, so this window must see a first
# replay at bucket >= 4096 (a 33k prompt transitions 2048 -> 4096 a few tokens
# into generation; a 40k prompt's first decode tick is already 4096).
MIN_NEW_BUCKET = 4096


class ProbeFail(Exception):
    def __init__(self, msg, rc=14):
        super().__init__(msg)
        self.rc = rc


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def load_prompts(path, tokenizer, want_n, min_tokens, max_tokens):
    """Read the real-prompt file (jsonl with a token/text field, or one record
    per line). Keep prompts within [min,max] tokens; return the first want_n id
    lists. Hard rc14 on shortage — never pad with synthetic prompts."""
    if not path or not os.path.exists(path):
        raise ProbeFail(f"--prompts file not found: {path!r}", rc=14)
    out = []
    with open(path) as pf:
        lines = [ln.strip() for ln in pf if ln.strip()]
    for ln in lines:
        ids = None
        if ln.startswith("{"):
            obj = json.loads(ln)
            for key in ("input_ids", "tokens", "ids", "prompt", "text", "content"):
                v = obj.get(key)
                if isinstance(v, list):
                    ids = [int(x) for x in v]
                    break
                if isinstance(v, str):
                    ids = tokenizer.encode(v)
                    break
        else:
            ids = tokenizer.encode(ln)
        if ids is not None and min_tokens <= len(ids) <= max_tokens:
            out.append(ids)
        if len(out) >= want_n:
            break
    if len(out) < want_n:
        raise ProbeFail(
            f"only {len(out)} prompts in [{min_tokens},{max_tokens}] tokens "
            f"(need {want_n}); widen the range or pass more prompts", rc=14)
    return out


# --------------------------------------------------------------------------- #
# engine construction — one build call; the arm differs only by its three knobs
# --------------------------------------------------------------------------- #
def _finalize_arm(e, arm, armed, disabled, built_graph_on):
    """Config assertions shared by the CUDA and CPU-smoke builds."""
    if armed:
        if not built_graph_on or disabled:
            raise ProbeFail(
                f"{arm}: sparse graph not armed after build "
                f"(on={built_graph_on}, warnings={disabled})", rc=14)
    elif built_graph_on:
        raise ProbeFail("baseline: _sparse_graph_on True with min_tokens=8192", rc=14)
    # Ref arm = graph arm with every sparse tick forced eager: same runtime,
    # same selection/fill, only the captured replay declined.
    if arm == "ref_eager_w2048":
        e._sparse_graph_on = False
    config = {"arm": arm, "built_sparse_graph_on": built_graph_on,
              "forced_eager": arm == "ref_eager_w2048",
              "auto_disabled_warnings": disabled,
              "min_tokens": 0 if armed else 8192,
              "draft_window": 2048}
    if not armed:
        # Baseline's graph-off comes from the min_tokens term of the build
        # expression (`... and not self._sparse_min_tokens`), NOT from the sm70
        # spec guard — do not report this as guard evidence.
        config["graph_off_reason"] = "sparse_min_tokens=8192 in build expression"
    return config


def _tiny_draft(cfg, model):
    """One-layer DraftHead over the tiny CPU trunk (the test-suite builder)."""
    from dataclasses import replace

    from tilerl.model import build_random
    from tilerl.spec import DraftHead

    dcfg = replace(cfg, num_layers=1, full_attn_layers=(0,), fp4=False)
    params = {k: v for k, v in build_random(dcfg, seed=3).params.items()
              if k.startswith("layers.")}
    import torch

    gen = torch.Generator().manual_seed(3)
    h = cfg.hidden_size
    params["fc"] = (torch.randn(h, 2 * h, generator=gen) * 0.02).to(torch.bfloat16)
    params["norm"] = torch.ones(h, dtype=torch.bfloat16)
    params["pre_fc_norm_hidden"] = torch.ones(h, dtype=torch.bfloat16)
    return DraftHead(model, params, num_layers=1, attn_window_tokens=2048)


def build_smoke_engine(arm):
    """CPU tiny end-to-end plumbing check: all three arms build, run a prompt,
    and write json. Not a fidelity gate (tiny vocab, CpuSparseGraph seam) — it
    exists to catch crash-level script bugs in the first second instead of on
    the device. Mirrors build_arm_engine's knobs at tiny scale."""
    from tilerl_kernels.backend import get_backend

    from tilerl.build import build_engine
    from tilerl.config import tiny
    from tilerl.model import build_random

    be = get_backend()
    cfg = tiny()
    armed = arm in ("graph_w2048", "ref_eager_w2048")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = build_random(cfg, seed=11)
        e = build_engine(
            cfg, model, be,
            num_blocks=64, num_slots=4, max_batch=4, max_total_tokens=4096,
            max_num_batched_tokens=512,
            sparse_k=2, sparse_min_tokens=0 if armed else 8192,
            sparse_device_select=True,
            scorer="bounds",
            kv_cold_bytes=1 << 30,
            decode_graph=True, draft=_tiny_draft(cfg, model), spec_depth=1,
        )
        disabled = [str(w.message) for w in caught if "auto-disabled" in str(w.message)]
    config = _finalize_arm(e, arm, armed, disabled, bool(e._sparse_graph_on))
    config["smoke"] = True
    return e, be, config


def build_arm_engine(model_name, source, draft_path, arm):
    from tilerl_kernels.backend import get_backend

    from tilerl import build as build_mod
    from tilerl.build import build_engine, build_model
    from tilerl.spec import load_draft

    build_mod.QWEN38_SOURCE = source
    be = get_backend()
    if be.device.type != "cuda":
        raise ProbeFail("this probe needs CUDA (use --smoke for the CPU check)", rc=14)

    # All three arms run W2048 (production does too); the arms differ only by
    # min_tokens (baseline 8192 vs graph/ref 0) and the ref arm's forced eager.
    armed = arm in ("graph_w2048", "ref_eager_w2048")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg, model = build_model(model_name, seed=0, fuse_projections=True)
        draft = load_draft(model, draft_path, attn_window_tokens=2048)
        e = build_engine(
            cfg, model, be,
            num_slots=4, max_batch=4, max_total_tokens=131072,
            max_num_batched_tokens=512,
            sparse_k=128, sparse_min_tokens=0 if armed else 8192,
            sparse_device_select=True,
            scorer="bounds",
            kv_cold_bytes=int(os.environ.get("H2_COLD_BYTES", str(1 << 30))),
            cold_ssd_path=os.environ.get("H2_COLD_SSD", ""),
            cold_ssd_bytes=int(os.environ.get("H2_COLD_SSD_BYTES", "0")),
            cold_format="f16",
            decode_graph=True, draft=draft, spec_depth=1,
        )
        disabled = [str(w.message) for w in caught if "auto-disabled" in str(w.message)]
    config = _finalize_arm(e, arm, armed, disabled, bool(e._sparse_graph_on))
    return e, be, config


# --------------------------------------------------------------------------- #
# run + instruments
# --------------------------------------------------------------------------- #
def _sampling(max_new):
    from tilerl.engine import SamplingParams

    return SamplingParams(temperature=0.0, max_new_tokens=max_new, seed=0)


def install_capture_recorder(e):
    """Wrap SparseRuntime.run_decode_graph. A call that lazily creates a graph
    for a new (B,W,bucket,own_w) key IS that graph's first replay; snapshot the
    req's committed slice right after raw() (sample/verify already committed).
    Returns (events, state, before_step, unwrap)."""
    runtime = e._sparse
    raw = runtime.run_decode_graph
    events = []
    state = {"rid": None, "before": None}

    def wrapped(reqs, chains=None):
        old = set(runtime.graphs)
        ok = raw(reqs, chains)
        if ok and reqs:
            r = reqs[0]
            if state["rid"] == r.req_id and state["before"] is not None:
                for key in sorted(set(runtime.graphs) - old):
                    B, W, cmax_b, own_w = key
                    events.append({
                        "B": B, "W": W, "bucket": cmax_b, "own_w": own_w,
                        "seq_before": state["before"],
                        "committed": list(r.output[state["before"]:]),
                    })
        return ok

    runtime.run_decode_graph = wrapped

    def before_step(rid):
        state["rid"] = rid
        row = next((x for x in e._running if x.req_id == rid), None)
        state["before"] = len(row.output) if row is not None else None

    def unwrap():
        runtime.run_decode_graph = raw

    return events, before_step, unwrap


def run_prompt(e, ids, max_new, on_decode=None, pre_step=None):
    """Submit one temp-0 prompt, drain to done. For every sparse decode tick
    while the rid is live calls
      on_decode(idx, wall_ms, tm, d_forwards, d_accepted, is_close,
                refresh_after, phase_pre)
    where is_close marks the tick that finished the request in-step (its wall
    carries release_cold_forget/release_blocks/ssd_mmap — excluded from the warm
    window, reported separately), refresh_after is the sparse runtime's
    ticks_since_refresh AFTER the tick (0 exactly when this tick was the
    periodic refresh, which runs eager), and phase_pre is the tick's prefill
    row count (>0 marks a prefill/mixed boundary, also eager). Returns
    {output, decode_ticks, graph_ticks, close_ticks}."""
    rid = e.submit(list(ids), _sampling(max_new))
    tm = e._step_timing
    idx = graph_ticks = close_ticks = 0
    for _ in range(200000):
        live = any(r.req_id == rid for r in e._running)
        if live and pre_step is not None:
            pre_step(rid)
        f0, a0 = e._decode_forwards, e._spec_accepted
        t0 = time.perf_counter()
        e.step()
        wall = (time.perf_counter() - t0) * 1000.0
        alive = any(r.req_id == rid for r in e._running)
        df, da = e._decode_forwards - f0, e._spec_accepted - a0
        if df and tm is not None and tm.fwd_sparse and live:
            is_close = not alive
            close_ticks += int(is_close)
            graph_ticks += int(tm.fwd_path == "graph")
            if on_decode is not None:
                on_decode(idx, wall, tm, df, da, is_close,
                          e._sparse.ticks_since_refresh, tm.phase_pre)
            idx += 1
        if not alive:
            out = list(e.poll().get(rid, []))
            return {"output": out, "decode_ticks": idx,
                    "graph_ticks": graph_ticks, "close_ticks": close_ticks}
    raise ProbeFail(f"rid {rid} did not finish within 200000 steps", rc=14)


# --------------------------------------------------------------------------- #
# per-arm worker
# --------------------------------------------------------------------------- #
def run_worker(arm, args):
    os.environ.setdefault("TILERL_STEP_TIMING", "1")
    os.environ.setdefault("TILERL_STEP_TIMING_SLOW_MS", "0")
    smoke = bool(getattr(args, "smoke", False))
    if not smoke:
        _assert_tree(args.expect_tree)

    if smoke:
        e, be, config = build_smoke_engine(arm)
        # Tiny vocab 320, max_total_tokens 4096: one deterministic ~96-token
        # prompt is enough to drive prefill + several sparse decode captures.
        prompts = [[7 + (i % 300) for i in range(96)]]
        max_new = 24
    else:
        from tilerl.cli import _qwen38_tokenizer

        e, be, config = build_arm_engine(args.model, args.source, args.draft, arm)
        tok = _qwen38_tokenizer()
        want = 1 if args.stage == 0 else args.n_prompts
        prompts = load_prompts(args.prompts, tok, want,
                               args.min_tokens, args.max_tokens)
        max_new = args.max_new_tokens

    results = {"arm": arm, "config": config, "prompts": [], "failures": [],
              "instrument_errors": []}
    tag = "smoke" if smoke else f"stage{args.stage}"
    out_path = f"{args.out_prefix}_{arm}.{tag}.json"
    measure = arm in ("graph_w2048", "baseline") and not smoke
    eff_tokens = 0
    wall_ms = 0.0
    graph_walls, eager_walls, close_walls = [], [], []
    # Graph arm only: every eager sparse tick must be attributable to a refresh
    # (counter reset to 0 after it) or a prefill/mixed boundary (phase_pre>0).
    # An unattributable eager tick means the graph declined for some unknown
    # reason — stronger than an occupancy threshold (the 8/9 refresh ceiling
    # caps graph occupancy at 88.9% by construction, so no fraction is gated).
    eager_attributed = {"refresh": 0, "prefill_boundary": 0}
    unattributed_eager = []

    def dump():
        # Red is an answer: persist evidence even when a later prompt raises.
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    try:
        for pi, ids in enumerate(prompts):
            events = before_step = unwrap = None
            if arm == "graph_w2048":
                events, before_step, unwrap = install_capture_recorder(e)

            acc = [0, 0.0]  # [effective tokens, wall ms] over warm ticks

            def on_decode(idx, w, tm, df, da, is_close, refresh_after, phase_pre,
                          pi=pi, acc=acc):
                if is_close:
                    close_walls.append(w)
                if arm == "graph_w2048" and tm.fwd_path != "graph":
                    if refresh_after == 0:
                        eager_attributed["refresh"] += 1
                    elif phase_pre > 0:
                        eager_attributed["prefill_boundary"] += 1
                    else:
                        unattributed_eager.append(
                            {"prompt": pi, "tick": idx,
                             "refresh_after": refresh_after, "phase_pre": phase_pre})
                if measure and idx >= WARMUP_DECODE and not is_close:
                    acc[0] += df + da
                    acc[1] += w
                    (graph_walls if tm.fwd_path == "graph" else eager_walls).append(w)

            r = run_prompt(e, ids, max_new, on_decode, before_step)
            if unwrap is not None:
                unwrap()
            eff_tokens += acc[0]
            wall_ms += acc[1]

            row = {"i": pi, "n_out": len(r["output"]),
                   "decode_ticks": r["decode_ticks"],
                   "graph_ticks": r["graph_ticks"],
                   "close_ticks": r["close_ticks"]}

            if arm == "ref_eager_w2048":
                # Forced eager must take ZERO graph paths — the single-variable
                # precondition for the graph-vs-ref gate.
                if r["graph_ticks"]:
                    raise ProbeFail(
                        f"ref arm took {r['graph_ticks']} graph ticks despite "
                        f"forced eager", rc=14)
                with open(os.path.join(args.reference_dir, f"ref_{pi:03d}.json"),
                          "w") as f:
                    json.dump({"output": r["output"]}, f)

            if arm == "graph_w2048":
                ref_path = os.path.join(args.reference_dir, f"ref_{pi:03d}.json")
                if not os.path.exists(ref_path):
                    raise ProbeFail(f"prompt {pi}: missing ref {ref_path} "
                                   f"(run ref_eager_w2048 first)", rc=14)
                with open(ref_path) as rf:
                    ref = json.load(rf)["output"]
                # Length precondition BEFORE any token compare: unequal output
                # lengths are an instrument/sequence divergence (rc14), never a
                # 09-17 wrong-token report.
                if len(ref) != len(r["output"]):
                    raise ProbeFail(
                        f"prompt {pi}: ref len {len(ref)} != graph len "
                        f"{len(r['output'])} — cannot value-align; not a 09-17 "
                        f"mismatch", rc=14)
                first_bucket = events[0]["bucket"] if events else None
                gate = []
                new_bucket_seen = False
                transition_seen = False
                for ev in events:
                    sb = ev["seq_before"]
                    got = ev["committed"]
                    want_ids = ref[sb:sb + len(got)]
                    match = got == want_ids
                    gate.append({"bucket": ev["bucket"], "B": ev["B"], "W": ev["W"],
                                 "own_w": ev["own_w"], "seq_before": sb,
                                 "n_committed": len(got), "match": match,
                                 "graph_tokens": got[:8], "ref_tokens": want_ids[:8]})
                    if not match:
                        results["failures"].append(
                            f"prompt{pi} B={ev['B']} W={ev['W']} bucket={ev['bucket']} "
                            f"own_w={ev['own_w']} FIRST replay mismatch: "
                            f"seq_before={sb} graph={got[:6]} ref={want_ids[:6]} "
                            f"(errors/2026-09-17 shape)")
                    # The bucket this window arms for the first time. A 33k
                    # prompt transitions into it during generation; a 37.6k
                    # prompt's first decode tick is already on it.
                    if ev["bucket"] >= MIN_NEW_BUCKET:
                        new_bucket_seen = True
                    if sb > 0 and first_bucket is not None \
                            and ev["bucket"] > first_bucket:
                        transition_seen = True
                row["first_replays"] = gate
                row["first_decode_bucket"] = first_bucket
                row["buckets_seen"] = sorted({g["bucket"] for g in gate})
                row["generation_bucket_transition"] = transition_seen
                row["new_bucket_armed"] = new_bucket_seen

            results["prompts"].append(row)
            dump()

            if args.stage == 0 and arm == "graph_w2048" and new_bucket_seen:
                break

        # Reported for the graph arm in every mode: every eager sparse tick
        # must be attributable to a refresh or a prefill/mixed boundary.
        if arm == "graph_w2048":
            results["eager_tick_attribution"] = dict(eager_attributed)
            results["unattributed_eager_ticks"] = unattributed_eager

        if measure:
            results["throughput"] = {
                "comparison": "this window's same-machine engine baseline "
                              "(separate subprocess, in-engine counters); "
                              "earlier-window reads are background",
                "warm_window": f"ticks [{WARMUP_DECODE}, end), close tick excluded",
                "warm_effective_tokens": eff_tokens,
                "warm_wall_ms": round(wall_ms, 2),
                "warm_effective_tok_s": round(eff_tokens / (wall_ms / 1000.0), 3)
                    if wall_ms else None,
                "warm_step_p50_ms_graph": round(statistics.median(graph_walls), 3)
                    if graph_walls else None,
                "warm_step_p50_ms_eager": round(statistics.median(eager_walls), 3)
                    if eager_walls else None,
                "warm_graph_ticks": len(graph_walls),
                "warm_eager_ticks": len(eager_walls),
                # The per-request close tick (release_cold_forget/release_blocks/
                # ssd_mmap, 9-12 s at 32k) is NOT warm decode: reported on its
                # own and excluded from the warm window above.
                "close_ticks": len(close_walls),
                "close_step_p50_ms": round(statistics.median(close_walls), 3)
                    if close_walls else None,
            }
    except ProbeFail:
        dump()
        raise
    finally:
        dump()
        e.shutdown()
        if be.device.type == "cuda":
            import torch

            torch.cuda.synchronize()

    print(f"[serve-probe] {arm} {tag} wrote {out_path}", flush=True)

    if not smoke:
        # Observed tick paths: decode=sparse ticks, graph=ticks that replayed.
        # Reported for every arm; the graph arm's fraction is below 1.0 because
        # its refresh ticks run eager. For the baseline the sharp assertion is
        # zero graph ticks (sparse_graph_on=False makes a replay impossible) —
        # with g=0 the fraction is exactly 1.0, so no redundant threshold. A
        # dense zero-tick baseline (CPU tiny shape) proves nothing, device only.
        d_tot = sum(p.get("decode_ticks", 0) for p in results["prompts"])
        g_tot = sum(p.get("graph_ticks", 0) for p in results["prompts"])
        frac = (d_tot - g_tot) / d_tot if d_tot else None
        results["sparse_path_split"] = {
            "decode_ticks": d_tot, "graph_ticks": g_tot,
            "eager_sparse_fraction": round(frac, 4) if frac is not None else None}
        baseline_bad = arm == "baseline" and args.stage == 1 and (
            d_tot == 0 or g_tot != 0)
        if baseline_bad:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"INSUFFICIENT: baseline sparse split {g_tot} graph / {d_tot} "
                  f"total; production config must run eager sparse with ZERO "
                  f"graph ticks", file=sys.stderr)
            return 14

    if results["failures"]:
        for x in results["failures"][:10]:
            print("  MISMATCH " + x, file=sys.stderr)
        return 1  # the ONLY rc=1 path: the 09-17 gate actually went red
    if arm == "graph_w2048" and unattributed_eager:
        # Every eager sparse tick must be the periodic refresh (counter reset
        # to 0) or a prefill/mixed boundary. An unattributable one means the
        # graph declined for an unknown reason; no occupancy fraction covers
        # that failure mode.
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"INSUFFICIENT: {len(unattributed_eager)} eager sparse tick(s) "
              f"not attributable to refresh/prefill: "
              f"{unattributed_eager[:5]}", file=sys.stderr)
        return 14
    if arm == "graph_w2048":
        n_captures = sum(len(p.get("first_replays", [])) for p in results["prompts"])
        if smoke:
            # Plumbing check: at least one sparse capture must have fired and
            # every first replay aligned with the eager ref. The 4096
            # sufficiency criterion is device-window-specific.
            if n_captures == 0:
                print("INSUFFICIENT(smoke): graph arm recorded zero sparse "
                      "captures", file=sys.stderr)
                return 14
        elif not any(p.get("new_bucket_armed") for p in results["prompts"]):
            seen = sorted({b for p in results["prompts"]
                           for b in p.get("buckets_seen", [])})
            print(f"INSUFFICIENT: no first replay at bucket >= {MIN_NEW_BUCKET} "
                  f"(buckets seen: {seen}); the never-before-verified bucket was "
                  f"not armed — prompt too short or max_new too small?",
                  file=sys.stderr)
            return 14
        if not smoke and args.stage == 1 \
                and len(results["prompts"]) < args.min_prompts:
            print(f"INSUFFICIENT: only {len(results['prompts'])} prompts "
                  f"(floor --min-prompts {args.min_prompts})", file=sys.stderr)
            return 14
        if not smoke and args.stage == 1:
            results["verdict"] = {
                "prompts": len(results["prompts"]),
                "floor_min_prompts": args.min_prompts,
                "first_replay_failures": len(results["failures"]),
                "unattributed_eager_ticks": len(unattributed_eager)}
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    return 0


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _assert_tree(expect):
    import subprocess

    if not expect:
        raise ProbeFail("--expect-tree is required", rc=14)
    sha = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if sha != expect[:8]:
        raise ProbeFail(f"tree {sha} != expected {expect[:8]}", rc=14)


ARMS = ["baseline", "ref_eager_w2048", "graph_w2048"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--worker", default=None, choices=[None] + ARMS)
    ap.add_argument("--source", default="")
    ap.add_argument("--draft", default="")
    ap.add_argument("--prompts", default="", help="jsonl/txt of real long prompts")
    ap.add_argument("--reference-dir", default="", help="dir for ref-arm outputs")
    ap.add_argument("--expect-tree", default="")
    ap.add_argument("--out-prefix", default="serve805")
    ap.add_argument("--stage", type=int, default=0, choices=[0, 1])
    ap.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--min-prompts", type=int, default=N_PROMPTS_MIN,
                    help="stage1 verdict floor (default 20); pass the same value "
                         "as --n-prompts for a deliberately small window")
    ap.add_argument("--min-tokens", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated subset of " + ",".join(ARMS)
                         + "; run in canonical order so ref_eager_w2048 always "
                           "precedes graph_w2048 (graph reads its ref files)")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU tiny end-to-end plumbing check (no CUDA, no real "
                         "prompts, no tree assert); run before any window")
    args = ap.parse_args()

    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in selected if a not in ARMS]
    if bad:
        print(f"unknown arm(s) {bad}; valid: {ARMS}", file=sys.stderr)
        return 14
    if not selected:
        print("--arms must name at least one arm", file=sys.stderr)
        return 14
    # Canonical order regardless of the order given: graph must never run
    # before its reference has been written.
    arms = [a for a in ARMS if a in selected]

    if args.worker:
        try:
            return run_worker(args.worker, args)
        except ProbeFail as exc:
            # Instrument/insufficiency path.
            print(f"INSUFFICIENT(rc14) [{args.worker}]: {exc}", file=sys.stderr)
            return exc.rc
        except Exception:
            # Any other crash (TypeError, build failure, OOM, ...) is an
            # INSTRUMENT failure: rc14, never the rc=1 that means "09-17 gate
            # red". The two must be separable from the exit code alone.
            import traceback

            traceback.print_exc()
            print(f"INSTRUMENT ERROR(rc14) [{args.worker}]: uncaught exception; "
                  f"this is not a 09-17 gate red", file=sys.stderr)
            return 14

    import subprocess

    tag = "smoke" if args.smoke else f"stage{args.stage}"
    # Per-window reference dir: a stale ref file from an earlier window would
    # let graph align against the wrong run and go green, so never reuse the
    # default across windows and refuse a non-empty explicit one.
    args.reference_dir = args.reference_dir or f"{args.out_prefix}_refs"
    if os.path.isdir(args.reference_dir) and any(os.scandir(args.reference_dir)):
        print(f"reference dir {args.reference_dir!r} is not empty; use a fresh "
              f"--reference-dir per window (stale ref files fake a green)",
              file=sys.stderr)
        return 14
    os.makedirs(args.reference_dir, exist_ok=True)
    if "graph_w2048" in arms and "ref_eager_w2048" not in arms:
        # graph needs this window's ref files; running it against an old dir is
        # exactly the fake-green above.
        print("graph_w2048 requires ref_eager_w2048 in the same --arms",
              file=sys.stderr)
        return 14
    rcs = {}
    for arm in arms:
        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "--worker", arm, "--model", args.model, "--source", args.source,
               "--draft", args.draft, "--prompts", args.prompts,
               "--reference-dir", args.reference_dir, "--expect-tree", args.expect_tree,
               "--out-prefix", args.out_prefix, "--stage", str(args.stage),
               "--n-prompts", str(args.n_prompts),
               "--min-prompts", str(args.min_prompts),
               "--min-tokens", str(args.min_tokens),
               "--max-tokens", str(args.max_tokens),
               "--max-new-tokens", str(args.max_new_tokens)]
        if args.smoke:
            cmd.append("--smoke")
        with open(f"{args.out_prefix}_{arm}.{tag}.out", "w") as out_f, \
                open(f"{args.out_prefix}_{arm}.{tag}.err", "w") as err_f:
            rcs[arm] = subprocess.run(cmd, stdout=out_f, stderr=err_f).returncode
    print(json.dumps(rcs, indent=2))
    # 1 means a real 09-17 gate red; 14 means a crash or insufficiency.
    return 1 if 1 in rcs.values() else (14 if 14 in rcs.values() else 0)


if __name__ == "__main__":
    sys.exit(main())
