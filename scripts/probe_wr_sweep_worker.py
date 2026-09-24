#!/usr/bin/env python3
"""One (W, R) arm of the local-window × refresh-interval sweep — V100 sm70.

One process per arm (one 27B per process is the harness rule; two engines in
one process OOM). Two SEQUENTIAL runs share the one model load:

  free run   production decode, greedy; outputs + speed (the TF run changes
             draft acceptance, so its speed does not count).
  tf run     TeacherForceRecorder feeds the free-run's own token stream back
             as the anchor and records the trunk logits at every committed
             position. For R=1 this is the self-anchor positive control
             (committed tokens must be byte-identical to the free run).

Quality across arms is computed AFTER the sweep by wr_sweep_report.py: each
R arm's TF logits vs the same-W R=1 TF logits (identical anchor prefix) —
top1 agreement >= 0.99 the binding gate, KL/margins report-only.

Production decode config: sparse graph armed, sparse_min_tokens=0, depth-1
draft with the 2048 read window. Patched WITHOUT touching src, before the
engine is built:

  WINDOW_TOKENS / WINDOW_PAGES
      sparse_index owns the constants; sparse_engine imported WINDOW_PAGES BY
      VALUE at module load (own_bound/graph key), so both modules are patched.
  SPARSE_REFRESH_TICKS
      sparse_engine defines it; sparse_runtime imports the module attr inside
      its functions, so that one value is the patch point.

Structural gate (free run): captured graph keys carry own_w = pages + 1;
R=1 captures ZERO graphs and declines every attempt; R>1 declines fire at
counter R-1. Each prompt's report lands as it finishes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from probe_teacher_force import TeacherForceRecorder


def patch_geometry(window_tokens: int, refresh: int) -> int:
    from tilerl import sparse_engine, sparse_index

    pages = window_tokens // sparse_index.BLOCK_TOKENS
    sparse_index.WINDOW_TOKENS = window_tokens
    sparse_index.WINDOW_PAGES = pages
    sparse_engine.WINDOW_PAGES = pages
    sparse_engine.SPARSE_REFRESH_TICKS = refresh
    print(f"PATCH window_tokens={window_tokens} window_pages={pages} R={refresh}",
          flush=True)
    return pages


def load_prompts(path: str, tokenizer, want_n: int) -> list[list[int]]:
    with open(path) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    out = []
    for r in rows:
        if "input_ids" in r:
            out.append([int(x) for x in r["input_ids"]])
        elif "text" in r:
            out.append(tokenizer.encode(r["text"]))
    if len(out) < want_n:
        raise SystemExit(f"only {len(out)} prompts, need {want_n}: {path}")
    return out[:want_n]


def run_one_prompt(e, rid, tag_decline) -> tuple[list[int], dict]:
    """Step a submitted request to completion. Returns (output tokens, stats).
    tag_decline[r] flips True on the step when the graph declined for the
    refresh reason, splitting graph vs eager-refresh tick time."""
    f0 = e._decode_forwards
    acc0 = e._spec_accepted
    decoding = False
    graph_ms: list[float] = []
    eager_ms: list[float] = []
    wall0 = None
    out = None
    cuda = torch.cuda.is_available()
    for _ in range(200000):
        tag_decline["v"] = False
        if cuda:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        else:
            w0 = time.perf_counter()
        e.step()
        if cuda:
            t1.record()
            torch.cuda.synchronize()
        row = next((r for r in e._running if r.req_id == rid), None)
        if row is not None and row.phase == 2:
            if not decoding:
                decoding = True
                # P0 control: every prompt must enter decode at refresh phase 0
                # (the worker zeroed ticks_since_refresh before submit). On the
                # unpatched tree a new request inherits the prior request's
                # phase, contaminating per-prompt comparisons.
                ph = e._sparse.ticks_since_refresh
                print(f"PHASE_AT_FIRST_DECODE rid={rid} phase={ph}", flush=True)
                assert ph == 0, f"refresh phase {ph} != 0 at first decode"
                wall0 = time.perf_counter()
                f0 = e._decode_forwards
                acc0 = e._spec_accepted
            ms = (t0.elapsed_time(t1) if cuda
                  else (time.perf_counter() - w0) * 1000.0)
            (eager_ms if tag_decline["v"] else graph_ms).append(ms)
        # take() pops _finished; do NOT also poll() — poll clears the finished
        # dict, so a following take() would never see the completed request.
        out = e.take(rid)
        if out is not None:
            break
    if out is None:
        raise SystemExit(f"request {rid} never finished")
    wall = time.perf_counter() - wall0
    fwd = e._decode_forwards - f0
    accepted = e._spec_accepted - acc0
    return out, {
        "generated": len(out),
        "graph_ticks": len(graph_ms), "eager_refresh_ticks": len(eager_ms),
        "graph_tick_ms_mean": round(sum(graph_ms) / max(len(graph_ms), 1), 3),
        "eager_tick_ms_mean": round(sum(eager_ms) / max(len(eager_ms), 1), 3),
        "wall_s": round(wall, 3),
        "decode_forwards": fwd, "spec_accepted": accepted,
        "eff_tok_s": round((fwd + accepted) / wall, 3),
    }


def neg_anchor_run(e, prompts, args, decline_flag) -> int:
    """Red control: R=1 self-feed with the anchor shifted. The leg must FAIL
    the committed-output identity ('forced output != anchor'). Exits 0 only if
    the red is observed; 1 if the instrument accepts the shifted anchor (a
    vacuous gate)."""
    from tilerl.engine import SamplingParams

    red = False
    for ids in prompts:
        e._sparse.ticks_since_refresh = 0
        rid0 = e.submit(list(ids), SamplingParams(
            temperature=0.0, max_new_tokens=args.neg_tokens, seed=0))
        base, _ = run_one_prompt(e, rid0, decline_flag)
        shifted = {0: base[args.neg_anchor_offset:]}
        e._sparse.ticks_since_refresh = 0
        rec = TeacherForceRecorder(e, anchors=shifted, record_full_logits=False).install()
        rid1 = e.submit(list(ids), SamplingParams(
            temperature=0.0, max_new_tokens=args.neg_tokens, seed=0))
        out, _ = run_one_prompt(e, rid1, decline_flag)
        rec.uninstall()
        if out != shifted[0]:
            red = True
            print(f"NEG-OK prompt: forced output {len(out)} != shifted anchor "
                  f"{len(shifted[0])}; p0 committed={out[0] if out else None} "
                  f"anchor0={shifted[0][0]}", flush=True)
        else:
            print("NEG-BAD prompt: shifted anchor accepted byte-identically",
                  file=sys.stderr)
    e.shutdown()
    print(f"NEG_CONTROL red_observed={red}", flush=True)
    return 0 if red else 1


def floor_kl_check(e, ids, n_tok: int, decline_flag) -> float:
    """R=1 floor: the SAME forced trajectory run twice must produce the same
    per-position logits (perf1's cutover noise floor, symmetric KL <= 1e-4).

    A record-only free run cannot be the reference: under verification its
    draft-acceptance (n_ok, GDN state adoption) follows the real draws, while a
    teacher-forced run follows the anchor tokens, so even with identical final
    tokens their committed-position logits are not aligned 1:1. Instead: one
    short free run makes the anchor, then TWO teacher-forced runs over that
    identical anchor; only those two are compared. Returns the max symmetric
    KL seen."""
    from tilerl.engine import SamplingParams

    e._sparse.ticks_since_refresh = 0
    rid0 = e.submit(list(ids), SamplingParams(
        temperature=0.0, max_new_tokens=n_tok, seed=0))
    base, _ = run_one_prompt(e, rid0, decline_flag)

    kept_runs = []
    for _ in range(2):
        e._sparse.ticks_since_refresh = 0
        rec = TeacherForceRecorder(e, anchors={0: base}, record_full_logits=True).install()
        rid = e.submit(list(ids), SamplingParams(
            temperature=0.0, max_new_tokens=n_tok, seed=0))
        out, _ = run_one_prompt(e, rid, decline_flag)
        kept = rec.accepted_positions(0)
        rec.uninstall()
        if out != base:
            raise SystemExit("floor KL: TF output diverged from its anchor")
        kept_runs.append(kept)

    from probe_teacher_force import kl_from_logits
    k0, k1 = kept_runs
    if len(k0) != len(k1):
        raise SystemExit(f"floor KL: run length {len(k0)} != {len(k1)}")
    worst = 0.0
    for a, b in zip(k0, k1):
        worst = max(worst,
                    kl_from_logits(a["logits"], b["logits"]),
                    kl_from_logits(b["logits"], a["logits"]))
    print(f"FLOOR_KL n={len(k1)} max_symmetric_kl={worst:.2e}", flush=True)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-tokens", type=int, required=True, choices=[128, 1024])
    ap.add_argument("--refresh", type=int, required=True, choices=[1, 8, 16, 32])
    ap.add_argument("--prompts", default=os.path.expanduser("~/serve805_prompts.jsonl"))
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--free-only", action="store_true",
                    help="skip the teacher-forced leg (speed-only control arm)")
    ap.add_argument("--source", default=os.environ.get("TILERL_QWEN38_SOURCE", ""))
    ap.add_argument("--draft", default="/home/chenkailun.c/mmlu-assets/model_mtp.safetensors")
    ap.add_argument("--cold-ssd", default="/home/chenkailun.c/sparse_cold_128k.bin")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--anchor-dir", default="",
                    help="dir with ref_NNN.json anchors. R=1 self-feeds its own "
                         "free run (the byte-identical control); every other R "
                         "teacher-forces its W-group R=1 free run")
    ap.add_argument("--neg-anchor-offset", type=int, default=0,
                    help="negative control, R=1 only: shift the self anchor by N "
                         "positions; the leg must FAIL 'forced output != anchor'")
    ap.add_argument("--neg-prompts", type=int, default=1)
    ap.add_argument("--neg-tokens", type=int, default=64)
    ap.add_argument("--floor-kl-tokens", type=int, default=128)
    args = ap.parse_args()

    pages = patch_geometry(args.window_tokens, args.refresh)

    from tilerl_kernels.backend import get_backend

    from tilerl import build as build_mod
    from tilerl.build import build_engine, build_model
    from tilerl.cli import _qwen38_tokenizer
    from tilerl.engine import SamplingParams
    from tilerl.spec import load_draft

    if args.source:
        build_mod.QWEN38_SOURCE = args.source
    be = get_backend()
    cfg, model = build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft, attn_window_tokens=2048)
    e = build_engine(
        cfg, model, be,
        num_slots=4, max_batch=4, max_total_tokens=131072,
        max_num_batched_tokens=512,
        sparse_k=128, sparse_min_tokens=0, sparse_device_select=True,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        cold_ssd_path=args.cold_ssd, cold_ssd_bytes=8 << 30, cold_format="f16",
        decode_graph=True, draft=draft, spec_depth=1,
    )
    import tilerl
    print(f"TILERL_FILE {tilerl.__file__}", flush=True)
    print(f"SPARSE_GRAPH_ON {e._sparse_graph_on}", flush=True)
    # #818 guard A: with depth1 + draft the sparse decode graph must be armed;
    # R>1 measured without it measures eager, not the sweep arm. R=1 declines
    # by design, so it does not need the graph.
    if args.refresh > 1 and not e._sparse_graph_on:
        print("FATAL sparse graph forced eager (guard A/#818 not in this tree)",
              file=sys.stderr)
        return 14
    tok = _qwen38_tokenizer()
    prompts = load_prompts(args.prompts, tok, args.n_prompts)

    # Spy on the refresh decline: spy_rdg must flag ONLY the counter-driven
    # decline, never a first-capture/pad decline (which would mislabel a graph
    # tick eager).
    declines: list[int] = []
    orig_rdg = type(e._sparse).run_decode_graph
    decline_flag = {"v": False}

    def spy_rdg(self2, reqs, chains):
        before = self2.ticks_since_refresh
        ok = orig_rdg(self2, reqs, chains)
        if not ok and before + 1 >= args.refresh:
            declines.append(before)
            decline_flag["v"] = True
        return ok

    type(e._sparse).run_decode_graph = spy_rdg

    # ---- negative control (R=1): self-feed with the anchor shifted. Must fail
    # with "forced output != anchor".
    if args.neg_anchor_offset:
        return neg_anchor_run(e, prompts[: args.neg_prompts], args, decline_flag)

    pp_dir = f"{args.out_prefix}_pp"
    tf_dir = f"{args.out_prefix}_tf"
    os.makedirs(pp_dir, exist_ok=True)
    os.makedirs(tf_dir, exist_ok=True)

    def write_meta(per_prompt, gate=None):
        key_own_ws = sorted({k[3] for k in e._sparse.graphs})
        keys_ok = key_own_ws == [] if args.refresh == 1 else pages + 1 in key_own_ws
        declines_ok = (
            all(d == args.refresh - 1 for d in declines) if args.refresh > 1
            else bool(declines) and all(d == 0 for d in declines)
        )
        structural = keys_ok and declines_ok
        summary = {
            "window_tokens": args.window_tokens, "window_pages": pages,
            "refresh_ticks": args.refresh, "graph_key_own_w": pages + 1,
            "observed_graph_key_own_ws": key_own_ws,
            "n_captured_graphs": len(e._sparse.graphs),
            "declines_seen": len(declines),
            "decline_counter_values": sorted(set(declines))[:8],
            "structural_gate_ok": bool(structural if gate is None else gate and structural),
            "self_anchor_gate_ok": gate,
            "floor_kl_max": floor_kl,
            "prompts": per_prompt,
        }
        with open(f"{args.out_prefix}.json", "w") as f:
            json.dump(summary, f, indent=2)

    free_stats = []
    anchors: dict[int, list[int]] = {}
    floor_kl = None
    if args.refresh == 1:
        # cutover floor: free-run logits vs self-TF logits on the same prefix,
        # before the 1024-token legs, so a broken instrument fails early.
        floor_kl = floor_kl_check(
            e, prompts[0], args.floor_kl_tokens, decline_flag)
        if floor_kl > 1e-4:
            print(f"FATAL floor KL {floor_kl:.2e} > 1e-4", file=sys.stderr)
            return 1
    for idx, ids in enumerate(prompts):
        # P0: zero the engine-wide refresh phase before each admission so a new
        # prompt never inherits the previous request's phase (probe-only).
        e._sparse.ticks_since_refresh = 0
        rid = e.submit(list(ids), SamplingParams(
            temperature=0.0, max_new_tokens=args.max_new_tokens, seed=0))
        out, st = run_one_prompt(e, rid, decline_flag)
        st["idx"] = idx
        st["prompt_tokens"] = len(ids)
        free_stats.append(st)
        anchors[idx] = out
        with open(f"{pp_dir}/ref_{idx:03d}.json", "w") as f:
            json.dump({"output": [int(x) for x in out]}, f)
        write_meta(free_stats)
        print(f"[W{args.window_tokens}/R{args.refresh}] free prompt {idx} "
              f"{len(out)} tok, eff {st['eff_tok_s']} tok/s", flush=True)

    if args.free_only:
        write_meta(free_stats)
        print("STRUCTURAL " + json.dumps({
            "window_pages": pages, "refresh_ticks": args.refresh,
            "n_captured_graphs": len(e._sparse.graphs),
            "declines_seen": len(declines), "free_only": True}), flush=True)
        e.shutdown()
        return 0

    # ---- teacher-forced run. The TF output is forced to the anchor, so
    # out==anchor by construction; the measured quantity is the trunk's top1
    # argmax vs the anchor at every committed position.
    # R=1 self-feeds: top1 must equal the anchor at every position (1.0),
    # perf1's self-anchor gate. R>1 feeds its W-group's R=1 free run; its
    # per-position agreement is the cross-arm quality number gated at >=0.99.
    tf_anchors = dict(anchors)
    if args.anchor_dir:
        tf_anchors = {}
        for p in sorted(os.listdir(args.anchor_dir)):
            if p.startswith("ref_") and p.endswith(".json"):
                i = int(p[len("ref_"):-len(".json")])
                with open(os.path.join(args.anchor_dir, p)) as f:
                    tf_anchors[i] = [int(x) for x in json.load(f)["output"]]
        missing = set(range(len(prompts))) - set(tf_anchors)
        if missing:
            print(f"FATAL anchor dir lacks prompts {sorted(missing)}", file=sys.stderr)
            return 14
    self_gate = True
    tf_agreement = []
    rec = TeacherForceRecorder(e, anchors=tf_anchors, record_full_logits=True).install()
    for idx, ids in enumerate(prompts):
        e._sparse.ticks_since_refresh = 0
        rid = e.submit(list(ids), SamplingParams(
            temperature=0.0, max_new_tokens=args.max_new_tokens, seed=0))
        out, _ = run_one_prompt(e, rid, decline_flag)
        kept = rec.accepted_positions(idx)
        anchor = tf_anchors[idx]
        if len(kept) != len(anchor):
            print(f"FATAL prompt {idx}: {len(kept)} committed records != "
                  f"{len(anchor)} anchor tokens", file=sys.stderr)
            return 1
        if out != anchor:
            print(f"FATAL prompt {idx}: forced output != anchor (instrument defect)",
                  file=sys.stderr)
            return 1
        top1s = [r["top1"] for r in kept]
        agree = sum(t == a for t, a in zip(top1s, anchor)) / len(anchor)
        tf_agreement.append(round(agree, 6))
        if args.refresh == 1 and agree != 1.0:
            self_gate = False
        torch.save(
            {"top1": torch.tensor(top1s, dtype=torch.long),
             "logits": torch.stack([r["logits"] for r in kept]),
             "committed": [r["committed"] for r in kept]},
            f"{tf_dir}/tf_{idx:03d}.pt")
        with open(f"{tf_dir}/tf_{idx:03d}.json", "w") as f:
            json.dump({"output": [int(x) for x in out], "n_positions": len(kept),
                       "top1_agreement_vs_anchor": round(agree, 6)}, f)
        print(f"[W{args.window_tokens}/R{args.refresh}] tf prompt {idx} "
              f"top1-vs-anchor {agree:.4f}", flush=True)
    rec.uninstall()

    for st in free_stats:
        st["tf_top1_agreement"] = tf_agreement[st["idx"]]
    write_meta(free_stats, gate=self_gate)
    print("STRUCTURAL " + json.dumps({
        "window_pages": pages, "refresh_ticks": args.refresh,
        "n_captured_graphs": len(e._sparse.graphs),
        "declines_seen": len(declines),
        "tf_top1_agreement": tf_agreement,
        "self_anchor_gate_ok": self_gate}), flush=True)
    if not self_gate:
        return 1
    e.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
