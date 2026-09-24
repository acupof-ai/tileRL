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
    for _ in range(200000):
        tag_decline["v"] = False
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        e.step()
        t1.record()
        torch.cuda.synchronize()
        row = next((r for r in e._running if r.req_id == rid), None)
        if row is not None and row.phase == 2:
            if not decoding:
                decoding = True
                wall0 = time.perf_counter()
                f0 = e._decode_forwards
                acc0 = e._spec_accepted
            (eager_ms if tag_decline["v"] else graph_ms).append(t0.elapsed_time(t1))
        e.poll()
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


def accepted_records(recs: list[dict]) -> list[dict]:
    """Keep one record per committed generated position. A verify tick records
    every chain slot; a rejected draft slot's gen_idx equals a position the
    NEXT tick commits, so per gen_idx keep the smallest chain slot
    (gen_idx - out_len_before) — that is the accepted slot."""
    best: dict[int, dict] = {}
    for r in recs:
        slot = r["gen_idx"] - r["out_len_before"]
        cur = best.get(r["gen_idx"])
        if cur is None or slot < cur["gen_idx"] - cur["out_len_before"]:
            best[r["gen_idx"]] = r
    return [best[k] for k in sorted(best)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-tokens", type=int, required=True, choices=[128, 1024])
    ap.add_argument("--refresh", type=int, required=True, choices=[1, 8, 16, 32])
    ap.add_argument("--prompts", default=os.path.expanduser("~/serve805_prompts.jsonl"))
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--source", default=os.environ.get("TILERL_QWEN38_SOURCE", ""))
    ap.add_argument("--draft", default="/home/chenkailun.c/mmlu-assets/model_mtp.safetensors")
    ap.add_argument("--cold-ssd", default="/home/chenkailun.c/sparse_cold_128k.bin")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--anchor-dir", default="",
                    help="dir with ref_NNN.json anchors. R=1 self-feeds its own "
                         "free run (the byte-identical control); every other R "
                         "teacher-forces its W-group R=1 free run")
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
            "prompts": per_prompt,
        }
        with open(f"{args.out_prefix}.json", "w") as f:
            json.dump(summary, f, indent=2)

    free_stats = []
    anchors: dict[int, list[int]] = {}
    for idx, ids in enumerate(prompts):
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
        rid = e.submit(list(ids), SamplingParams(
            temperature=0.0, max_new_tokens=args.max_new_tokens, seed=0))
        out, _ = run_one_prompt(e, rid, decline_flag)
        kept = accepted_records(rec.rows[idx])
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
