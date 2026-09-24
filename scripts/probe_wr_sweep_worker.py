#!/usr/bin/env python3
"""One (W, R) arm of the local-window × refresh-interval sweep — V100 sm70.

One process per arm (one 27B per process is the harness rule; two engines in
one process OOM). Production decode config: sparse graph armed, sparse_min
tokens=0, depth-1 draft with the 2048 read window. Patched WITHOUT touching
src, before the engine is built:

  WINDOW_TOKENS / WINDOW_PAGES
      sparse_index owns the constants; sparse_engine imported WINDOW_PAGES BY
      VALUE at module load (own_bound/graph key), so both modules are patched.
      sparse_runtime/kernel_cost/memory import WINDOW_PAGES inside functions,
      so they re-read sparse_index at call time and need no patch.
  SPARSE_REFRESH_TICKS
      defined in sparse_engine; sparse_runtime imports it inside its two
      functions, so the module attr is the single value to patch.

Structural gate printed at startup: the patched (window_pages, R), the
observed refresh spacing over a probe run (eager must land exactly every R
graph ticks), and graph key own_w = window_pages + 1. R=1 refreshes every
tick, so every tick is eager by construction -- it is the quality reference.

Outputs match compare_refresh_runs.py's input contract:
  <out>_pp/ref_NNN.json  {"output": [ids]} per prompt
  <out>.json             config, per-prompt speed, graph/eager tick split

Speed: per-tick CUDA-synced ms over the STEADY span (prefill excluded),
eff tok/s = (decode_forwards + spec_accepted) / steady wall seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def patch_geometry(window_tokens: int, refresh: int) -> None:
    from tilerl import sparse_engine, sparse_index

    pages = window_tokens // sparse_index.BLOCK_TOKENS
    sparse_index.WINDOW_TOKENS = window_tokens
    sparse_index.WINDOW_PAGES = pages
    sparse_engine.WINDOW_PAGES = pages
    sparse_engine.SPARSE_REFRESH_TICKS = refresh
    print(f"PATCH window_tokens={window_tokens} window_pages={pages} R={refresh}",
          flush=True)


def load_prompts(path: str, tokenizer, want_n: int) -> list[list[int]]:
    with open(path) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    out = []
    for r in rows:
        if "input_ids" in r:
            ids = [int(x) for x in r["input_ids"]]
        elif "text" in r:
            ids = tokenizer.encode(r["text"])
        else:
            continue
        out.append(ids)
    if len(out) < want_n:
        raise SystemExit(f"only {len(out)} prompts, need {want_n}: {path}")
    return out[:want_n]


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
    args = ap.parse_args()

    patch_geometry(args.window_tokens, args.refresh)

    import torch
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
    tok = _qwen38_tokenizer()
    prompts = load_prompts(args.prompts, tok, args.n_prompts)

    # Structural positive control: observed graph-decline spacing on ONE probe
    # prompt. run_decode_graph declines exactly when ticks_since_refresh+1 >= R.
    declines: list[int] = []
    orig_rdg = type(e._sparse).run_decode_graph
    last_decline = {"v": False}

    def spy_rdg(self2, reqs, chains):
        before = self2.ticks_since_refresh
        ok = orig_rdg(self2, reqs, chains)
        if not ok and before + 1 >= args.refresh:
            declines.append(before)
            last_decline["v"] = True
        return ok

    type(e._sparse).run_decode_graph = spy_rdg
    pp_dir = f"{args.out_prefix}_pp"
    os.makedirs(pp_dir, exist_ok=True)

    per_prompt = []
    for idx, ids in enumerate(prompts):
        rid = e.submit(list(ids), SamplingParams(
            temperature=0.0, max_new_tokens=args.max_new_tokens, seed=0))
        decoding = False
        graph_ms: list[float] = []
        eager_ms: list[float] = []
        wall0 = f0 = acc0 = None
        for _ in range(200000):
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
                (eager_ms if last_decline["v"] else graph_ms).append(t0.elapsed_time(t1))
                last_decline["v"] = False
            e.poll()
            out = e.take(rid)
            if out is not None:
                with open(f"{pp_dir}/ref_{idx:03d}.json", "w") as f:
                    json.dump({"output": [int(x) for x in out]}, f)
                break
        else:
            raise SystemExit(f"prompt {idx} never finished")
        wall = time.perf_counter() - wall0
        fwd = e._decode_forwards - f0
        accepted = e._spec_accepted - acc0
        per_prompt.append({
            "idx": idx, "prompt_tokens": len(ids), "generated": len(out),
            "graph_ticks": len(graph_ms), "eager_refresh_ticks": len(eager_ms),
            "graph_tick_ms_mean": round(sum(graph_ms) / max(len(graph_ms), 1), 3),
            "eager_tick_ms_mean": round(sum(eager_ms) / max(len(eager_ms), 1), 3),
            "wall_s": round(wall, 3),
            "decode_forwards": fwd, "spec_accepted": accepted,
            "eff_tok_s": round((fwd + accepted) / wall, 3),
        })
        print(f"[{args.window_tokens}/R{args.refresh}] prompt {idx} done "
              f"{len(out)} tok, eff {per_prompt[-1]['eff_tok_s']} tok/s",
              flush=True)

    # Structural gate, read off the runtime itself — not recomputed constants:
    # 1. captured graph keys carry own_w = window_pages + 1 (W=2 verify width);
    # 2. R=1 declines every attempt, so it must have captured ZERO graphs;
    # 3. R>1 refresh declines fired at counter R-1 (every R graph ticks).
    pages = args.window_tokens // 16
    key_own_ws = sorted({k[3] for k in e._sparse.graphs})
    keys_ok = (key_own_ws == [] if args.refresh == 1 else key_own_ws == [pages + 1])
    expected_decline_at = args.refresh - 1
    declines_ok = (
        all(d == expected_decline_at for d in declines) if args.refresh > 1
        else len(declines) > 0 and all(d == 0 for d in declines)
    )
    own_w = pages + 1
    summary = {
        "window_tokens": args.window_tokens, "window_pages": pages,
        "refresh_ticks": args.refresh, "graph_key_own_w": own_w,
        "observed_graph_key_own_ws": key_own_ws,
        "n_captured_graphs": len(e._sparse.graphs),
        "declines_seen": len(declines),
        "decline_counter_values": sorted(set(declines))[:8],
        "structural_gate_ok": bool(keys_ok and declines_ok),
        "prompts": per_prompt,
    }
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("STRUCTURAL " + json.dumps({k: summary[k] for k in (
        "window_pages", "refresh_ticks", "graph_key_own_w",
        "observed_graph_key_own_ws", "n_captured_graphs",
        "declines_seen", "structural_gate_ok")}), flush=True)
    if not (keys_ok and declines_ok):
        return 1
    e.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
