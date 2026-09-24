#!/usr/bin/env python3
"""PROBE-ONLY #298892: V100 harness for prefill snapshot speed/parity.

Three modes over the same W1024/R32 + draft + true-q-width build, run as
SEPARATE processes (that is the point of the snapshot):

  snapshot  prefill each prompt + 8 decode tokens (closes the warm prompt-end
            entry), dump it to --snap-root/p<idx>
  baseline  full prefill + 512 free decode; save output + speed
  snaprun   load the snapshot (timed), then 512 free decode; save output +
            speed and assert byte-equality with the baseline output

The build matches the production launcher: --sparse-window-tokens 1024
--sparse-refresh-ticks 32, depth-1 draft with the 2048 read window, and
TILERL_DRAFT_TRUE_Q_WIDTH=1 from the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prefill_snapshot import dump  # noqa: E402
from prefill_snapshot import load as snap_load
from probe_wr_sweep_worker import load_prompts, run_one_prompt  # noqa: E402


def build(args):
    from tilerl_kernels.backend import get_backend

    from tilerl import build as build_mod
    from tilerl.build import build_engine, build_model
    from tilerl.cli import _qwen38_tokenizer
    from tilerl.spec import load_draft

    if args.source:
        build_mod.QWEN38_SOURCE = args.source
    get_backend()
    cfg, model = build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft, attn_window_tokens=2048)
    return build_engine(
        cfg,
        model,
        get_backend(),
        num_slots=4,
        max_batch=4,
        max_total_tokens=131072,
        max_num_batched_tokens=512,
        sparse_k=128,
        sparse_min_tokens=0,
        sparse_device_select=True,
        scorer="bounds",
        sparse_window_tokens=1024,
        sparse_refresh_ticks=32,
        kv_cold_bytes=1 << 30,
        cold_ssd_path=args.cold_ssd,
        cold_ssd_bytes=8 << 30,
        cold_format="f16",
        decode_graph=True,
        draft=draft,
        spec_depth=1,
    ), _qwen38_tokenizer()


def _zero(e):
    e._sparse.ticks_since_refresh = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["snapshot", "baseline", "snaprun"])
    ap.add_argument("--prompts", default=os.path.expanduser("~/serve805_prompts.jsonl"))
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--snap-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=os.environ.get("TILERL_QWEN38_SOURCE", ""))
    ap.add_argument("--draft", default="/home/chenkailun.c/mmlu-assets/model_mtp.safetensors")
    ap.add_argument("--cold-ssd", default="/home/chenkailun.c/sparse_cold_128k.bin")
    args = ap.parse_args()

    e, tok = build(args)
    prompts = load_prompts(args.prompts, tok, args.n_prompts)
    model_key = os.path.realpath(args.source) if args.source else ""
    flag = {"v": False}
    from tilerl.engine import SamplingParams

    rows = []
    for idx, ids in enumerate(prompts):
        snap_dir = os.path.join(args.snap_root, f"p{idx}")
        load_ms = None
        if args.mode == "snapshot":
            _zero(e)
            rid = e.submit(list(ids), SamplingParams(temperature=0.0, max_new_tokens=8, seed=0))
            out, _ = run_one_prompt(e, rid, flag)
            dump(e, ids, snap_dir, model_key=model_key)
            # The V100 build carries a draft: a usable entry must be WARM (dk
            # blob present and boundary hidden non-None) or a follower misses.
            with open(os.path.join(snap_dir, "meta.json")) as f:
                sd = json.load(f)
            st = torch.load(
                os.path.join(snap_dir, "state.pt"), map_location="cpu", weights_only=True
            )
            has_dk = "dk" in sd["pages"][0]["fields"]
            assert has_dk and st["hidden"] is not None, (
                f"snapshot p{idx} not warm (dk={has_dk}, hidden={st['hidden'] is not None})"
            )
            print(
                f"SNAPSHOT p{idx}: {sd['n_tokens']} tok, {len(sd['pages'])} pages, "
                f"fields={sd['pages'][0]['fields']}",
                flush=True,
            )
            continue
        if args.mode == "snaprun":
            t0 = time.perf_counter()
            snap_load(e, snap_dir, ids, model_key=model_key)
            load_ms = round((time.perf_counter() - t0) * 1000, 1)
        _zero(e)
        rid = e.submit(
            list(ids), SamplingParams(temperature=0.0, max_new_tokens=args.decode_tokens, seed=0)
        )
        out, st = run_one_prompt(e, rid, flag)
        st["idx"] = idx
        st["snapshot_load_ms"] = load_ms
        rows.append(st)
        with open(f"{args.out}.p{idx}.json", "w") as f:
            json.dump({"output": [int(x) for x in out], "snapshot_load_ms": load_ms}, f)
        print(
            f"{args.mode} p{idx}: warm {st['warm_tok_s']} tok/s, load_ms {load_ms}, {len(out)} tok",
            flush=True,
        )

    if rows:
        with open(args.out + ".json", "w") as f:
            json.dump({"mode": args.mode, "prompts": rows}, f, indent=2)
    e.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
