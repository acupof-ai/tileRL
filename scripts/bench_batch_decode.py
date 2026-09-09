"""Decode throughput vs batch size on the NVFP4 slice: B concurrent requests, timed once all decode.

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src TILERL_TARGET=cuda python3 scripts/bench_batch_decode.py /host/tc27-nvfp4-slice4 --layers 4 --card 0

Emits one decode_agg_tok_s record per B (plus spec_goodput_ratio rows with
--draft, against the store's dense row for the same population) to
docs/experience/bench/measurements.jsonl (schema: docs/bench-schema.md).
Build is derived from --fuse/--decode-graph/--draft.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchrec  # noqa: E402
import torch
from tilerl_kernels.backend import get_backend

from tilerl.config import qwen36_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import load_hf
from tilerl.spec import load_draft

WARMUP = 8  # ticks: flushes every 16-token prompt's prefill, leaves decode headroom


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ticks", type=int, default=30)
    p.add_argument("--batches", type=str, default="1,2,4,8")
    p.add_argument("--fuse", action="store_true", help="fuse same-input fp4 projections")
    p.add_argument(
        "--decode-graph", action="store_true", help="capture decode graphs per batch bucket"
    )
    p.add_argument("--slots", type=int, help="state slots / max_batch (default: max batch swept)")
    p.add_argument("--draft", help="draft head safetensors: speculative decode")
    p.add_argument("--depth", type=int, default=4, help="drafts per row per tick")
    benchrec.add_record_args(p, default_device=None)
    args = p.parse_args()
    args.model_name = f"27B-nvfp4-slice{args.layers}"

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = replace(
        qwen36_27b(),
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in qwen36_27b().full_attn_layers if i < args.layers),
    )
    model = load_hf(cfg, args.source, fuse_projections=args.fuse)
    batches = [int(x) for x in args.batches.split(",")]
    bmax = args.slots or max(batches)
    engine = build_engine(
        # 16 blocks/request covers 16-token prompts generating ~170; sizing off bmax alone thrashed at 94 GiB.
        cfg, model, backend, num_blocks=16 * bmax, num_slots=bmax, max_batch=bmax,
        max_total_tokens=256 * bmax, decode_graph=args.decode_graph,
        draft=load_draft(model, args.draft) if args.draft else None, spec_depth=args.depth,
    )

    gen = torch.Generator().manual_seed(7)
    build = (("fused" if args.fuse else "eager")
             + ("+graph" if args.decode_graph else "")
             + ("+draft" if args.draft else ""))
    common = benchrec.record_common(args, build=build)
    print(
        f"\n=== decode throughput vs batch "
        f"(slice {args.layers} layers, {'graph' if args.decode_graph else 'eager'}) ==="
    )
    print(f"  {'B':>3} {'ms/tick':>9} {'tok/tick':>9} {'accept':>7} "
          f"{'per-request tok/s':>18} {'aggregate tok/s':>17}")
    for B in batches:
        prompts = [
            torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist() for _ in range(B)
        ]
        wids = [
            engine.submit(
                p, SamplingParams(
                    temperature=0.0, seed=i,
                    max_new_tokens=(WARMUP + args.ticks + 4) * (1 + args.depth),
                )
            )
            for i, p in enumerate(prompts)
        ]
        for _ in range(WARMUP):
            engine.step()
        torch.cuda.synchronize()
        s0 = engine.stats()
        t0 = time.perf_counter()
        for _ in range(args.ticks):
            engine.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / args.ticks * 1e3
        s1 = engine.stats()
        per_tick = (s1["tokens_generated"] - s0["tokens_generated"]) / args.ticks / B
        drafted = s1["spec_drafted"] - s0["spec_drafted"]
        acc = (s1["spec_accepted"] - s0["spec_accepted"]) / max(drafted, 1)
        print(f"  {B:>3} {ms:>9.3f} {per_tick:>9.2f} {100 * acc:>6.1f}% "
              f"{1000 * per_tick / ms:>18.1f} {1000 * B * per_tick / ms:>17.1f}")
        agg = 1000 * B * per_tick / ms
        rec = {
            "metric": "decode_agg_tok_s", "value": round(agg, 1), "unit": "tok/s",
            "shape": {"batch": B, "ctx": 16},
            "warm": {"state": "warm", "compiles": 0},
            "n": 1, "spread": 0.0, **common,
        }
        rec["floor"] = benchrec.measured_best_floor(rec, lower_is_better=False)
        print(f"  record {benchrec.append(rec)} appended", flush=True)
        if args.draft:
            dense_rec = {**rec, "build": build.replace("+draft", "")}
            dense_best, dense_id = benchrec.measured_best(dense_rec, lower_is_better=False)
            if dense_id is None:
                print(f"  spec_goodput_ratio skipped: no dense row for B={B} yet", flush=True)
            else:
                srec = {
                    "metric": "spec_goodput_ratio", "value": round(agg / dense_best, 3),
                    "unit": "ratio", "shape": {"batch": B, "depth": args.depth},
                    "warm": {"state": "warm", "compiles": 0},
                    "n": 1, "spread": 0.0, **common,
                }
                srec["floor"] = {
                    "value": 1.0, "unit": "ratio", "kind": "baseline",
                    "derivation": f"1.0 = spec matches dense ({dense_best:.1f} tok/s, "
                                  f"row {dense_id}); below is a loss",
                }
                print(f"  record {benchrec.append(srec)} appended", flush=True)
        # Drain fully: the next B reuses the same slot pool.
        done: dict = {}
        while not all(w in done for w in wids):
            engine.step()
            done.update(engine.poll())


if __name__ == "__main__":
    main()
