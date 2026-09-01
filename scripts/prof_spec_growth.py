"""Does a speculative tick get slower as the sequence grows?

The server measured 705 ms/token at 288 tokens while a tick profiled 30 tokens
in costs 119 ms. This times ticks across the generation to find the growth.

  scripts/v100.sh run gr 'CKPT=...; /usr/bin/python3 -u scripts/prof_spec_growth.py \
      --source $CKPT --draft $CKPT/model-00018-of-00018.safetensors --depth 3'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.spec import load_draft
from tilerl_kernels.backend import get_backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--serve-like", action="store_true",
                    help="the pools cmd_serve uses (max_batch 8), not the probe defaults")
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft) if args.draft else None
    e = build_engine(cfg, model, backend, num_blocks=512, num_slots=4,
                     max_batch=8 if args.serve_like else 4,
                     max_total_tokens=8192, draft=draft,
                     spec_depth=args.depth if draft else 1)
    rid = e.submit(list(range(10, 40)),
                   SamplingParams(temperature=0.0, max_new_tokens=args.tokens, seed=0))
    torch.cuda.synchronize()
    n_prev, tick = 0, 0
    print(f"{'tick':>5} {'seq_len':>8} {'ms':>8} {'new tok':>8}")
    while True:
        t0 = time.perf_counter()
        e.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        tick += 1
        done = e.poll().get(rid)
        rows = list(e._running)
        sl = rows[0].seq_len if rows else -1
        n = len(rows[0].output) if rows else n_prev
        if tick % 5 == 0 or tick < 4:
            print(f"{tick:>5} {sl:>8} {ms:>8.1f} {n - n_prev:>8}")
        n_prev = n
        if done is not None:
            break
    print(f"\n{len(done)} tokens in {tick} ticks")


if __name__ == "__main__":
    main()
