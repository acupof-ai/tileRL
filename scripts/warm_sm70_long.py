"""Warm the host-shared TILELANG_CACHE_DIR for a long prefill/decode of the 27B
on sm70, ahead of the unit-F sparse run. Compiles every kernel at the 128k
block-pool width. One request, prefill to --ctx then a few decode tokens.

  scripts/v100.sh run warm128 'CKPT=...; /usr/bin/python3 -u scripts/warm_sm70_long.py \
      --source $CKPT --ctx 131072'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl import cli  # noqa: E402
from tilerl.cli import _build_model  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.kv_cache import BLOCK_TOKENS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--tokens", type=int, default=4)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    blocks = (-(-(args.ctx + args.tokens + 4) // BLOCK_TOKENS)) + 32
    print(f"pool: {blocks} blocks ({blocks * 2.125:.0f} MiB), prefill to {args.ctx}",
          flush=True)
    e = build_engine(cfg, model, backend, num_blocks=blocks, num_slots=3, max_batch=2,
                     max_total_tokens=args.ctx + args.tokens + 4)
    g = torch.Generator().manual_seed(1000)
    prompt = torch.randint(0, cfg.vocab_size, (args.ctx,), generator=g).tolist()
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=args.tokens, seed=0))
    done = {}
    t0 = time.perf_counter()
    last = t0
    toks = 0
    deadline = t0 + 5 * 3600
    for _ in range(2_000_000):
        e.step()
        done.update(e.poll())
        now = time.perf_counter()
        if now - last > 60:
            toks += 1
            print(f"  {now - t0:.0f}s elapsed, running={len(e._running)} "
                  f"waiting={len(e._waiting)}", flush=True)
            last = now
        if rid in done:
            print(f"WARM_DONE in {time.perf_counter() - t0:.0f}s, "
                  f"generated {len(done[rid].tokens) if hasattr(done[rid], 'tokens') else '?'}",
                  flush=True)
            return
        if now > deadline:
            raise SystemExit("warm stalled >5h")
    raise SystemExit("warm ran out of tick budget")


if __name__ == "__main__":
    main()
