"""V100 sm70 27B prefill split: linear vs attention vs GDN(chunk) vs other, ms.

Times one real prefill tick at --ctx via CUDA events around:
  - paged_attention / paged_attention_prefill (the 16 full-attn layers),
  - linear_attn_chunk (the 48 GDN layers' chunk math),
  - the whole model forward.
Everything else = forward - attention - gdn (linears, norms, writes, lm_head).
Two draws (warm + measured), 32k and 64k.

  scripts/v100.sh run presplit 'CKPT=...; /usr/bin/python3 -u scripts/bench_prefill_split_v100.py \
      --source $CKPT --ctxs 32768,65536'
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

ATTN_NAMES = ("paged_attention", "paged_attention_prefill")
GDN_NAMES = ("linear_attn_chunk",)


def install(backend, acc: dict):
    # Record event pairs without syncing (a per-call sync would serialize and
    # inflate the wall); timestamps are read after one final sync.
    for name in ATTN_NAMES:
        orig = getattr(backend, name, None)
        if orig is None:
            continue

        def make(o=orig):
            def wrap(*a, **kw):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                r = o(*a, **kw)
                e.record()
                acc["attn"].append((s, e))
                return r
            return wrap

        setattr(backend, name, make())
    for name in GDN_NAMES:
        orig = getattr(backend, name, None)
        if orig is None:
            continue

        def make(o=orig):
            def wrap(*a, **kw):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                r = o(*a, **kw)
                e.record()
                acc["gdn"].append((s, e))
                return r
            return wrap

        setattr(backend, name, make())


def _total(pairs):
    return sum(s.elapsed_time(e) for s, e in pairs)


def _prompt(ctx, vocab):
    g = torch.Generator().manual_seed(1000)
    return torch.randint(0, vocab, (ctx,), generator=g).tolist()


def run(e, ctx, vocab):
    return e.submit(_prompt(ctx, vocab), SamplingParams(temperature=0.0, max_new_tokens=1, seed=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--ctxs", default="32768,65536")
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source
    ctxs = [int(c) for c in args.ctxs.split(",")]

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    blocks = (-(-(max(ctxs) + 8) // BLOCK_TOKENS)) + 32
    print(f"pool {blocks} blocks ({blocks*2.125:.0f} MiB)", flush=True)
    e = build_engine(cfg, model, backend, num_blocks=blocks, num_slots=3, max_batch=2,
                     max_total_tokens=max(ctxs) + 8)
    acc = {"attn": [], "gdn": []}
    install(backend, acc)  # wrap once; the lists are reset between draws
    print(f"{'ctx':>6} {'forward_ms':>11} {'attn_ms':>9} {'gdn_ms':>9} "
          f"{'other_ms':>9} {'ms/token':>9}", flush=True)
    for ctx in ctxs:
        for draw in ("warm", "measured"):
            acc["attn"].clear()
            acc["gdn"].clear()
            rid = run(e, ctx, cfg.vocab_size)
            done = {}
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(4_000_000):
                e.step()
                done.update(e.poll())
                if rid in done:
                    break
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000
            attn_ms, gdn_ms = _total(acc["attn"]), _total(acc["gdn"])
            if draw == "measured":
                other = wall - attn_ms - gdn_ms
                print(f"{ctx:>6} {wall:>11.0f} {attn_ms:>9.0f} {gdn_ms:>9.0f} "
                      f"{other:>9.0f} {wall/ctx:>9.2f}", flush=True)
    print("SPLIT_DONE", flush=True)


if __name__ == "__main__":
    main()
