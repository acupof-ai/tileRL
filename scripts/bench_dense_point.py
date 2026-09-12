"""One long-context point: prefill wall-seconds to a held span + steady decode
tok/s, dense or sparse. The smallest job for the statistic (ckl 2026-09-11): one
context, one request, greedy. Drives the engine directly with the span's token ids.

The engine generates greedily and exposes req.output via poll() ONLY at finish,
so per-token time cannot come from poll(). Timing wraps _run_forward and
classifies each tick prefill vs decode by its plan rows (the bench_ctx_decode.py
idiom), CUDA-synced. Decode tok/s drops the first --skip decode ticks (JIT/graph
capture warmup). A wall-clock guard turns any silent core-spin into a TimeoutError.

  dense H20:  --source <ckpt> --span held-128k.json --ctx 131072
  sparse:     same plus --sparse-k 128 --scorer bounds --cold-bytes N
  V100 adds --eager (no decode graph).
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="27B checkpoint dir")
    ap.add_argument("--span", required=True, help="held span json ({ids:[...]}) or jsonl")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--skip", type=int, default=16, help="decode ticks excluded (JIT/graph)")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--eager", action="store_true", help="force decode graph off (V100)")
    ap.add_argument("--sparse-k", type=int, default=0, help="selected hot pages (0=dense)")
    ap.add_argument("--scorer", default="bounds", choices=["bounds", "index"])
    ap.add_argument("--cold-bytes", type=int, default=1 << 32, help="host cold-tier budget")
    ap.add_argument("--max-wait-s", type=float, default=3600,
                    help="wall-clock guard so a stalled tick never spins silently")
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    if args.span.endswith(".jsonl"):
        row = json.loads(Path(args.span).read_text().splitlines()[0])
    else:
        row = json.loads(Path(args.span).read_text())
    ids = [int(x) for x in row["ids"]][: args.ctx]
    assert len(ids) == args.ctx, f"span has {len(ids)} < ctx {args.ctx}"

    be = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    max_total = args.ctx + args.new_tokens + 256
    kw = dict(num_blocks=0, num_slots=args.slots, max_batch=1,
              max_total_tokens=max_total, max_blocks=(max_total // 16) * args.slots + 2,
              decode_graph=False if args.eager else None)
    if args.sparse_k:
        kw.update(sparse_k=args.sparse_k, scorer=args.scorer,
                  kv_cold_bytes=args.cold_bytes)
    e = build_engine(cfg, model, be, **kw)
    arm = f"sparse-k{args.sparse_k}-{args.scorer}" if args.sparse_k else "dense"
    print(f"# arm {arm} arch {be.arch} ctx {args.ctx} pool_blocks {e._kv.num_blocks} "
          f"graph {e._decode_graph_on}", flush=True)

    pf_secs: list[float] = []
    dec_secs: list[float] = []
    orig_forward = type(e)._run_forward

    def timed_forward(self, decodes, prefills, chunks):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig_forward(self, decodes, prefills, chunks)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        (dec_secs if (decodes and not prefills) else pf_secs).append(dt)
        return r

    type(e)._run_forward = timed_forward
    rid = e.submit(ids, SamplingParams(temperature=0.0,
                                      max_new_tokens=args.new_tokens, seed=0))
    deadline = time.perf_counter() + args.max_wait_s
    out: list = []
    while len(out) < args.new_tokens:
        e.step()
        out = e.poll().get(rid, out)
        if time.perf_counter() > deadline:
            raise TimeoutError(
                f"stalled at {len(out)}/{args.new_tokens} tokens; "
                f"{len(pf_secs)} prefill / {len(dec_secs)} decode ticks")
    type(e)._run_forward = orig_forward

    prefill_s = sum(pf_secs)
    print(f"PREFILL_SECONDS {prefill_s:.3f}  ms/tok {prefill_s * 1000 / args.ctx:.3f} "
          f"({len(pf_secs)} prefill ticks)", flush=True)
    steady = dec_secs[args.skip:]
    if steady:
        dec_s = sum(steady)
        print(f"DECODE_TOK_S {len(steady) / dec_s:.3f}  over_ticks {args.skip}.."
              f"{len(dec_secs)} ({dec_s:.3f}s, {dec_s / len(steady) * 1000:.1f} ms/tok)",
              flush=True)
    else:
        print(f"DECODE_TOK_S NA (only {len(dec_secs)} decode ticks, skip={args.skip})",
              flush=True)
    print(f"FIRST_TOKENS {list(out[:8])}", flush=True)
    kv = e._kv
    dev_bytes = kv.k_pool.nelement() * kv.k_pool.element_size()
    cold_bytes = getattr(kv, "cold", None).bytes_held if getattr(kv, "cold", None) else 0
    print(f"KV_DEVICE_BYTES {dev_bytes}  KV_COLD_HOST_BYTES {cold_bytes}", flush=True)
    print(f"CARDTAG {torch.cuda.get_device_name(0)}")
    e.shutdown()


if __name__ == "__main__":
    main()
