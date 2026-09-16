"""NVTX-windowed sparse/dense decode tick profiler for nsys (H20 sm90).

Builds ONE 27B engine, prefills one long request to ~ctx tokens, then runs
NTICKS steady-state decode ticks inside an NVTX range named PROFILE so
`nsys profile --capture-range=nvtx --nvtx-capture=PROFILE` records ONLY the
steady decode ticks (not the long prefill). Graph-on vs eager via --graph.

nsys:
  nsys profile -t cuda,nvtx --capture-range=nvtx --nvtx-capture=PROFILE \
    -o sparse32k_graph --delay=10 python scripts/perf2_nsys_ticks.py --ctx 32768
"""

from __future__ import annotations

import argparse
import random

import torch
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine, build_model
from tilerl.engine import SamplingParams
from tilerl.kv_cache import NoPrefixStore


def ids(n, seed, vocab):
    rng = random.Random(seed)
    return [rng.randrange(10, vocab - 10) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--nticks", type=int, default=12)
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--graph", type=int, default=1)
    ap.add_argument("--source", default="/work/tilerl-ckpt/Qwen3.8-27B-NVFP4")
    ap.add_argument("--draft", default="/work/tilerl-ckpt/Qwen3.8-27B-NVFP4/model_mtp.safetensors")
    args = ap.parse_args()

    import os

    os.environ["TILERL_QWEN38_SOURCE"] = args.source
    backend = get_backend()
    cfg, model = build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = None
    if args.draft:
        from tilerl.spec import load_draft

        draft = load_draft(model, args.draft)

    nblocks = (args.ctx + 4608) // 16
    eng = build_engine(
        cfg,
        model,
        backend,
        num_blocks=nblocks,
        num_slots=4,
        max_batch=4,
        max_total_tokens=args.ctx + 4352,
        max_num_batched_tokens=512,
        sparse_k=(128 if args.sparse else 0),
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        draft=draft,
        spec_depth=1,
        prefix_store=NoPrefixStore(),
        decode_graph=bool(args.graph),
    )
    print(
        f"engine ready graph={eng._decode_graph_on} sparse={bool(args.sparse)} blocks={nblocks}",
        flush=True,
    )

    prompt = ids(args.ctx, 7, cfg.vocab_size)
    rid = eng.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4096, seed=0))

    # Drive by decode_forwards stats, not poll() accumulation: a poll either
    # returns nothing (prefill) or drains many tokens at once, so a token-count
    # loop either stalls through prefill or runs the request to completion.
    # Burn prefill + 64 warm decode ticks, leaving 4000+ tokens of headroom.
    def nfwd():
        return eng.stats()["decode_forwards"]

    f0 = nfwd()
    while nfwd() - f0 < 64 or eng._waiting:
        eng.step()
        eng.poll()
    torch.cuda.synchronize()
    assert rid in {r.req_id for r in eng._running}, "request finished during warm"
    print(f"warm done at {nfwd() - f0} decode forwards; starting PROFILE range", flush=True)

    # cudaProfilerApi capture range: more reliable under nsys than torch NVTX
    # (the NVTX push/pop produced "No reports were generated"). nsys is launched
    # with --capture-range=cudaProfilerApi, so only work between Start/Stop lands
    # in the .nsys-rep; the long prefill above is excluded.
    import time as _time

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    p0 = nfwd()
    t0.record()
    for i in range(args.nticks):
        eng.step()
        eng.poll()
    t1.record()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()
    _time.sleep(2)  # let nsys flush before shutdown exits the process
    real = nfwd() - p0
    print(
        f"PROFILE {args.nticks} steps, {real} decode forwards, "
        f"{t0.elapsed_time(t1):.2f} ms = {t0.elapsed_time(t1) / max(real, 1):.2f} ms/forward",
        flush=True,
    )
    eng.shutdown()


if __name__ == "__main__":
    main()
