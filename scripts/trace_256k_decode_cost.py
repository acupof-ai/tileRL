#!/usr/bin/env python3
"""Decode ms/tick on the sparse cold tier after a long prefill (256k vs 32k).

The sparse cold-tier win entries are prefill-only. This measures the DECODE
cost on top of a k=128 hot set whose pages mostly live in the host tier / mmap
spill. On the measured head the serve path never enabled device selection, so
every pure-decode tick ran the eager full-candidate re-selection (the cost
this prices); per-tick refresh tagging is recorded for the post-#557 graph
path, where 7 captured ticks alternate with one eager refresh every
SPARSE_REFRESH_TICKS. Reports per-tick ms (CUDA-synced) plus host RSS.

H20 card 3, one ctx per process:
  scripts/pod_run.sh dec256k 3 -- python3 scripts/trace_256k_decode_cost.py \
      --source /work/Qwen3.8-27B-NVFP4 --spill /work/tilerl-s-5f/dcost.bin --ctx 262144
  scripts/pod_run.sh dec32k  3 -- python3 scripts/trace_256k_decode_cost.py \
      --source /work/Qwen3.8-27B-NVFP4 --spill /work/tilerl-s-5f/dcost32.bin --ctx 32768
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time

import numpy as np
import torch
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine, build_model
from tilerl.engine import SamplingParams
from tilerl.sparse_engine import SPARSE_REFRESH_TICKS

GIB = 1 << 30


def vm_rss_gib() -> float:
    with open(f"/proc/{os.getpid()}/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / GIB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--spill", required=True)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--warmup-ticks", type=int, default=SPARSE_REFRESH_TICKS)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--host-gib", type=float, default=6.0)
    ap.add_argument("--ssd-gib", type=float, default=12.0)
    ap.add_argument("--out", default="/work/trace_decode_cost.json")
    args = ap.parse_args()

    os.environ.setdefault("TILERL_TARGET", "cuda")
    import tilerl.build as build  # noqa: E402
    build.QWEN38_SOURCE = args.source

    backend = get_backend()
    cfg, model = build_model("qwen38-27b", seed=0, fuse_projections=True)

    engine = build_engine(
        cfg=cfg, model=model, backend=backend,
        num_slots=1, max_batch=1,
        max_total_tokens=args.ctx + 256, max_num_batched_tokens=512,
        sparse_k=args.k, scorer="bounds",
        kv_cold_bytes=int(args.host_gib * GIB),
        cold_ssd_path=args.spill, cold_ssd_bytes=int(args.ssd_gib * GIB))
    cold = engine._kv.cold

    rss: list[float] = []
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(1.0):
            rss.append(round(vm_rss_gib(), 3))

    prompt = (np.arange(args.ctx, dtype=np.int64) % cfg.vocab_size)
    params = SamplingParams(temperature=0.0, max_new_tokens=args.decode_tokens, seed=0)
    rid = engine.submit(prompt, params)

    t0 = time.time()
    in_decode = False
    prefill_s = None
    ticks: list[dict] = []
    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    while True:
        done = engine.poll()
        if rid in done and len(done[rid]) >= args.decode_tokens:
            break
        r = next((x for x in engine._running if x.req_id == rid), None)
        if r is not None and r.phase == 2 and not in_decode:
            in_decode = True
            prefill_s = time.time() - t0
            rss.clear()  # decode-only RSS (sampler also ran through prefill)
            print(f"PREFILL_DONE {prefill_s:.1f}s", flush=True)
        if in_decode:
            # refresh fires when the pre-step counter increments to R
            # (engine.py: do_refresh on pure-decode ticks only).
            before = engine._sparse_ticks_since_refresh
            is_refresh = before + 1 >= SPARSE_REFRESH_TICKS
            torch.cuda.synchronize()
            ts = time.perf_counter()
            engine.step()
            torch.cuda.synchronize()
            ticks.append({"ms": round((time.perf_counter() - ts) * 1e3, 3),
                          "refresh": is_refresh})
        else:
            engine.step()
    stop.set()
    sampler.join(timeout=1)
    total_s = time.time() - t0

    # The first few decode ticks JIT decode shapes; exclude them from the
    # statistic but keep them in the per-tick record.
    timed = ticks[args.warmup_ticks:]
    ms_all = [t["ms"] for t in timed]
    refresh = [t["ms"] for t in timed if t["refresh"]]
    ordinary = [t["ms"] for t in timed if not t["refresh"]]
    st = cold.stats()

    def pct(xs: list[float], q: float) -> float:
        return round(statistics.quantiles(xs, n=100, method="inclusive")[q - 1], 3) if xs else 0.0

    result = {
        "ctx": args.ctx,
        "k": args.k,
        "prefill_s": round(prefill_s, 1),
        "total_s": round(total_s, 1),
        "decode_ticks": len(ticks),
        "warmup_ticks_excluded": args.warmup_ticks,
        "refresh_every_ticks": SPARSE_REFRESH_TICKS,
        "ms_per_tick": {
            "median": round(statistics.median(ms_all), 3),
            "p10": pct(ms_all, 10),
            "p90": pct(ms_all, 90),
            "max": max(ms_all),
            "refresh_median": round(statistics.median(refresh), 3) if refresh else None,
            "ordinary_median": round(statistics.median(ordinary), 3) if ordinary else None,
            "refresh_ticks": len(refresh),
        },
        "rss_gib": {"min": min(rss) if rss else None,
                    "max": max(rss) if rss else None},
        "cold": {
            "host_gib": round(cold.bytes_held / GIB, 3),
            "private_ssd_gib": round(st.get("kv_cold_ssd_bytes", 0) / GIB, 3),
            "shared_ram_gib": round(st.get("kv_cold_shared_bytes", 0) / GIB, 3),
            "shared_ssd_gib": round(st.get("kv_cold_shared_ssd_bytes", 0) / GIB, 3),
            "demotions": st.get("kv_cold_demotions", 0),
            "promotions": st.get("kv_cold_promotions", 0),
        },
        "per_tick_ms": ticks,    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_tick_ms"},
                     indent=2), flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
