#!/usr/bin/env python3
"""256k sparse prefill host-RSS trace for the SSD-spill prefix-copy fix (#556).

Builds the production sparse engine (k=128, bounds) with a SMALL pinned host
tier + mmap SSD spill on H20, prefills 262144 tokens, decodes 64 more, and
samples the python pid's VmRSS every --interval s alongside the cold-tier
COLD_STATS. Asserts total anonymous cold host bytes (private + shared) stay
within the host budget + one D2H batch through the whole prefill.

H20 card 3:
  scripts/pod_run.sh rss3 3 -- python3 scripts/trace_256k_spill_rss.py \
      --source /work/Qwen3.8-27B-NVFP4 --spill /work/tilerl-s-5f/spill.bin
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import numpy as np

from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS

GIB = 1 << 30
MIB = 1 << 20


def vm_rss_bytes() -> int:
    """Anonymous+file resident pages of this pid from /proc (Linux/pod)."""
    with open(f"/proc/{os.getpid()}/statm") as f:
        rss_pages = int(f.read().split()[1])
    return rss_pages * os.sysconf("SC_PAGE_SIZE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--spill", required=True)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--host-gib", type=float, default=6.0)
    ap.add_argument("--ssd-gib", type=float, default=12.0)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--out", default="/work/trace_256k_spill_rss.jsonl")
    args = ap.parse_args()

    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    backend = get_backend()
    cfg, model = cli._build_model("qwen38-27b", seed=0, fuse_projections=True)
    host_bytes = int(args.host_gib * GIB)
    ssd_bytes = int(args.ssd_gib * GIB)

    engine = build_engine(
        cfg=cfg, model=model, backend=backend,
        num_slots=1, max_batch=1,
        max_total_tokens=args.ctx + 256, max_num_batched_tokens=512,
        sparse_k=args.k, scorer="bounds",
        kv_cold_bytes=host_bytes,
        cold_ssd_path=args.spill, cold_ssd_bytes=ssd_bytes)
    cold = engine._kv.cold
    per_page = engine._kv.cold_page_nbytes()

    samples: list[dict] = []
    stop = threading.Event()

    def cold_host_bytes() -> int:
        # EVERY anonymous host container in the cold tier: private + shared RAM
        return int(cold.bytes_held)

    def sample(phase: str) -> dict:
        st = cold.stats()
        snap = {
            "t": round(time.time() - t0, 1),
            "phase": phase,
            "rss_gib": round(vm_rss_bytes() / GIB, 3),
            "cold_host_gib": round(cold_host_bytes() / GIB, 3),
            "private_ssd_gib": round(st.get("kv_cold_ssd_bytes", 0) / GIB, 3),
            "shared_ram_gib": round(st.get("kv_cold_shared_bytes", 0) / GIB, 3),
            "shared_ssd_gib": round(st.get("kv_cold_shared_ssd_bytes", 0) / GIB, 3),
            "demotions": st.get("kv_cold_demotions", 0),
            "promotions": st.get("kv_cold_promotions", 0),
        }
        samples.append(snap)
        return snap

    def loop(phase_box):
        while not stop.wait(args.interval):
            sample(phase_box[0])

    phase_box = ["init"]
    sampler = threading.Thread(target=loop, args=(phase_box,), daemon=True)
    t0 = time.time()
    sampler.start()

    prompt = (np.arange(args.ctx, dtype=np.int64) % cfg.vocab_size)
    params = SamplingParams(temperature=0.0, max_new_tokens=args.decode_tokens, seed=0)
    rid = engine.submit(prompt, params)

    phase_box[0] = "prefill"
    t_pre = time.time()
    # drain prefill then decode, sampling between steps
    while True:
        d = engine.poll()
        if rid in d and len(d[rid]) >= args.decode_tokens:
            break
        engine.step()
        r = next((x for x in engine._running if x.req_id == rid), None)
        if r is not None and r.phase == 2 and phase_box[0] == "prefill":
            prefill_s = time.time() - t_pre
            print(f"PREFILL_DONE {prefill_s:.1f}s", flush=True)
            phase_box[0] = "decode"

    prefill_s = None
    total_s = time.time() - t0
    stop.set()
    sampler.join(timeout=1)
    final = sample("done")

    budget_gib = args.host_gib
    # one D2H batch: up to a chunk's worth of departing pages, bounded loosely
    batch_cap_gib = ((512 // BLOCK_TOKENS + 40) * per_page) / GIB
    over = [s for s in samples if s["cold_host_gib"] > budget_gib + batch_cap_gib + 0.05]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(json.dumps({
        "total_s": round(total_s, 1),
        "host_budget_gib": budget_gib,
        "batch_cap_gib": round(batch_cap_gib, 3),
        "max_cold_host_gib": max(s["cold_host_gib"] for s in samples),
        "max_rss_gib": max(s["rss_gib"] for s in samples),
        "final": final,
        "over_budget_samples": over,
        "trace": args.out,
    }, indent=2), flush=True)

    assert not over, f"cold host bytes exceeded budget+batch: {over[:3]}"
    if total_s > 900:
        print("WARN: run exceeded 15 min check")
    engine.shutdown()


if __name__ == "__main__":
    main()
