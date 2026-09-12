#!/usr/bin/env python3
"""Where the 256k sparse prefill seconds go — spill write vs read-through vs rest.

Step-1 mechanism probe for the ~16 MiB/s spill-write attribution in the #556
win entry. The disk's own bound (DIRECT/buffered 4 GiB, fsync-per-page 12 GiB)
is ~175-190 MiB/s, 10x the observed production rate, so a rate match does not
prove the mmap write sets prefill wall-clock. This wraps ColdSsdFile.write /
read / read_field with cumulative perf-counter timers and reports their share
of the 997.6 s prefill. If write_s is a small fraction of prefill_s, the write
is not the bound and batching it cannot recover the time.

H20 card 3 (same arms as trace_256k_spill_rss.py):
  python3 scripts/trace_256k_spill_time.py \
      --source /work/tilerl-ckpt/Qwen3.8-27B-NVFP4 --spill /work/tilerl-s-5f/spill.bin
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import ColdSsdFile

GIB = 1 << 30
MIB = 1 << 20

# cumulative across both spill files (private + prefix-shared)
T = {"write": 0.0, "write_bytes": 0, "write_n": 0,
     "read": 0.0, "read_bytes": 0, "read_n": 0,
     "read_field": 0.0, "read_field_n": 0}


def _wrap() -> None:
    orig_write = ColdSsdFile.write
    orig_read = ColdSsdFile.read
    orig_field = ColdSsdFile.read_field

    def write(self, key, blob):
        t0 = time.perf_counter()
        r = orig_write(self, key, blob)
        T["write"] += time.perf_counter() - t0
        T["write_n"] += 1
        T["write_bytes"] += self.stride
        return r

    def read(self, key, pin=False):
        t0 = time.perf_counter()
        r = orig_read(self, key, pin)
        T["read"] += time.perf_counter() - t0
        T["read_n"] += 1
        T["read_bytes"] += self.stride
        return r

    def read_field(self, key, field, pin=False):
        t0 = time.perf_counter()
        rf = orig_field(self, key, field, pin)
        T["read_field"] += time.perf_counter() - t0
        T["read_field_n"] += 1
        return rf

    ColdSsdFile.write = write
    ColdSsdFile.read = read
    ColdSsdFile.read_field = read_field


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--spill", required=True)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--host-gib", type=float, default=6.0)
    ap.add_argument("--ssd-gib", type=float, default=12.0)
    args = ap.parse_args()

    os_env = __import__("os").environ
    os_env.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    _wrap()
    backend = get_backend()
    cfg, model = cli._build_model("qwen38-27b", seed=0, fuse_projections=True)
    engine = build_engine(
        cfg=cfg, model=model, backend=backend,
        num_slots=1, max_batch=1,
        max_total_tokens=args.ctx + 256, max_num_batched_tokens=512,
        sparse_k=args.k, scorer="bounds",
        kv_cold_bytes=int(args.host_gib * GIB),
        cold_ssd_path=args.spill, cold_ssd_bytes=int(args.ssd_gib * GIB))

    prompt = np.arange(args.ctx, dtype=np.int64) % cfg.vocab_size
    params = SamplingParams(temperature=0.0, max_new_tokens=args.decode_tokens, seed=0)

    t0 = time.perf_counter()
    rid = engine.submit(prompt, params)
    prefill_s = None
    while True:
        d = engine.poll()
        if rid in d and len(d[rid]) >= args.decode_tokens:
            break
        engine.step()
        r = next((x for x in engine._running if x.req_id == rid), None)
        if r is not None and r.phase == 2 and prefill_s is None:
            prefill_s = time.perf_counter() - t0
            print(f"PREFILL_DONE {prefill_s:.1f}s", flush=True)
    total_s = time.perf_counter() - t0

    st = engine._kv.cold.stats()
    wgib = T["write_bytes"] / GIB
    rgib = T["read_bytes"] / GIB
    print({
        "prefill_s": round(prefill_s, 1),
        "total_s": round(total_s, 1),
        "spill_write_s": round(T["write"], 1),
        "spill_write_pct_of_prefill": round(100 * T["write"] / prefill_s, 1),
        "spill_write_n": T["write_n"],
        "spill_write_gib": round(wgib, 2),
        "spill_write_effective_MiB_s": round(wgib * 1024 / T["write"], 1),
        "ssd_read_s": round(T["read"], 2),
        "ssd_read_n": T["read_n"],
        "ssd_read_gib": round(rgib, 3),
        "read_field_s": round(T["read_field"], 2),
        "read_field_n": T["read_field_n"],
        "demotions": st.get("kv_cold_demotions", 0),
        "promotions": st.get("kv_cold_promotions", 0),
        "unaccounted_prefill_s": round(prefill_s - T["write"] - T["read"] - T["read_field"], 1),
    }, flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
