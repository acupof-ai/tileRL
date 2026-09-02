"""Per-kernel GPU time of one prefill forward, serial GDN vs fla chunked GDN, in one process.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/profile_prefill.py /data00/Qwen3.8-27B-NVFP4
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import replace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--len", dest="length", type=int, default=512)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile

    from tilerl.config import qwen38_27b
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.model import load_hf
    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    base = qwen38_27b()
    cfg = replace(base, num_layers=args.layers,
                  full_attn_layers=tuple(i for i in base.full_attn_layers if i < args.layers))
    model = load_hf(cfg, args.source)
    gen = torch.Generator().manual_seed(3)
    prompt = torch.randint(0, cfg.vocab_size, (args.length,), generator=gen).tolist()

    def one_prefill():
        engine = build_engine(cfg, model, backend, num_blocks=args.length // 16 + 64,
                              num_slots=8, max_total_tokens=args.length + 64)
        engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=1))
        while engine.stats()["running"] or engine.stats()["waiting"]:
            engine.step()

    for arm, env in (("shipped serial GDN", {}),
                     ("fla chunked GDN", {"TILERL_GDN_CHUNKWISE": "64", "TILERL_GDN_FLA": "1"})):
        old = {k: os.environ.get(k) for k in ("TILERL_GDN_CHUNKWISE", "TILERL_GDN_FLA")}
        os.environ.pop("TILERL_GDN_CHUNKWISE", None)
        os.environ.pop("TILERL_GDN_FLA", None)
        os.environ.update(env)
        try:
            one_prefill()  # warm JIT
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                one_prefill()
                torch.cuda.synchronize()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
        by: dict = defaultdict(lambda: [0, 0.0])
        tot = 0.0
        for e in prof.events():
            if e.device_type.name != "CUDA":
                continue
            us = e.time_range.elapsed_us()
            by[e.name[:52]][0] += 1
            by[e.name[:52]][1] += us
            tot += us
        n = sum(c for c, _ in by.values())
        print(f"\n=== {arm}: GPU-busy {tot / 1e3:.1f} ms, {n} kernels ===")
        print(f"{'kernel':<52} {'n':>6} {'ms':>8}")
        for name, (c, us) in sorted(by.items(), key=lambda kv: -kv[1][1])[: args.top]:
            print(f"{name:<52} {c:>6} {us / 1e3:>8.2f}")


if __name__ == "__main__":
    main()
