"""Wall clock beside GPU-busy for one prefill, serial GDN vs fla, so inter-kernel gaps are a
number rather than a subtraction of two runs."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--len", dest="length", type=int, default=512)
    ap.add_argument("--iters", type=int, default=5)
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
    engine = build_engine(cfg, model, backend, num_blocks=args.length // 16 + 128,
                          num_slots=8, max_total_tokens=args.length + 64)

    def one():
        rid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=1))
        done = {}
        while rid not in done:
            engine.step()
            done.update(engine.poll())

    for arm, env in (("shipped serial GDN", {}),
                     ("fla chunked GDN", {"TILERL_GDN_CHUNKWISE": "64", "TILERL_GDN_FLA": "1"})):
        old = {k: os.environ.get(k) for k in ("TILERL_GDN_CHUNKWISE", "TILERL_GDN_FLA")}
        for k in old:
            os.environ.pop(k, None)
        os.environ.update(env)
        try:
            one()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                one()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) / args.iters * 1e3
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(args.iters):
                    one()
                torch.cuda.synchronize()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
        n = busy = 0
        for e in prof.events():
            if e.device_type.name == "CUDA":
                n += 1
                busy += e.time_range.elapsed_us()
        n //= args.iters
        busy = busy / args.iters / 1e3
        print(f"{arm:>20}: wall {wall:7.2f} ms  gpu-busy {busy:7.2f} ms  "
              f"gaps {wall - busy:7.2f} ms  kernels {n:>5}  "
              f"gap/kernel {(wall - busy) * 1e3 / max(n, 1):5.2f} us")


if __name__ == "__main__":
    main()
