"""Per-kernel profile of the captured decode graph at verify widths 1..5.
``_DecodeGraph.run`` only stages inputs and replays, so nothing is committed.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/profile_verify_replay.py \
        /data00/Qwen3.8-27B-NVFP4 --widths 1,2,3,5
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams, _DecodeGraph, build_engine
from tilerl.model import load_hf
from tilerl_kernels.backend import get_backend


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--batches", default="1")
    ap.add_argument("--widths", default="1,2,3,5")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--draft", help="build with a draft head so keep=W is exercised")
    args = ap.parse_args()

    from dataclasses import replace
    from torch.profiler import ProfilerActivity, profile

    backend = get_backend()
    base = qwen38_27b()
    cfg = replace(base, num_layers=args.layers,
                  full_attn_layers=tuple(i for i in base.full_attn_layers if i < args.layers))
    model = load_hf(cfg, args.source)
    draft = None
    if args.draft:
        from tilerl.spec import load_draft

        draft = load_draft(model, args.draft)
    engine = build_engine(cfg, model, backend, num_blocks=1024, num_slots=16,
                          decode_graph=True, draft=draft, spec_depth=4)

    gen = torch.Generator().manual_seed(7)
    batches = [int(x) for x in args.batches.split(",")]
    for _ in range(max(batches)):
        engine.submit(torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist(),
                      SamplingParams(temperature=0.0, max_new_tokens=4096))
    for _ in range(8):
        engine.step()
    prev = None
    for B in batches:
      rows = list(engine._running)[:B]
      for W in (int(x) for x in args.widths.split(",")):
          chains = [[r.output[-1]] * W for r in rows] if W > 1 else None
          # keep=W as in a real verify tick (state written per chain step); needs --draft.
          keep = W if (chains and draft is not None) else 0
          g = _DecodeGraph(model, backend, engine._kv, engine._states, B,
                           width=W, keep=keep)
          ms = timed(lambda: g.run(rows, chains))
          with profile(activities=[ProfilerActivity.CUDA]) as prof:
              for _ in range(5):
                  g.run(rows, chains)
              torch.cuda.synchronize()
          by: dict[str, list] = defaultdict(lambda: [0, 0.0])
          tot = 0.0
          for e in prof.events():
              if e.device_type.name != "CUDA":
                  continue
              us = e.time_range.elapsed_us() / 5
              by[e.name[:52]][0] += 1
              by[e.name[:52]][1] += us
              tot += us
          print(f"\n=== W={W} B={B}: replay {ms:.3f} ms, GPU-busy {tot / 1e3:.2f} ms, "
                f"{sum(c for c, _ in by.values()) // 5} kernels ===")
          print(f"{'kernel':<52} {'n':>6} {'us ea':>8} {'ms':>8}")
          for name, (c, us) in sorted(by.items(), key=lambda kv: -kv[1][1])[: args.top]:
              print(f"{name:<52} {c // 5:>6} {us / c * 5:>8.1f} {us / 1e3:>8.3f}")
          if W == 1:
              # the tick is more than the replay; show the host half.
              step = timed(lambda: engine.step(), reps=5)
              print(f"  engine.step() {step:.3f} ms  -> {step - ms:.3f} ms outside the replay")
          if prev is not None:
              print("  delta vs previous width, ms:")
              for name in sorted(set(by) | set(prev), key=lambda n: -(by[n][1] - prev.get(n, [0, 0])[1])):
                  d = (by[name][1] - prev.get(name, [0, 0.0])[1]) / 1e3
                  if abs(d) > 0.2:
                      print(f"    {name:<52} {d:+8.3f}")
          prev = {k: list(v) for k, v in by.items()}


if __name__ == "__main__":
    main()
