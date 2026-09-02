"""Per-kernel budget of ONE prefill tick: where do the 15.9 s go?

Prefill is 31 ms per prompt token against decode's 26.6 — a 512-row forward
costs more per token than a 1-row one, which is backwards. The arithmetic says
it behaves as if M=1: one prefill forward is 15.9 s, 512 rows decoded one at a
time would be 13.6 s, and 16 chunked GEMV launches re-streaming 14.40 GB each
should be 0.3 s. So something in the tick is linear in tokens with no batching
benefit, and it is worth ~50x.

prof_decode_budget.py answers the same question for decode; this windows on a
PREFILL tick instead. Prefill runs eager (no graph capture), so torch.profiler
sees the real kernels directly and there is no replay to see through.

The suspect is stated so the output can refute it: gdn_chunk_fused is 48 of 64
layers and its own entry records "one block per batch x head, ~6% SM
utilization", tuned for launch count rather than occupancy. A serial scan that
does not scale with T yields exactly this signature.

  scripts/v100.sh run pk 'CKPT=...; /usr/bin/python3 -u scripts/prof_prefill_budget.py \
      --source $CKPT --ctx 512'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.profiler import ProfilerActivity, profile

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl_kernels.backend import get_backend

#: Kernel-name substring -> op class. First match wins, so order matters.
CLASSES = [
    ("attention", ("paged_attention", "attn")),
    ("fp4 GEMV", ("linear_fp4", "gemv")),
    ("GDN", ("gdn", "chunk")),
    ("rmsnorm", ("rmsnorm", "norm")),
    ("rope", ("rope",)),
    ("kv write", ("write_tokens", "scatter", "copy_kv")),
    ("elementwise", ("silu", "mul", "add", "cast", "convert", "elementwise", "copy")),
]


def classify(name: str) -> str:
    low = name.lower()
    for cls, keys in CLASSES:
        if any(k in low for k in keys):
            return cls
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--ctx", type=int, default=512, help="one chunk at the default budget")
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    be = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    e = build_engine(cfg, model, be, num_blocks=1024, num_slots=4, max_batch=4,
                     max_total_tokens=8192)

    def one_prefill_tick(base: int) -> tuple[int, float]:
        """Submit, run exactly ONE tick, and report its wall ms."""
        rid = e.submit(list(range(base, base + args.ctx)),
                       SamplingParams(temperature=0.0, max_new_tokens=1, seed=0))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        return rid, ms

    rid, _ = one_prefill_tick(10)  # warm: JIT every prefill-shaped kernel
    while e.poll().get(rid) is None:
        e.step()

    rid, wall = one_prefill_tick(500000)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e.step()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000
    while e.poll().get(rid) is None:
        e.step()

    by_cls: dict[str, float] = defaultdict(float)
    by_kernel: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    for ev in prof.key_averages():
        us = getattr(ev, "self_device_time_total", 0) or 0
        if us <= 0:
            continue
        by_cls[classify(ev.key)] += us
        ms, cnt = by_kernel[ev.key]
        by_kernel[ev.key] = (ms + us, cnt + ev.count)
    total = sum(by_cls.values()) or 1.0

    print(f"\n# ONE prefill tick, {args.ctx} rows: {total / 1000:.0f} ms GPU / {wall:.0f} ms wall")
    print(f"# {total / 1000 / args.ctx:.2f} ms GPU per prompt token "
          f"(decode is 26.6 ms/token at M=1; the weight roofline is 16.0 ms per PASS)")
    if total / 1000 > 2 * wall:
        print(f"\n!! {total / 1000:.0f} ms GPU inside a {wall:.0f} ms tick — the window is wrong.")
    print(f"\n{'class':>14} {'ms/tick':>9} {'% GPU':>7} {'ms/token':>9}")
    for cls, us in sorted(by_cls.items(), key=lambda kv: -kv[1]):
        print(f"{cls:>14} {us / 1000:>9.1f} {100 * us / total:>6.1f}% "
              f"{us / 1000 / args.ctx:>9.3f}")
    print(f"\n{'kernel':>52} {'ms/tick':>9} {'calls':>7}")
    for name, (us, cnt) in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:15]:
        print(f"{name[-52:]:>52} {us / 1000:>9.1f} {cnt:>7}")


if __name__ == "__main__":
    main()
