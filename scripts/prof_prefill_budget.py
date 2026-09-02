"""Per-kernel budget of ONE prefill tick: what is prefill's 8.4 ms/token?

Prefill is 8.35-9.37 ms per prompt token after the rung-32 flag fix
(wins/2026-09-02-rung-8-cliff-was-a-missing-flag.md), down from 31. The question
is no longer "why is it slower per token than decode" — it isn't — but how far
the remainder is from what the card can do.

Use the FLOP roofline here, not the byte one. At M=512 a chunk re-reads the
weights once and does 512 rows of work against them, so prefill is compute-bound
where decode is bandwidth-bound: 51.2 GFLOP/row x 512 rows = 26.2 TFLOP per
chunk, and V100's fp16 SCALAR peak is 31.4 TFLOPS (the fp4 extern is scalar FMA;
mma.sync's 125 TFLOPS is unreachable from this path). That puts a 512-row chunk's
floor near 836 ms against a measured 4277 — about 5x, not the ~50x a byte
roofline suggests. A byte roofline at M=512 is the wrong denominator by a factor
of M.

The suspect is stated so the output can refute it: gdn_chunk_fused is 48 of 64
layers and its own entry records "one block per batch x head, ~6% SM
utilization", tuned for launch count rather than occupancy. A serial scan that
does not scale with T yields exactly this signature.

Prefill runs eager (no graph capture), so torch.profiler sees the real kernels
directly and there is no replay to see through.

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

    def submit(base: int) -> int:
        return e.submit(list(range(base, base + args.ctx)),
                        SamplingParams(temperature=0.0, max_new_tokens=1, seed=0))

    def drain(rid: int) -> None:
        while e.poll().get(rid) is None:
            e.step()

    drain(submit(10))  # warm: JIT every prefill-shaped kernel

    # Profile the FIRST step after submitting, not a later one: at ctx == the
    # chunk budget the prompt is one chunk, so step() #1 does the whole prefill
    # AND emits the token. Profiling a second step then windows on an empty
    # queue and reports 0 ms — which it did, before this was split up.
    rid = submit(500000)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        t0 = time.perf_counter()
        e.step()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000
    drain(rid)

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
    # An empty window is not a measurement. This read 0 ms once (the profiled
    # step ran after the prefill had already happened) and printed a clean table
    # of nothing, which is the failure mode that looks like a result.
    if not by_cls or total <= 1.0:
        raise SystemExit("no CUDA kernels in the window — the profiled step did no work")

    print(f"\n# ONE prefill tick, {args.ctx} rows: {total / 1000:.0f} ms GPU / {wall:.0f} ms wall")
    # FLOP, not bytes: at M=512 the chunk re-reads the weights once and does 512
    # rows against them, so the byte roofline is off by a factor of M here.
    tflop = 2 * 25.62e9 * args.ctx / 1e12
    floor = tflop / 31.4 * 1000
    print(f"# {total / 1000 / args.ctx:.2f} ms GPU per prompt token "
          f"(decode is 26.6 ms/token at M=1, but that path is bandwidth-bound)")
    print(f"# {tflop:.1f} TFLOP this tick / 31.4 TFLOPS fp16 scalar = {floor:.0f} ms floor "
          f"-> {total / 1000 / floor:.1f}x off")
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
