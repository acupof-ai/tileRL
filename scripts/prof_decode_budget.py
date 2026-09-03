"""Per-kernel budget of one decode token: where do the 33 ms go?

Attention is now ~2.6 ms of a 33 ms token at 4096 ctx (wins/2026-09-01-sm70-
attention-thread-redundancy.md), so 30 ms is elsewhere and unattributed. The
weight roofline says 17.8 ms of it is unavoidable — the 16.04 GB a dense token
streams (trunk + lm_head) at 900 GB/s. This attributes the rest to kernels by
name, which is the step that turns "30 ms somewhere" into a target.

torch.profiler rather than another hand-rolled timer: the decode path is CUDA
-graph captured, and a graph replay shows up as its constituent kernels here
while manual event timing around the replay only gives the total.

  scripts/v100.sh run bud '/usr/bin/python3 -u scripts/prof_decode_budget.py'
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
from tilerl_kernels.backend import get_backend
from torch.profiler import ProfilerActivity, profile

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import _PHASE_DECODE, SamplingParams, build_engine
from tilerl.spec import load_draft

#: Kernel-name substring -> the op class it belongs to. First match wins, so
#: order matters: the fp4 GEMV names contain "gemv", attention contains "attn".
CLASSES = [
    ("attention", ("paged_attention", "attn")),
    ("fp4 GEMV", ("linear_fp4", "gemv")),
    ("GDN", ("gdn",)),
    ("rmsnorm", ("rmsnorm", "norm")),
    ("rope", ("rope",)),
    ("kv write", ("write_tokens", "scatter", "copy_kv")),
    ("sampler", ("softmax", "topk", "sort", "multinomial", "argmax")),
    ("elementwise", ("silu", "mul", "add", "cast", "convert", "elementwise")),
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
    ap.add_argument("--draft")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=16)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft) if args.draft else None
    e = build_engine(cfg, model, backend, num_blocks=1024, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft, spec_depth=3 if draft else 1)

    def to_decode(tokens: int):
        """Submit and burn the prefill chunks; return (rid, request)."""
        rid = e.submit(list(range(10, 10 + args.ctx)),
                       SamplingParams(temperature=0.0, max_new_tokens=tokens, seed=0))
        req = None
        while req is None or req.phase != _PHASE_DECODE:
            e.step()
            req = next((r for r in e._running if r.req_id == rid), None)
            if req is None:
                raise SystemExit(f"ctx={args.ctx}: finished during prefill")
        return rid

    def drain(rid: int) -> int:
        s0 = e.stats()["tokens_generated"]
        out = None
        while out is None:
            e.step()
            out = e.poll().get(rid)
        return e.stats()["tokens_generated"] - s0

    for _ in range(2):  # warm: JIT, graph capture, allocator
        drain(to_decode(args.tokens))
    torch.cuda.synchronize()

    # Profile the DECODE ticks only. A 4096 prompt is 8 chunked-prefill ticks
    # and they dwarf a decode tick, so profiling across them reported 8217
    # ms/token and 2900 GEMV calls/token — the prefill's work over the decode's
    # token count.
    rid = to_decode(args.tokens)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        n = drain(rid)
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000 / max(n, 1)

    by_cls: dict[str, float] = defaultdict(float)
    by_kernel: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    for ev in prof.key_averages():
        us = getattr(ev, "self_device_time_total", 0) or 0
        if us <= 0:
            continue
        by_cls[classify(ev.key)] += us
        ms, cnt = by_kernel[ev.key]
        by_kernel[ev.key] = (ms + us, cnt + ev.count)

    total = sum(by_cls.values())
    label = "spec d3" if draft else "dense"
    gpu_ms = total / 1000 / max(n, 1)
    print(f"\n# {label}, ctx={args.ctx}, {n} tokens")
    print(f"# {gpu_ms:.2f} ms/token GPU vs {wall_ms:.2f} ms/token wall")
    print("# roofline: 16.04 GB streamed / 900 GB/s = 17.8 ms/token = 56.1 tok/s")
    print("#   (trunk 15.24 + lm_head 0.80; embed_tokens and the visual tower are")
    print("#    resident but not streamed — errors/2026-09-02-roofline-is-the-streamed-subset)")
    # A profile that does not roughly reconcile with the clock is measuring the
    # wrong window — profiling across the prefill chunks once read 8217 ms/token
    # against a 33 ms token.
    if gpu_ms > 2 * wall_ms:
        print(f"\n!! {gpu_ms:.0f} ms GPU inside a {wall_ms:.0f} ms token — the window "
              "includes work that is not this token's. Numbers below are not usable.")
    print()
    print(f"{'class':>14} {'ms/tok':>8} {'% GPU':>7}")
    for cls, us in sorted(by_cls.items(), key=lambda kv: -kv[1]):
        print(f"{cls:>14} {us/1000/max(n,1):>8.2f} {100*us/total:>6.1f}%")

    print(f"\n{'kernel':>52} {'ms/tok':>8} {'calls/tok':>10}")
    for name, (us, cnt) in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:20]:
        print(f"{name[-52:]:>52} {us/1000/max(n,1):>8.2f} {cnt/max(n,1):>10.1f}")


if __name__ == "__main__":
    main()
