"""Attribute a spec tick's CUDA kernels to REGIONS, by wrapping the model's own
methods in `record_function` and reading the profiler's parent range.

Differencing configurations (errors/2026-09-02-differencing-attributed-the-trunk-to-the-draft.md)
said one draft forward adds
955 torch elementwise launches while each further depth step adds 29 -- a 33x
first-vs-later ratio where tilelang scales 4.6x. That is not something one
1-layer draft forward can do, so the difference was attributing more than the
draft: `spec d1` and `dense` differ in the trunk too (a verify tick runs W rows,
a dense tick runs 1), and everything that scales with rows landed in the "draft"
column.

So attribute directly instead. Monkeypatching is the whole instrument: each
region enters a `record_function`, and torch's profiler nests CUDA events under
the CPU range that launched them. Launch counts and summed kernel durations both
survive the profiler's serialization (it reads 121.5 ms/tick where the tick is
66.46) -- wall clock and host share do not, and are not printed.

Regions: trunk.attn, trunk.gdn, trunk.mlp, trunk.linear, draft.fwd, sample.
Everything left over is `-` (bookkeeping between regions: index_copy, block-table
builds, the draft-KV zeroing at engine.py:926).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl_kernels.backend import get_backend

from tilerl import model as model_mod
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.spec import DraftHead, load_draft

TORCH_MARKS = ("elementwise_kernel", "index_elementwise", "unrolled_elementwise",
               "vectorized_elementwise", "reduce_kernel", "CatArrayBatched")


def _wrap(obj, name: str, region: str) -> None:
    fn = getattr(obj, name)

    def wrapped(*a, **k):
        with torch.profiler.record_function("R:" + region):
            return fn(*a, **k)

    setattr(obj, name, wrapped)


def instrument() -> None:
    """Wrap on the CLASS, so it covers trunk and draft-head layers alike."""
    for meth, region in (("_full_attn", "trunk.attn"), ("_gdn", "trunk.gdn"),
                         ("_mlp", "trunk.mlp"), ("_linear", "trunk.linear")):
        _wrap(model_mod.Model, meth, region)
    _wrap(DraftHead, "forward", "draft.fwd")
    _wrap(DraftHead, "confidence", "draft.conf")


def attribute(e, reps: int = 2, settle: int = 6) -> dict:
    """{region: [launches, gpu_us]} for one steady tick, torch-only and total."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(settle):
        e.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(reps):
            e.step()
        torch.cuda.synchronize()

    # Region per CPU-range id, then per kernel via its launching range's ancestry.
    ev = prof.events()
    region_of: dict[int, str] = {}
    for x in ev:
        if x.device_type.name != "CPU":
            continue
        name = x.name
        r = name[2:] if name.startswith("R:") else None
        if r is None:
            p = x.cpu_parent
            while p is not None and not p.name.startswith("R:"):
                p = p.cpu_parent
            r = p.name[2:] if p is not None else "-"
        region_of[id(x)] = r

    out: dict = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # n, us, torch_n, torch_us
    for x in ev:
        if x.device_type.name != "CPU" or not x.kernels:
            continue
        r = region_of.get(id(x), "-")
        for k in x.kernels:
            us = k.duration / reps
            torchy = any(m in k.name for m in TORCH_MARKS)
            out[r][0] += 1 / reps
            out[r][1] += us
            if torchy:
                out[r][2] += 1 / reps
                out[r][3] += us
                # x.name is the aten op that launched this kernel -- the profiler
                # already carries the attribution `with_stack` could not give.
                # Input shapes separate two callers of one op (a state gather and
                # a window gather are both aten::index).
                shp = ";".join(str(list(s)) for s in (x.input_shapes or []) if s)[:60]
                out[f"O:{r}/{x.name} {shp}"][2] += 1 / reps
                out[f"O:{r}/{x.name} {shp}"][3] += us
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--prompt", type=int, default=30)
    ap.add_argument("--settle", type=int, default=6)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")

    instrument()
    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True, source=args.source)
    draft = load_draft(model, args.draft) if args.depth else None
    e = build_engine(cfg, model, backend, num_blocks=512, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft, spec_depth=args.depth,
                     decode_graph=False)
    rid = e.submit(list(range(10, 10 + args.prompt)),
                   SamplingParams(temperature=0.0, max_new_tokens=64, seed=0))
    while not any(r.req_id == rid and r.phase == 2 for r in e._running):
        e.step()
    got = attribute(e, settle=args.settle)

    print(f"\ndepth={args.depth} prompt={args.prompt}   decode_graph=off (regions need CPU ranges)")
    print(f"{'region':<14} {'n':>7} {'ms':>8} {'torch n':>8} {'torch ms':>9}")
    rows = [(k, v) for k, v in got.items() if not k.startswith("O:")]
    for k, v in sorted(rows, key=lambda kv: -kv[1][1]):
        print(f"{k:<14} {v[0]:>7.0f} {v[1] / 1e3:>8.2f} {v[2]:>8.0f} {v[3] / 1e3:>9.2f}")
    tot = [sum(v[i] for _, v in rows) for i in range(4)]
    print(f"{'TOTAL':<14} {tot[0]:>7.0f} {tot[1] / 1e3:>8.2f} {tot[2]:>8.0f} {tot[3] / 1e3:>9.2f}")

    print(f"\n{'region / aten op / input shapes':<76} {'n':>7} {'ms':>8}")
    for k, v in sorted(got.items(), key=lambda kv: -kv[1][3])[:30]:
        if k.startswith("O:") and v[2]:
            print(f"{k[2:]:<76} {v[2]:>7.0f} {v[3] / 1e3:>8.3f}")


if __name__ == "__main__":
    main()
