"""Where does a SPECULATIVE tick's time go, in the engine, not in isolation?

BROKEN FOR THE CAPTURED DECODE PATH — use scripts/ab_draft_depth.py instead.
The wrap() below syncs around each method, which breaks CUDA-graph replay: every
tick re-captures, and the harness times graph construction. It read 0.4 tok/s
where the same config serves 48.4, and charged 4972 ms/tick to a draft step of a
few tens of ms (errors/2026-09-02-synchronize-inside-a-captured-graph.md). Kept
only for the eager path, where there is no graph to break.

prof_draft_step.py timed the draft with W=1 on a fresh KV and got 4.98 ms/step,
which predicted 41.6 tok/s at depth 3. Serving measured 2.7. So the isolated
timing misses most of the cost — this instruments the engine's real tick
(_run_forward / _draft_step / _verify) instead of a synthetic call.

  TILERL_TARGET=cuda python3 scripts/prof_spec_tick.py \
      --source <ckpt> --draft <shard> --depth 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl import cli
from tilerl import engine as eng
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.spec import load_draft
from tilerl_kernels.backend import get_backend

BUCKETS: dict[str, list[float]] = {}


def wrap(obj, name: str, label: str) -> None:
    """Time a bound method into BUCKETS, syncing so GPU work is attributed."""
    fn = getattr(obj, name)

    def timed(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize()
            BUCKETS.setdefault(label, []).append((time.perf_counter() - t0) * 1000)

    setattr(obj, name, timed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--ctx", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source  # cli binds this from the env at import time

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft) if args.depth else None
    e = build_engine(cfg, model, backend, num_blocks=512, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft, spec_depth=max(args.depth, 1))

    for meth, label in (("_run_forward", "run_forward"), ("_draft_step", "draft_step"),
                        ("_verify", "verify"), ("_run_decode_graph", "decode_graph"),
                        ("_sample_batch", "sample_batch"), ("_make_kv", "make_kv")):
        if hasattr(e, meth):
            wrap(e, meth, label)

    rid = e.submit(list(range(10, 10 + args.ctx)),
                   SamplingParams(temperature=0.0, max_new_tokens=args.tokens, seed=0))
    # Burn the prefill chunks BEFORE the clock: at ctx 1024+ they are most of the
    # wall time and would swamp the per-bucket percentages they do not belong to.
    from tilerl.engine import _PHASE_DECODE
    req = None
    while req is None or req.phase != _PHASE_DECODE:
        e.step()
        req = next((r for r in e._running if r.req_id == rid), None)
        if req is None:
            raise SystemExit(f"ctx={args.ctx}: finished during prefill")
    BUCKETS.clear()
    s0 = e.stats()
    t0 = time.perf_counter()
    done = None
    while done is None:
        e.step()
        done = e.poll().get(rid)
    wall = (time.perf_counter() - t0) * 1000
    n = e.stats()["tokens_generated"] - s0["tokens_generated"]
    s = {k: v - s0.get(k, 0) if isinstance(v, int | float) else v
         for k, v in e.stats().items()}

    print(f"\ngenerated {n} tokens in {wall:.0f} ms = {n / wall * 1000:.1f} tok/s "
          f"(ctx={args.ctx}, decode ticks only)")
    print(f"forwards: decode={s['decode_forwards']} prefill={s['prefill_forwards']} "
          f"mixed={s['mixed_forwards']}  accept={s['spec_accepted']}/{s['spec_drafted']}")
    print(f"{n / max(s['decode_forwards'], 1):.2f} tok/decode-forward\n")
    print(f"{'bucket':14s} {'calls':>6s} {'total ms':>10s} {'mean ms':>9s} {'% wall':>7s}")
    for label, xs in sorted(BUCKETS.items(), key=lambda kv: -sum(kv[1])):
        tot = sum(xs)
        print(f"{label:14s} {len(xs):6d} {tot:10.1f} {tot / len(xs):9.2f} {tot / wall * 100:6.1f}%")
    # run_forward nests decode_graph/verify; draft_step is the sibling that the
    # isolated profiler modelled at 4.98 ms/step.
    ds = BUCKETS.get("draft_step", [])
    if ds:
        print(f"\ndraft_step: {sum(ds) / len(ds):.1f} ms/tick "
              f"({sum(ds) / len(ds) / max(args.depth, 1):.1f} ms per depth step) "
              f"vs 4.98 ms/step measured with W=1 in isolation")


if __name__ == "__main__":
    main()
