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
from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.spec import load_draft

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
    ap.add_argument("--batch", type=int, default=8,
                    help="concurrent requests. The default is 8 because the rung a verify "
                         "tick compiles keys on B*(1+depth), and only B=8 at depth 3 fills "
                         "the 32 rung serving uses (B=1 gives 4 rows -> the 4 rung, a "
                         "different kernel: errors/2026-09-03-the-spec-ncols-ab-ran-at-b1.md)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source  # cli binds this from the env at import time

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft) if args.depth else None
    # slots > max_batch: a tick narrower than its graph bucket keeps one slot for the
    # padding rows for good (engine.py:827) -- bench_ctx_decode.py died on this twice.
    e = build_engine(cfg, model, backend, num_blocks=512, num_slots=args.batch + 2,
                     max_batch=args.batch, max_total_tokens=8192, draft=draft,
                     spec_depth=max(args.depth, 1))

    for meth, label in (("_run_forward", "run_forward"), ("_draft_step", "draft_step"),
                        ("_verify", "verify"), ("_run_decode_graph", "decode_graph"),
                        ("_sample_batch", "sample_batch"), ("_make_kv", "make_kv")):
        if hasattr(e, meth):
            wrap(e, meth, label)

    rids = [e.submit(list(range(10 + i * args.ctx, 10 + (i + 1) * args.ctx)),
                     SamplingParams(temperature=0.0, max_new_tokens=args.tokens, seed=0))
            for i in range(args.batch)]
    # Burn the prefill chunks BEFORE the clock: at ctx 1024+ they are most of the
    # wall time and would swamp the per-bucket percentages they do not belong to.
    from tilerl.engine import _PHASE_DECODE
    # Bounded in WALL CLOCK, not just ticks: an empty tick costs nothing, so a tick cap
    # cannot tell spinning from working -- that is how a stall once ran six minutes
    # looking like a live job (errors/2026-09-03-the-spec-ncols-ab-ran-at-b1.md).
    deadline = time.perf_counter() + 300.0
    while True:
        e.step()
        reqs = [next((r for r in e._running if r.req_id == i), None) for i in rids]
        if any(r is None for r in reqs):
            raise SystemExit(f"ctx={args.ctx}: a request finished during prefill")
        if all(r.phase == _PHASE_DECODE for r in reqs):
            break
        if time.perf_counter() > deadline:
            raise SystemExit(f"ctx={args.ctx}: prefill stalled 300 s, "
                             f"{sum(r.phase == _PHASE_DECODE for r in reqs)}/{args.batch} "
                             f"in decode")
    BUCKETS.clear()
    s0 = e.stats()
    t0 = time.perf_counter()
    # Close at the FIRST completion: past that the batch runs narrower and dilutes the
    # per-bucket shares with ticks below the rung this profile is about.
    done: dict = {}
    deadline = time.perf_counter() + 600.0
    while not any(rid in done for rid in rids):
        e.step()
        done.update(e.poll())  # poll() drains _finished for EVERY request; keep it all
        if time.perf_counter() > deadline:
            raise SystemExit(f"ctx={args.ctx}: decode stalled 600 s, {len(done)} finished")
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
