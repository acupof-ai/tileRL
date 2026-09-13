"""Hybrid wall-time fairness: is the fairness clock charged at a synchronized
tick boundary? (probe for #586, V100)

The scheduler gives dense ticks until their accumulated wall time reaches the
last sparse tick's. On an async CUDA backend a sparse prefill tick can RETURN
while its GPU work is still queued; if the charge is read before a
synchronize, the sparse tick looks short, dense is under-served, and the work
is actually waited on by the next dense tick's sample sync.

This drives a real hybrid engine with one short dense and one long sparse
request and records, per tick: mode, unsynced wall time (perf_counter around
_run_forward), and SYNCED wall time (perf_counter with
torch.cuda.synchronize() at both ends). If sparse ticks look much shorter
unsynced than synced, the scheduler must charge at a synced boundary.

Run (V100):
  TILERL_TARGET=cuda /usr/bin/python3 scripts/probe_hybrid_tick_clock.py \
      --source /path/to/qwen38-27b --draft /path/to/model_mtp.safetensors
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--long-tokens", type=int, default=32768)
    args = ap.parse_args()

    import torch
    from tilerl_kernels.backend import get_backend

    from tilerl import cli
    from tilerl.cli import _build_model
    from tilerl.engine import SamplingParams, build_engine

    cli._QWEN38_SOURCE = args.source
    be = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    from tilerl.spec import load_draft

    e = build_engine(
        cfg, model, be, num_slots=args.slots, max_batch=args.slots,
        max_total_tokens=args.ctx, sparse_k=128, scorer="bounds",
        kv_cold_bytes=1 << 33, sparse_min_tokens=8192,
        decode_graph=True, draft=load_draft(model, args.draft), spec_depth=1)

    rng = np.random.default_rng(3)
    e.submit((rng.integers(3, 300, args.long_tokens)).astype(np.int64),
             SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    e.submit((rng.integers(3, 300, 512)).astype(np.int64),
             SamplingParams(temperature=0.0, max_new_tokens=200, seed=1))

    # Wrap _run_forward to time each tick both ways across separate runs is
    # impossible in one process (work is issued once), so instead time with an
    # explicit sync at the START: unsynced = enqueue-to-return; synced =
    # enqueue-to-completion of the PRIOR queue drained first.
    orig = e._run_forward
    rows = []

    # Time each tick with a sync BEFORE the measurement (a clean queue start):
    # unsynced = enqueue-to-return; synced = enqueue-to-completion. If sparse
    # ticks look much shorter unsynced, their GPU work spills to the next tick.
    def timed(decodes, prefills, chunks):
        rr = decodes + prefills
        mode = "sparse" if rr and rr[0].sparse_on else "dense"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        orig(decodes, prefills, chunks)
        t_unsynced = time.perf_counter() - t0
        torch.cuda.synchronize()
        t_synced = time.perf_counter() - t0
        rows.append((mode, t_unsynced * 1000, t_synced * 1000))

    e._run_forward = timed
    for _ in range(4000):
        e.step()
        if not (e._running or e._waiting):
            break

    sp = [r for r in rows if r[0] == "sparse"]
    de = [r for r in rows if r[0] == "dense"]

    def agg(xs, i):
        v = sorted(x[i] for x in xs)
        return (sum(v) / len(v), v[len(v) // 2]) if v else (0.0, 0.0)

    print(f"{'mode':8s} {'n':>4s} {'unsynced ms (mean/med)':>26s} {'synced ms':>20s}")
    for name, xs in (("sparse", sp), ("dense", de)):
        u = agg(xs, 1)
        s = agg(xs, 2)
        print(f"{name:8s} {len(xs):4d} {u[0]:12.2f} / {u[1]:8.2f}     "
              f"{s[0]:8.2f} / {s[1]:8.2f}")
    if sp:
        us = agg(sp, 1)[0]
        ss = agg(sp, 2)[0]
        print(f"\nsparse unsynced/synced ratio: {us / ss:.3f} "
              f"(<<1 means GPU work spills to the next dense tick -> charge at a "
              "synced boundary)")
    e.shutdown()


if __name__ == "__main__":
    main()
