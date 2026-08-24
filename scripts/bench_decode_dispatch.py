"""Measure the CPU-side dispatch cost of one decode tick: eager vs captured.

Times ``model.forward`` (eager) / ``_DecodeGraph.run`` (captured) WITHOUT
syncing — the CPU-side launch cost only, robust to GPU contention (the
co-tenant on a shared pod inflates wall-with-sync, not launch cost). The GPU
execution time is measured separately by ``profile_slice.py`` (CUDA events).

Usage:
    TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=1 \\
        PYTHONPATH=src python3 scripts/bench_decode_dispatch.py /host/tc27-nvfp4-slice2 --layers 2
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="HF checkpoint directory")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ticks", type=int, default=20)
    args = p.parse_args()

    from tilerl.config import qwen36_27b
    from tilerl.engine import SamplingParams, _DecodeGraph, build_engine
    from tilerl.model import load_hf
    from tilerl.ops.backend import get_backend

    backend = get_backend()
    if backend.device.type != "cuda":
        raise SystemExit("dispatch bench needs the CUDA target (TILERL_TARGET=cuda)")
    base = qwen36_27b()
    cfg = replace(
        base,
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in base.full_attn_layers if i < args.layers),
    )

    t0 = time.perf_counter()
    model = load_hf(cfg, args.source)
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)
    engine = build_engine(cfg, model, backend, num_blocks=128, num_slots=4, decode_graph=False)

    # Warmup: prefill + a few decodes (JIT every decode shape). Keep the
    # request alive past warmup so _running[0] exists for the timed loop.
    gen = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist()
    engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=args.ticks + 10, seed=0))
    for _ in range(4):
        engine.step()
    print("warmup: done", flush=True)

    req = engine._running[0]

    # Eager: time N forwards with NO sync — pure CPU-side dispatch. Allocate
    # the per-tick inputs inside the loop (the engine does too). Report the
    # MIN: on a contended pod the GPU occasionally idles, and the minimum
    # tick is the least-contaminated dispatch estimate.
    eager_samples = []
    for _ in range(args.ticks):
        ids = torch.tensor([[req.output[-1]]], dtype=torch.long, device=backend.device)
        pos = torch.tensor([[req.seq_len - 1]], dtype=torch.long, device=backend.device)
        kv = engine._make_kv([req])
        t0 = time.perf_counter()
        model.forward(ids, pos, kv, backend)
        eager_samples.append((time.perf_counter() - t0) * 1000.0)

    # Captured: build the graph (warmup forwards hit the JIT cache), then time
    # N replays with NO sync — copies + one graph replay. Decompose the pinned
    # copies (the g.run path) from the replay to locate the cost.
    g = _DecodeGraph(model, backend, engine._kv, engine._states, 1)
    captured_samples = []
    copy_samples = []
    replay_samples = []
    for _ in range(args.ticks):
        t0 = time.perf_counter()
        g.run([req])
        captured_samples.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        g._ids_h[0, 0] = req.output[-1]
        g._pos_h[0, 0] = req.seq_len - 1
        g._sl_h[0] = req.seq_len
        g._ss_h[0] = req.state_slot
        g._bt_h[0, : len(req.blocks)] = torch.tensor(req.blocks, dtype=torch.int32)
        g._ids.copy_(g._ids_h, non_blocking=True)
        g._pos.copy_(g._pos_h, non_blocking=True)
        g._sl.copy_(g._sl_h, non_blocking=True)
        g._ss.copy_(g._ss_h, non_blocking=True)
        g._bt.copy_(g._bt_h, non_blocking=True)
        copy_samples.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        g._graph.replay()
        replay_samples.append((time.perf_counter() - t0) * 1000.0)

    print(f"\nDECODE dispatch (CPU-side, no sync, {args.ticks} ticks, {args.layers} layers)")
    print(f"  {'':<20} {'min':>10} {'median':>10} {'max':>10}")
    for name, s in (
        ("eager forward", eager_samples),
        ("captured run", captured_samples),
        ("  pinned copies", copy_samples),
        ("  replay only", replay_samples),
    ):
        ss = sorted(s)
        print(f"  {name:<20} {ss[0]:>10.3f} {ss[len(ss) // 2]:>10.3f} {ss[-1]:>10.3f} ms/tick")
    print(f"  {'min speedup':<20} {min(eager_samples) / min(captured_samples):>10.1f}x")


if __name__ == "__main__":
    main()
