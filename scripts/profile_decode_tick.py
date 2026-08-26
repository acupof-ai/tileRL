"""Decode tick per-op profile at B=1 and B=8 on the NVFP4 slice (dev tool).

CUDA-event breakdown of the eager decode tick — the exact kernel sequence the
shipped decode graph replays — plus the shipped graph tick (replay vs sampling
split) and a 64-layer extrapolation. Eager is instrumented because a captured
graph cannot be per-op timed from Python: the wrappers do not run during a
replay, so the graph run reports the total only.

Linears are timed at the Model._linear seam, so every GEMV/MMA carries its
projection name (qkvz / gate_up / lm_head / ...) and its dispatch path
(fp4-gemv / fp8-mma / ...). RMSNorms are named by their weight tensor.

Usage:
    CUDA_VISIBLE_DEVICES=6 TILERL_TARGET=cuda PYTHONPATH=src \\
        python3 scripts/profile_decode_tick.py /host/tc27-nvfp4-slice4 --layers 4
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace

import torch

from tilerl.config import qwen36_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import load_hf
from tilerl.ops.backend import get_backend

WARM = 12  # ticks: flushes B=8's 8 one-per-tick prefill admissions + 4 pure decodes
TICKS = 30


class Tracer:
    """CUDA-event spans per op, via instance-method wrappers (off until enabled)."""

    def __init__(self, backend, model):
        self.on = False
        self.records: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.linear_paths: dict[str, str] = {}
        id2name = {id(t): k for k, t in model.params.items()}

        # Named linears: wrap the model seam (not the backend) so each call
        # carries its projection key and dispatch path.
        orig_linear = model._linear

        def timed_linear(b, x, key):
            if not self.on:
                return orig_linear(b, x, key)
            m = x.numel() // x.shape[-1]
            if f"{key}.wq" in model.params:
                kind = "fp4"
            elif f"{key}.w8" in model.params:
                kind = "fp8"
            else:
                kind = "bf16"
            path = f"{kind}-{'gemv' if m == 1 else 'mma'}"
            self.linear_paths[key] = path
            return self._rec(f"linear[{path}]:{key}", orig_linear, b, x, key)

        model._linear = timed_linear

        # RMSNorm: name by weight identity (input_norm / q_norm / final_norm / ...).
        orig_rmsnorm = backend.rmsnorm

        def timed_rmsnorm(x, w, eps):
            if not self.on:
                return orig_rmsnorm(x, w, eps)
            return self._rec(f"rmsnorm:{id2name.get(id(w), '?')}", orig_rmsnorm, x, w, eps)

        backend.rmsnorm = timed_rmsnorm

        for name, orig in (
            ("rope", backend.rope),
            ("paged_attention", backend.paged_attention),
            ("write_tokens", backend.write_tokens),
            ("gdn_decode", backend.linear_attn_chunk),
            ("state_gather", backend.state_gather),
            ("state_scatter", backend.state_scatter),
            ("silu_mul", backend.silu_mul),
            ("add", backend.add),
            ("embedding", backend.embedding),
        ):
            self._wrap_simple(backend, name, orig)

    def install(self, engine):
        """Wrap one engine's sampling (argmax + the D2H result sync)."""
        orig_sample = engine._sample

        def timed_sample(logits_row, req, idx):
            if not self.on:
                return orig_sample(logits_row, req, idx)
            return self._rec("sample", orig_sample, logits_row, req, idx)

        engine._sample = timed_sample

    def wrap_method(self, obj, attr, label):
        """Time any method call (used for the captured graph's replay)."""
        orig = getattr(obj, attr)

        def timed(*args, **kwargs):
            if not self.on:
                return orig(*args, **kwargs)
            return self._rec(label, orig, *args, **kwargs)

        setattr(obj, attr, timed)

    def _wrap_simple(self, backend, name, orig):
        def wrapper(*args, **kwargs):
            if not self.on:
                return orig(*args, **kwargs)
            return self._rec(name, orig, *args, **kwargs)

        setattr(backend, name, wrapper)

    def _rec(self, label, fn, *args, **kwargs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(*args, **kwargs)
        e.record()
        self.records.append((label, s, e))
        return out

    def collect(self):
        torch.cuda.synchronize()
        for label, s, e in self.records:
            self.totals[label] = self.totals.get(label, 0.0) + s.elapsed_time(e)
            self.counts[label] = self.counts.get(label, 0) + 1
        self.records.clear()

    def per_tick(self, ticks):
        return {k: v / ticks for k, v in self.totals.items()}

    def calls_per_tick(self, ticks):
        return {k: v / ticks for k, v in self.counts.items()}

    def categories(self, ticks):
        cats: dict[str, float] = defaultdict(float)
        for k, v in self.totals.items():
            if k.startswith("linear["):
                cats["linears"] += v
            elif k.startswith("rmsnorm:"):
                cats["rmsnorm"] += v
            elif k in ("gdn_decode", "state_gather", "state_scatter"):
                cats["gdn-core"] += v
            elif k in ("paged_attention", "write_tokens", "rope"):
                cats["attn"] += v
            elif k == "sample":
                cats["sampling"] += v
            else:
                cats["other"] += v
        return {k: v / ticks for k, v in cats.items()}


def submit_batch(engine, cfg, b):
    gen = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist() for _ in range(b)]
    return [
        engine.submit(p, SamplingParams(temperature=0.0, max_new_tokens=WARM + TICKS + 2, seed=i))
        for i, p in enumerate(prompts)
    ]


def drain(engine, wids):
    done: dict[int, list[int]] = {}
    for _ in range(WARM + TICKS + 8):
        done.update(engine.poll())
        if all(w in done for w in wids):
            return
        engine.step()


def measure(engine, tracer, ticks):
    """Steady-state ticks: per-op events + wall (ms/tick avg)."""
    walls = []
    for _ in range(ticks):
        tracer.on = True
        t0 = time.perf_counter()
        engine.step()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        tracer.on = False
        tracer.collect()
    return sum(walls) / len(walls)


def print_eager(b, tracer, wall, ticks):
    per = tracer.per_tick(ticks)
    calls = tracer.calls_per_tick(ticks)
    gpu = sum(per.values())
    print(f"\nEAGER B={b} decode tick ({ticks}-tick avg)")
    print(f"  wall (host+events): {wall:.4f} ms   GPU-event sum: {gpu:.4f} ms")
    print(f"  {'op':<44} {'ms/tick':>9} {'%gpu':>6} {'calls':>6}")
    for name, ms in sorted(per.items(), key=lambda r: -r[1]):
        print(f"  {name:<44} {ms:>9.4f} {100 * ms / gpu:>5.1f}% {calls[name]:>6.1f}")
    cats = tracer.categories(ticks)
    print(
        "  categories: "
        + ", ".join(f"{k} {v:.4f}ms ({100 * v / gpu:.0f}%)" for k, v in cats.items())
    )
    return gpu, cats


def print_graph(b, tracer, wall, ticks):
    per = tracer.per_tick(ticks)
    replay = per.get("graph_replay", 0.0)
    smp = per.get("sample", 0.0)
    host = wall - replay - smp
    print(f"\nGRAPH B={b} decode tick ({ticks}-tick avg, shipped path)")
    print(
        f"  wall:        {wall:.4f} ms  ({1000 / wall:.1f} tok/s per-request, "
        f"{1000 * b / wall:.1f} aggregate)"
    )
    print(f"  replay:      {replay:.4f} ms ({100 * replay / wall:.1f}%)")
    print(f"  sampling:    {smp:.4f} ms ({100 * smp / wall:.1f}%)")
    print(f"  host+copies: {host:.4f} ms ({100 * host / wall:.1f}%)")
    return replay, smp, host


def lm_head_ms(tracer, ticks):
    for k, v in tracer.totals.items():
        if k.endswith(":lm_head"):
            return v / ticks
    return 0.0


def extrapolate(replay_ms, lm_head, fixed_ms, slice_layers, full_layers=64):
    # Final-bench method (2026-08-25-gemv-instr-gdn-wy): per-layer cost x
    # (full/slice), lm_head + fixed counted once. embed/final_norm stay in
    # the per-layer term (negligible: ~0.03 ms of a ~20 ms tick).
    per_layer = (replay_ms - lm_head) / slice_layers
    return per_layer * full_layers + lm_head + fixed_ms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batches", type=str, default="1,8")
    args = p.parse_args()

    backend = get_backend()
    if backend.device.type != "cuda":
        raise SystemExit("profiling needs the CUDA target (TILERL_TARGET=cuda)")
    base = qwen36_27b()
    cfg = replace(
        base,
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in base.full_attn_layers if i < args.layers),
    )
    t0 = time.perf_counter()
    model = load_hf(cfg, args.source, fuse_projections=True)
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

    batches = [int(x) for x in args.batches.split(",")]
    report = {"layers": args.layers, "batches": {}}

    # Eager engine first: its warmup JITs every decode shape, so the graph
    # engine's capture hits warm tilelang artifacts (no NVCC in capture).
    eager = build_engine(cfg, model, backend, num_blocks=512, num_slots=8, decode_graph=False)
    tracer = Tracer(backend, model)
    tracer.install(eager)
    for b in batches:
        print(f"\n=== eager B={b}: warmup ({WARM} ticks; first pass pays JIT) ===", flush=True)
        wids = submit_batch(eager, cfg, b)
        for _ in range(WARM):
            eager.step()
        torch.cuda.synchronize()
        wall = measure(eager, tracer, TICKS)
        drain(eager, wids)
        gpu, cats = print_eager(b, tracer, wall, TICKS)
        report["batches"].setdefault(b, {})["eager"] = {
            "wall_ms": wall,
            "gpu_sum_ms": gpu,
            "per_op_ms": tracer.per_tick(TICKS),
            "calls_per_tick": tracer.calls_per_tick(TICKS),
            "categories_ms": cats,
            "lm_head_ms": lm_head_ms(tracer, TICKS),
        }
        tracer.totals.clear()
        tracer.counts.clear()

    # Graph engine (shipped): replay total + sampling split. Same backend and
    # model, so the wrappers and the warm JIT carry over.
    graph = build_engine(cfg, model, backend, num_blocks=512, num_slots=8, decode_graph=True)
    tracer.install(graph)
    for b in batches:
        print(
            f"\n=== graph B={b}: warmup ({WARM} ticks; first pure-decode captures) ===", flush=True
        )
        wids = submit_batch(graph, cfg, b)
        for _ in range(WARM):
            graph.step()
        torch.cuda.synchronize()
        g = graph._decode_graphs[b]
        tracer.wrap_method(g, "run", "graph_replay")
        wall = measure(graph, tracer, TICKS)
        drain(graph, wids)
        replay, smp, host = print_graph(b, tracer, wall, TICKS)
        lh = report["batches"][b]["eager"]["lm_head_ms"]
        full = extrapolate(replay, lh, wall - replay, args.layers)
        print(
            f"  extrapolated 27B: {full:.3f} ms/tick "
            f"({1000 / full:.1f} tok/s per-request, {1000 * b / full:.1f} aggregate)"
        )
        report["batches"][b]["graph"] = {
            "wall_ms": wall,
            "replay_ms": replay,
            "sampling_ms": smp,
            "host_ms": host,
            "per_request_tok_s": 1000 / wall,
            "aggregate_tok_s": 1000 * b / wall,
            "extrapolated_27b_ms": full,
            "extrapolated_27b_aggregate_tok_s": 1000 * b / full,
        }
        tracer.totals.clear()
        tracer.counts.clear()

    report["linear_paths"] = tracer.linear_paths
    print("\nJSON " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
