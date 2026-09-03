"""Per-op CUDA-event time of one decode/prefill tick on the NVFP4 slice, the wall-vs-GPU
dispatch gap, and a naive full-model extrapolation.

    TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=1 \\
        PYTHONPATH=src python3 scripts/profile_slice.py /host/tc27-nvfp4-slice2 --layers 2
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from tilerl.engine import SamplingParams

_OPS = (
    "add",
    "rmsnorm",
    "rope",
    "linear",
    "linear_fp4",
    "linear_fp8",
    "paged_attention",
    "attention",
    "linear_attn_chunk",
    "silu_mul",
    "softmax",
    "embedding",
    "sample",
)

#: Once-per-tick ops (everything else scales with layer count).
_FIXED = ("embedding", "sample")


class Tracer:
    def __init__(self, backend) -> None:
        self.on = False
        self.records: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        for name in _OPS:
            orig = getattr(backend, name)
            setattr(backend, name, self._wrap(name, orig))

    def _wrap(self, name, orig):
        records = self.records

        def wrapper(*args, **kwargs):
            if not self.on:
                return orig(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = orig(*args, **kwargs)
            end.record()
            records.append((name, start, end))
            return out

        return wrapper


def _drive(engine, wid, max_steps) -> None:
    for _ in range(max_steps):
        engine.step()
        if wid in engine.poll():
            return
    raise RuntimeError("request did not finish")


def _aggregate(records):
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, s, e in records:
        totals[name] = totals.get(name, 0.0) + s.elapsed_time(e)
        counts[name] = counts.get(name, 0) + 1
    return totals, counts


def time_decode(engine, tracer, vocab, ticks):
    gen = torch.Generator().manual_seed(2)
    prompt = torch.randint(0, vocab, (16,), generator=gen).tolist()
    wid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=ticks + 2, seed=0))
    engine.step()  # untimed prefill tick (compiles nothing new after warmup)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    walls = []
    for _ in range(ticks):
        torch.cuda.synchronize()
        tracer.records.clear()
        tracer.on = True
        t0 = time.perf_counter()
        engine.step()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        tracer.on = False
        t, c = _aggregate(tracer.records)
        for k, v in t.items():
            totals[k] = totals.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + c[k]
    _drive(engine, wid, 64)  # drain the last 2 tokens
    return totals, counts, walls


def time_prefill(engine, tracer, vocab, length):
    gen = torch.Generator().manual_seed(3)
    prompt = torch.randint(0, vocab, (length,), generator=gen).tolist()
    engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=1, seed=0))
    torch.cuda.synchronize()
    tracer.records.clear()
    tracer.on = True
    t0 = time.perf_counter()
    engine.step()  # prefill tick (no decodes pending)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    tracer.on = False
    totals, counts = _aggregate(tracer.records)
    for _ in range(8):  # drain the 1-token decode finish
        if engine.poll():
            break
        engine.step()
    return totals, counts, wall


def _table(title, rows, total_ms) -> None:
    print(f"\n{title}")
    print(f"  {'op':<20} {'ms':>10} {'%':>6}")
    for name, ms in rows:
        print(f"  {name:<20} {ms:>10.3f} {100 * ms / total_ms:>5.1f}%")


def report(cfg, dec, pre, args) -> None:
    dt, dc, dwalls = dec
    pt, pc, pwall = pre
    ticks = args.decode_ticks
    layers = cfg.num_layers

    gpu_sum = sum(dt.values()) / ticks
    wall = sum(dwalls) / ticks * 1000.0
    overhead = wall - gpu_sum
    calls_per_tick = sum(dc.values()) / ticks
    per_op_overhead = overhead / calls_per_tick if calls_per_tick else 0.0

    if args.decode_graph:
        # the tracer sees nothing inside a replay, so wall is the tick.
        print(f"\nDECODE (captured) per tick (avg of {ticks} ticks, {layers} layers)")
        print(f"  {'wall (replay+copies+sample)':<20} {wall:>10.3f}")
        print(f"  {'throughput':<20} {1000.0 / wall:>10.1f} tok/s")
        print("  note: per-op GPU sum is empty — the forward is one graph replay")
    else:
        rows = sorted(((k, v / ticks) for k, v in dt.items()), key=lambda r: -r[1])
        _table(f"DECODE per tick (avg of {ticks} ticks, {layers} layers)", rows, gpu_sum)
        print(f"  {'GPU sum':<20} {gpu_sum:>10.3f}")
        print(f"  {'wall':<20} {wall:>10.3f}")
        print(
            f"  {'dispatch overhead':<20} {overhead:>10.3f} "
            f"({per_op_overhead * 1000:.1f} us/op, {calls_per_tick:.0f} ops/tick)"
        )

    p_gpu = sum(pt.values())
    p_overhead = pwall * 1000.0 - p_gpu
    prows = sorted(pt.items(), key=lambda r: -r[1])
    _table(f"PREFILL {args.prefill_len} tokens (one tick)", prows, p_gpu)
    print(f"  {'GPU sum':<20} {p_gpu:>10.3f} ms  ({p_gpu / args.prefill_len:.4f} ms/tok)")
    print(
        f"  {'wall':<20} {pwall * 1000.0:>10.3f} ms  "
        f"({pwall * 1000.0 / args.prefill_len:.4f} ms/tok)"
    )
    print(f"  {'dispatch overhead':<20} {p_overhead:>10.3f} ms")

    full_layers = 64
    if not args.decode_graph:
        fixed = sum(dt.get(k, 0.0) for k in _FIXED) / ticks
        per_layer_gpu = (gpu_sum - fixed) / layers
        gpu_full = per_layer_gpu * full_layers + fixed
        fixed_calls = sum(dc.get(k, 0) for k in _FIXED) / ticks
        per_layer_calls = (calls_per_tick - fixed_calls) / layers
        calls_full = per_layer_calls * full_layers + fixed_calls
        overhead_full = per_op_overhead * calls_full
        total_full = gpu_full + overhead_full

        print("\nEXTRAPOLATION (slice -> 64-layer 27B)")
        print(f"  per-layer GPU sum:        {per_layer_gpu:.3f} ms/tok")
        print(f"  fixed (embed+sample):     {fixed:.3f} ms/tok")
        print(f"  full-model GPU sum:       {gpu_full:.3f} ms/tok")
        print(f"  full-model dispatch:      {overhead_full:.3f} ms/tok ({calls_full:.0f} ops/tick)")
        print(
            f"  full-model decode:        {total_full:.3f} ms/tok "
            f"({1000.0 / total_full:.1f} tok/s)  target 12.5 ms (80 tok/s)"
        )
        print(
            "  caveat: slice has 2 GDN layers, 0 full-attn; the 16 full-attn "
            "layers of the 27B are unmeasured (GDN per-layer used as the average)."
        )

    per_tok_prefill = (p_gpu + p_overhead) / args.prefill_len
    per_layer_prefill = (p_gpu - sum(pt.get(k, 0.0) for k in _FIXED)) / layers
    full_prefill_ms = (
        per_layer_prefill * full_layers + sum(pt.get(k, 0.0) for k in _FIXED)
    ) / args.prefill_len + p_overhead / args.prefill_len
    print(
        f"  slice prefill:            {per_tok_prefill:.4f} ms/tok "
        f"({1000.0 / per_tok_prefill:.0f} tok/s)"
    )
    print(
        f"  full-model prefill:       {full_prefill_ms:.4f} ms/tok "
        f"({1000.0 / full_prefill_ms:.0f} tok/s)  target 0.263 ms (3800 tok/s)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="HF checkpoint directory")
    p.add_argument("--model", choices=["qwen36-27b"], default="qwen36-27b")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--decode-ticks", type=int, default=10)
    p.add_argument("--prefill-len", type=int, default=512)
    p.add_argument(
        "--decode-graph",
        action="store_true",
        help="capture the decode tick as a CUDA graph (replay per token)",
    )
    p.add_argument(
        "--fuse",
        action="store_true",
        help="fuse same-input fp4 projections (qkv/ab/gate_up) into one GEMV each",
    )
    args = p.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.config import qwen36_27b
    from tilerl.engine import build_engine
    from tilerl.model import load_hf

    backend = get_backend()
    if backend.device.type != "cuda":
        raise SystemExit("profiling needs the CUDA target (TILERL_TARGET=cuda)")
    cfg = qwen36_27b()
    cfg = replace(
        cfg,
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in cfg.full_attn_layers if i < args.layers),
    )

    t0 = time.perf_counter()
    model = load_hf(cfg, args.source, fuse_projections=args.fuse)
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

    # 128 blocks: the prefix store pins each 512-token prompt's 32 blocks for process life.
    engine = build_engine(
        cfg, model, backend, num_blocks=128, num_slots=4, decode_graph=args.decode_graph
    )
    tracer = Tracer(backend)

    # one prefill+decode pair JITs every M=512 and M=1 kernel; pass 2 proves it.
    print("warmup pass 1: 512-token prefill + decode (NVCC JIT, slow)...", flush=True)
    gen = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, cfg.vocab_size, (args.prefill_len,), generator=gen).tolist()
    wid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    _drive(engine, wid, 1024)
    print("warmup pass 2: same shapes (JIT-free)...", flush=True)
    wid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    _drive(engine, wid, 1024)
    print("warmup: done", flush=True)

    dec = time_decode(engine, tracer, cfg.vocab_size, args.decode_ticks)
    pre = time_prefill(engine, tracer, cfg.vocab_size, args.prefill_len)
    report(cfg, dec, pre, args)


if __name__ == "__main__":
    main()
