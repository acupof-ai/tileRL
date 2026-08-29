"""A/B prefill bench: fused GDN chunk kernel vs torch-eager reference.

Single process, same GPU, JIT-free: one warmup pass compiles every shape the
timed ticks touch (including the chunk kernel at the prefill T), then two
prefill ticks are timed with CUDA events — arm A through the normal dispatch
(fused chunk kernel on sm90), arm B with linear_attn_chunk pinned to the
torch-eager reference. Same engine, same pools, same tick code path; only the
GDN implementation differs.

Usage:
    TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=1 \\
        PYTHONPATH=src python3 scripts/bench_gdn_prefill.py /host/tc27-nvfp4-slice2 --layers 2
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from tilerl.engine import SamplingParams
from tilerl_kernels import reference


def _drive(engine, wid, max_steps) -> None:
    for _ in range(max_steps):
        engine.step()
        if wid in engine.poll():
            return
    raise RuntimeError("request did not finish")


def _time_prefill_tick(engine, vocab, length) -> float:
    """Submit a length-token prompt and time its single prefill tick (ms)."""
    gen = torch.Generator().manual_seed(3)
    prompt = torch.randint(0, vocab, (length,), generator=gen).tolist()
    engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=1, seed=0))
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    engine.step()  # prefill tick (no decodes pending on a fresh engine)
    end.record()
    torch.cuda.synchronize()
    for _ in range(8):  # drain the 1-token decode finish
        if engine.poll():
            break
        engine.step()
    return start.elapsed_time(end)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="HF checkpoint directory")
    p.add_argument("--model", choices=["qwen36-27b"], default="qwen36-27b")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--prefill-len", type=int, default=512)
    args = p.parse_args()

    from tilerl.config import qwen36_27b
    from tilerl.engine import build_engine
    from tilerl.model import load_hf
    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    if backend.device.type != "cuda":
        raise SystemExit("bench needs the CUDA target (TILERL_TARGET=cuda)")
    cfg = qwen36_27b()
    cfg = replace(
        cfg,
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in cfg.full_attn_layers if i < args.layers),
    )

    t0 = time.perf_counter()
    model = load_hf(cfg, args.source)
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)
    engine = build_engine(cfg, model, backend, num_blocks=128, num_slots=4)

    # Warmup: compile EVERY (shape, dtype) the timed ticks touch. Two passes
    # (the second confirms JIT-free), same as profile_slice.py.
    print("warmup pass 1: prefill + decode (NVCC JIT, slow)...", flush=True)
    gen = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, cfg.vocab_size, (args.prefill_len,), generator=gen).tolist()
    wid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    _drive(engine, wid, 1024)
    print("warmup pass 2: same shapes (JIT-free)...", flush=True)
    wid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    _drive(engine, wid, 1024)
    print("warmup: done", flush=True)

    ms_a = _time_prefill_tick(engine, cfg.vocab_size, args.prefill_len)
    print(f"arm A (fused chunk kernel): {ms_a:.1f} ms tick, {ms_a / args.prefill_len:.4f} ms/tok")

    # Arm B: pin the GDN op to the torch-eager reference (same signature),
    # then time an identical prefill tick.
    ref = reference.gdn_forward
    backend.linear_attn_chunk = lambda q, k, v, g, beta, state, **kw: ref(
        q, k, v, g, beta, state, **kw
    )
    ms_b = _time_prefill_tick(engine, cfg.vocab_size, args.prefill_len)
    print(
        f"arm B (torch-eager reference): {ms_b:.1f} ms tick, {ms_b / args.prefill_len:.4f} ms/tok"
    )
    print(f"speedup: {ms_b / ms_a:.1f}x")


if __name__ == "__main__":
    main()
