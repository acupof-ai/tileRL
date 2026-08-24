"""Smoke-drive a real HF checkpoint through tileRL: load, generate, train_step.

Bring-up tool for real-weight runs (the 2-layer NVFP4 slice today, the full
64-layer model tomorrow). Prints timings, flagging which numbers include
tilelang's per-shape NVCC JIT (first call per shape, 30-120s on CUDA).

Usage:
    TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=5 \
        python3 scripts/real_ckpt_smoke.py /host/tc27-nvfp4-slice2 --layers 2
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="HF checkpoint directory")
    p.add_argument("--model", choices=["qwen36-27b"], default="qwen36-27b")
    p.add_argument("--layers", type=int, default=None, help="truncate to first N layers")
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--gen", type=int, default=8)
    p.add_argument(
        "--train-steps", type=int, default=2, help="timed train_steps (step 1 includes JIT)"
    )
    args = p.parse_args()

    from tilerl.autograd import AdamW, Tape
    from tilerl.config import qwen36_27b
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.model import load_hf
    from tilerl.ops.backend import get_backend
    from tilerl.train import train_step

    backend = get_backend()
    cfg = qwen36_27b()
    if args.layers is not None and args.layers < cfg.num_layers:
        # A slice export's config.json carries the truncated layer count, so
        # validate against the truncated cfg (the full checkpoint validates
        # against the full cfg with --layers unset).
        cfg = replace(
            cfg,
            num_layers=args.layers,
            full_attn_layers=tuple(i for i in cfg.full_attn_layers if i < args.layers),
        )

    t0 = time.perf_counter()
    model = load_hf(cfg, args.source)
    print(f"load: {time.perf_counter() - t0:.1f}s  params={len(model.params)}")

    engine = build_engine(cfg, model, backend, num_blocks=32, num_slots=4, max_total_tokens=512)
    gen = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (args.prompt_len,), generator=gen).tolist()

    # Warmup with the SAME shapes as the timed run: tilelang JITs per shape,
    # so a different prompt length would leak NVCC time into the measurement.
    wid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=1, seed=0))
    for _ in range(256):
        engine.step()
        if wid in engine.poll():
            break
    else:
        raise RuntimeError("warmup did not finish")

    rid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=args.gen, seed=0))
    t0 = time.perf_counter()
    finished: dict[int, list[int]] = {}
    for _ in range(256 + args.gen * 4):
        engine.step()
        finished = engine.poll()  # poll clears — capture before the next tick
        if rid in finished:
            break
    else:
        raise RuntimeError("generation did not finish")
    dt = time.perf_counter() - t0
    out = finished[rid]
    print(
        f"generate: {len(out)} tokens in {dt * 1000:.1f} ms "
        f"({dt * 1000 / max(len(out), 1):.2f} ms/tok, JIT-free)"
    )
    print(f"tokens: {out}")

    opt = AdamW(lr=1e-3)
    seq = np.asarray(prompt + out, dtype=np.int64)[None, :]
    for step in range(args.train_steps):
        t0 = time.perf_counter()
        loss = train_step(model, seq, backend, opt, Tape())
        if backend.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tag = "includes JIT for training shapes" if step == 0 else "JIT-free"
        print(f"train_step {step + 1}: loss={loss:.4f}  {dt * 1000:.1f} ms ({tag})")


if __name__ == "__main__":
    main()
