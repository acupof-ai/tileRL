"""Bench the fp4 decode path on the H20 pod: GEMV vs WGMMA-padded per linear,
roofline efficiency, and slice decode before/after (same process, same GPU).

The "before" path is the exact pre-GEMV decode path: the GEMV key is popped
from the sm90 registry cell, so backend.linear_fp4(M=1) pads to 16 WGMMA
rows. The "after" path dispatches to the GEMV. Both go through the same
backend entry point, so the numbers include identical Python overhead.

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src TILERL_TARGET=cuda \\
        python3 scripts/bench_fp4_gemv.py /host/tc27-nvfp4-slice2 --layers 2
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import torch

sys.path.insert(0, "scripts")
import profile_slice as ps  # noqa: E402

from tilerl.config import qwen36_27b  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.model import fp4_param_keys, load_hf  # noqa: E402
from tilerl.ops import kernels_mma  # noqa: E402
from tilerl.ops.registry import _REGISTRY
from tilerl.ops.backend import get_backend  # noqa: E402
from tilerl.ops.reference import linear_fp4  # noqa: E402

_SM90_KEY = ("bf16", "sm90")


def _set_gemv(enabled: bool) -> None:
    cell = _REGISTRY[_SM90_KEY]
    if enabled:
        cell["linear_fp4_gemv"] = kernels_mma.make_linear_fp4_gemv
    else:
        cell.pop("linear_fp4_gemv", None)


def _measure_bw_gbs() -> float:
    """Achievable HBM BW from a device-to-device copy (read+write = 2N bytes)."""
    n = 256 * 2**20 // 4
    src = torch.empty(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(10):
        dst.copy_(src)
    iters = 50
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        dst.copy_(src)
    e.record()
    torch.cuda.synchronize()
    return 2 * n * 4 / (s.elapsed_time(e) / iters / 1e3) / 1e9


def _time_calls(fn, iters: int) -> float:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def bench_shapes(backend, model, cfg, bw_gbs: float) -> None:
    print(f"\n=== per-linear GEMV vs WGMMA-padded (BW {bw_gbs:.1f} GB/s) ===")
    print(
        f"  {'shape (N,K)':<22} {'bytes MB':>9} {'roof ms':>8} {'GEMV ms':>9} "
        f"{'WGMMA ms':>9} {'GEMV %roof':>10} {'speedup':>8}"
    )
    keys = sorted(k for k in fp4_param_keys(cfg) if k + ".wq" in model.params)
    for key in keys:
        wq = model.params[key + ".wq"]
        scale = model.params[key + ".scale"]
        N, K = wq.shape[0], wq.shape[1] * 2  # wq is [N, K//2]
        x = torch.randn(1, K, device=backend.device)
        ref = None

        def run_gemv():
            nonlocal ref
            y = backend.linear_fp4(x, wq, scale)
            if ref is None:
                ref = linear_fp4(x.cpu(), wq.cpu(), scale.cpu())
                assert (y.cpu() - ref).abs().max() < 1e-2 * ref.abs().max() + 1e-3, (
                    f"{key}: GEMV parity failed"
                )
            return y

        _set_gemv(True)
        for _ in range(5):
            run_gemv()
        gemv_ms = _time_calls(run_gemv, 50)

        _set_gemv(False)
        for _ in range(5):
            backend.linear_fp4(x, wq, scale)
        mma_ms = _time_calls(lambda: backend.linear_fp4(x, wq, scale), 50)
        _set_gemv(True)

        bytes_ = N * K * 0.75 + 2 * K  # bf16 X
        roof_ms = bytes_ / (bw_gbs * 1e9) * 1e3
        print(
            f"  {f'{N},{K}':<22} {bytes_ / 2**20:>9.2f} {roof_ms:>8.4f} {gemv_ms:>9.4f} "
            f"{mma_ms:>9.4f} {100 * roof_ms / gemv_ms:>9.1f}% {mma_ms / gemv_ms:>7.1f}x"
        )


def _decode_ticks(backend, model, cfg, ticks: int):
    engine = build_engine(cfg, model, backend, num_blocks=128, num_slots=4)
    tracer = ps.Tracer(backend)
    gen = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, cfg.vocab_size, (512,), generator=gen).tolist()
    wid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    ps._drive(engine, wid, 1024)
    return ps.time_decode(engine, tracer, cfg.vocab_size, ticks)


def bench_slice(backend, model, cfg, ticks: int) -> None:
    print("\n=== slice decode (2 GDN layers), before (WGMMA) vs after (GEMV) ===")
    _set_gemv(False)
    t0 = time.perf_counter()
    bt, bc, bwalls = _decode_ticks(backend, model, cfg, ticks)
    print(f"  before warmup+time: {time.perf_counter() - t0:.0f}s")
    _set_gemv(True)
    t0 = time.perf_counter()
    at, ac, awalls = _decode_ticks(backend, model, cfg, ticks)
    print(f"  after  warmup+time: {time.perf_counter() - t0:.0f}s")

    for tag, totals, counts, walls in (("BEFORE", bt, bc, bwalls), ("AFTER", at, ac, awalls)):
        gpu = sum(totals.values()) / ticks
        wall = sum(walls) / ticks * 1e3
        lf = totals.get("linear_fp4", 0.0) / ticks
        print(
            f"  {tag}: GPU {gpu:.3f} ms/tick, wall {wall:.3f} ms/tick "
            f"({1000 / wall:.1f} tok/s), linear_fp4 {lf:.3f} ms "
            f"({100 * lf / gpu:.0f}% of GPU)"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ticks", type=int, default=10)
    args = p.parse_args()

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = replace(
        qwen36_27b(),
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in qwen36_27b().full_attn_layers if i < args.layers),
    )
    t0 = time.perf_counter()
    model = load_hf(cfg, args.source)
    # Migrate params once (engine build does the same; bench_shapes runs
    # before any engine exists, so without this it would time per-call H2D
    # copies instead of the kernel).
    model.params = {k: v.to(backend.device) for k, v in model.params.items()}
    print(f"load: {time.perf_counter() - t0:.0f}s", flush=True)

    bw = _measure_bw_gbs()
    bench_shapes(backend, model, cfg, bw)
    bench_slice(backend, model, cfg, args.ticks)


if __name__ == "__main__":
    main()
