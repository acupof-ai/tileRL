"""A/B: batched-decode arms (shipped / ks1 / smallm) at B=1..8 on the slice4 decode graph.

Win gate per arm: B=8 aggregate tok/s gain >= 3%, max fro-relerr vs shipped <= 1e-2, B=1 neutral.
Usage: CUDA_VISIBLE_DEVICES=6 TILERL_TARGET=cuda PYTHONPATH=src python3 scripts/ab_batch_decode.py /host/tc27-nvfp4-slice4 --layers 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import torch
from tilerl_kernels import backend as backend_mod
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend

from tilerl.config import qwen36_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import load_hf

WARM = 12  # ticks: flushes B=8's 8 one-per-tick prefill admissions + 4 decodes
TICKS = 30

ARMS = {
    "shipped": {"_DECODE_KS1": False, "_SMALLM_GEMV": False},
    "ks1": {"_DECODE_KS1": True, "_SMALLM_GEMV": False},
    "smallm": {"_DECODE_KS1": False, "_SMALLM_GEMV": True},
}


def _quantize_fp8(w_master):
    """Per-128-block quant in the loader's layout; inlined so tests/ need not be on PYTHONPATH."""
    n, k = w_master.shape
    ns, ks = (n + 127) // 128, (k + 127) // 128
    padded = w_master.float().new_zeros(ns * 128, ks * 128)
    padded[:n, :k] = w_master.float()
    blocks = padded.reshape(ns, 128, ks, 128)
    block_max = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = (block_max / 448.0).reshape(ns, ks).contiguous()
    w8 = (blocks / (block_max / 448.0)).reshape(ns * 128, ks * 128)[:n, :k]
    return w8.to(torch.float8_e4m3fn).contiguous(), scale


def prompts_for(cfg, b):
    gen = torch.Generator().manual_seed(7)
    return [torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist() for _ in range(b)]


def run_arm(model, backend, cfg, batches, arm):
    for flag, val in ARMS[arm].items():
        setattr(backend_mod, flag, val)
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=8, decode_graph=True)
    out = {}
    for b in batches:
        prompts = prompts_for(cfg, b)
        wids = [
            engine.submit(
                p, SamplingParams(temperature=0.0, max_new_tokens=WARM + TICKS + 4, seed=i)
            )
            for i, p in enumerate(prompts)
        ]
        for _ in range(WARM):
            engine.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(TICKS):
            engine.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / TICKS * 1e3
        done: dict[int, list[int]] = {}
        for _ in range(WARM + TICKS + 8):
            done.update(engine.poll())
            if all(w in done for w in wids):
                break
            engine.step()
        out[b] = {
            "ms_per_tick": ms,
            "per_request_tok_s": 1000 / ms,
            "aggregate_tok_s": 1000 * b / ms,
            "tokens": [done[w] for w in wids],
            "graph_on": engine._decode_graph_on,
            "graph_captured": b in engine._decode_graphs,
        }
        print(
            f"  {arm:<8} B={b}: {ms:.4f} ms/tick "
            f"({1000 * b / ms:.1f} agg tok/s) graph={engine._decode_graph_on}"
        )
    return out


def kernel_relerr(backend, arm):
    # Frobenius, not allclose: past K~1024 elementwise tolerances flag summation-order noise.
    from tilerl_kernels.reference import pack_fp4

    torch.manual_seed(41)
    report = {}
    for kind, N, K in [("fp4", 4096, 5120), ("fp4", 5120, 17408), ("fp4", 2048, 288)]:
        w_master = torch.randn(N, K) * 0.1
        wq, scale = pack_fp4(w_master)
        x = torch.randn(8, K) * 0.5
        for flag, val in ARMS["shipped"].items():
            setattr(backend_mod, flag, val)
        y_ship = backend.linear_fp4(x, wq, scale)
        for flag, val in ARMS[arm].items():
            setattr(backend_mod, flag, val)
        y_cand = backend.linear_fp4(x, wq, scale)
        y_ref = reference.linear_fp4(x, wq, scale)
        y_cand = y_cand.float().cpu()
        y_ship = y_ship.float().cpu()
        d = (y_cand - y_ship).abs()
        report[f"fp4_N{N}_K{K}"] = {
            "fro_relerr_vs_shipped": (d.norm() / y_ship.norm()).item(),
            "fro_relerr_vs_f32_ref": ((y_cand - y_ref).norm() / y_ref.norm()).item(),
            "shipped_fro_vs_f32_ref": ((y_ship - y_ref).norm() / y_ref.norm()).item(),
        }
    for N, K in [(4096, 5120), (2048, 1024)]:
        w_master = torch.randn(N, K) * 0.1
        w8, wscale = _quantize_fp8(w_master)
        x = torch.randn(8, K) * 0.5
        for flag, val in ARMS["shipped"].items():
            setattr(backend_mod, flag, val)
        y_ship = backend.linear_fp8(x, w8, wscale)
        for flag, val in ARMS[arm].items():
            setattr(backend_mod, flag, val)
        y_cand = backend.linear_fp8(x, w8, wscale)
        y_ref = reference.linear_fp8(x, w8, wscale)
        y_cand = y_cand.float().cpu()
        y_ship = y_ship.float().cpu()
        d = (y_cand - y_ship).abs()
        report[f"fp8_N{N}_K{K}"] = {
            "fro_relerr_vs_shipped": (d.norm() / y_ship.norm()).item(),
            "fro_relerr_vs_f32_ref": ((y_cand - y_ref).norm() / y_ref.norm()).item(),
            "shipped_fro_vs_f32_ref": ((y_ship - y_ref).norm() / y_ref.norm()).item(),
        }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batches", type=str, default="1,2,4,8")
    p.add_argument("--arms", type=str, default="shipped,ks1,smallm")
    args = p.parse_args()

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = replace(
        qwen36_27b(),
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in qwen36_27b().full_attn_layers if i < args.layers),
    )
    t0 = time.perf_counter()
    model = load_hf(cfg, args.source, fuse_projections=True)
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

    batches = [int(x) for x in args.batches.split(",")]
    arms = args.arms.split(",")
    results = {}
    for arm in arms:
        print(f"\n=== arm: {arm} ===", flush=True)
        results[arm] = run_arm(model, backend, cfg, batches, arm)

    print("\n=== kernel relerr (B=8, identical inputs) ===", flush=True)
    relerr = {}
    for arm in arms:
        if arm == "shipped":
            continue
        relerr[arm] = kernel_relerr(backend, arm)
        for k, v in relerr[arm].items():
            print(
                f"  {arm:<8} {k}: fro-vs-shipped {v['fro_relerr_vs_shipped']:.4f} "
                f"fro-vs-f32 {v['fro_relerr_vs_f32_ref']:.4f} "
                f"(shipped-vs-f32 {v['shipped_fro_vs_f32_ref']:.4f})"
            )

    print("\n=== token equality (greedy, 30+ ticks) ===", flush=True)
    tok = {}
    for arm in arms:
        if arm == "shipped":
            continue
        tok[arm] = {}
        for b in batches:
            same = results["shipped"][b]["tokens"] == results[arm][b]["tokens"]
            tok[arm][b] = bool(same)
            print(f"  {arm:<8} B={b}: {'identical' if same else 'DIFFER'}")

    print("\n=== summary ===", flush=True)
    summary = {}
    for arm in arms:
        if arm == "shipped":
            continue
        max_relerr = max(v["fro_relerr_vs_shipped"] for v in relerr[arm].values())
        b8_gain = 100 * (
            results[arm][8]["aggregate_tok_s"] / results["shipped"][8]["aggregate_tok_s"] - 1
        )
        b1_neutral = abs(results[arm][1]["ms_per_tick"] / results["shipped"][1]["ms_per_tick"] - 1)
        win = b8_gain >= 3.0 and max_relerr <= 1e-2 and b1_neutral <= 0.02
        summary[arm] = {
            "b8_gain_pct": b8_gain,
            "max_fro_relerr_vs_shipped": max_relerr,
            "b1_neutral_pct": 100 * b1_neutral,
            "win": bool(win),
        }
        print(
            f"  {arm:<8}: B=8 gain {b8_gain:+.1f}%  max-fro-relerr {max_relerr:.4f}  "
            f"B=1 drift {100 * b1_neutral:+.2f}%  -> {'WIN' if win else 'no-win'}"
        )

    report = {
        "commit": os.environ.get("BENCH_COMMIT", "?"),
        "batches": {
            str(b): {
                arm: {
                    "ms_per_tick": results[arm][b]["ms_per_tick"],
                    "aggregate_tok_s": results[arm][b]["aggregate_tok_s"],
                }
                for arm in arms
            }
            for b in batches
        },
        "kernel_relerr": relerr,
        "token_equality": tok,
        "summary": summary,
    }
    print("\nJSON " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
