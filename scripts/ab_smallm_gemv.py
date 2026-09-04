"""A/B: shipped dispatch vs small-M GEMV (2<=M<=8) at B=1..8 on the slice4 decode graph.

Reports ms/tick, kernel relerr vs shipped, parity vs the f32 reference, greedy token equality.
Usage: CUDA_VISIBLE_DEVICES=7 TILERL_TARGET=cuda PYTHONPATH=src python3 scripts/ab_smallm_gemv.py /host/tc27-nvfp4-slice4 --layers 4
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


def run_arm(model, backend, cfg, batches, smallm):
    backend_mod._SMALLM_GEMV = smallm
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
            f"  {'cand' if smallm else 'ctrl'} B={b}: {ms:.4f} ms/tick "
            f"({1000 * b / ms:.1f} agg tok/s) graph={engine._decode_graph_on}"
        )
    return out


def kernel_relerr(backend):
    from tilerl_kernels.reference import pack_fp4

    torch.manual_seed(41)
    report = {}
    for kind, N, K in [("fp4", 4096, 5120), ("fp4", 5120, 17408), ("fp4", 2048, 288)]:
        w_master = torch.randn(N, K) * 0.1
        wq, scale = pack_fp4(w_master)
        x = torch.randn(8, K) * 0.5
        backend_mod._SMALLM_GEMV = False
        y_ship = backend.linear_fp4(x, wq, scale)
        backend_mod._SMALLM_GEMV = True
        y_cand = backend.linear_fp4(x, wq, scale)
        y_ref = reference.linear_fp4(x, wq, scale)
        d = (y_cand - y_ship).abs()
        parity = torch.allclose(y_cand.float().cpu(), y_ref.float().cpu(), rtol=1e-2, atol=1e-2)
        report[f"fp4_N{N}_K{K}"] = {
            "max_relerr_vs_shipped": (d.max() / y_ship.abs().max()).item(),
            "fro_relerr_vs_shipped": (d.norm() / y_ship.norm()).item(),
            "parity_vs_f32_ref": bool(parity),
        }
    for N, K in [(4096, 5120), (2048, 1024)]:
        w_master = torch.randn(N, K) * 0.1
        w8, wscale = _quantize_fp8(w_master)
        x = torch.randn(8, K) * 0.5
        backend_mod._SMALLM_GEMV = False
        y_ship = backend.linear_fp8(x, w8, wscale)
        backend_mod._SMALLM_GEMV = True
        y_cand = backend.linear_fp8(x, w8, wscale)
        y_ref = reference.linear_fp8(x, w8, wscale)
        d = (y_cand - y_ship).abs()
        parity = torch.allclose(y_cand.float().cpu(), y_ref.float().cpu(), rtol=1e-2, atol=1e-2)
        report[f"fp8_N{N}_K{K}"] = {
            "max_relerr_vs_shipped": (d.max() / y_ship.abs().max()).item(),
            "fro_relerr_vs_shipped": (d.norm() / y_ship.norm()).item(),
            "parity_vs_f32_ref": bool(parity),
        }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batches", type=str, default="1,2,4,8")
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
    print("\n=== control (shipped: M==1 GEMV gate) ===", flush=True)
    ctrl = run_arm(model, backend, cfg, batches, smallm=False)
    print("\n=== candidate (small-M GEMV, M<=8) ===", flush=True)
    cand = run_arm(model, backend, cfg, batches, smallm=True)

    print("\n=== kernel relerr (B=8, identical inputs) ===", flush=True)
    relerr = kernel_relerr(backend)
    for k, v in relerr.items():
        print(
            f"  {k}: max-relerr {v['max_relerr_vs_shipped']:.4f} "
            f"fro-relerr {v['fro_relerr_vs_shipped']:.4f} "
            f"parity-vs-f32 {v['parity_vs_f32_ref']}"
        )

    print("\n=== token equality (greedy, 30+ ticks) ===", flush=True)
    tok = {}
    for b in batches:
        same = ctrl[b]["tokens"] == cand[b]["tokens"]
        tok[b] = bool(same)
        print(f"  B={b}: {'identical' if same else 'DIFFER'}")

    report = {
        "commit": os.environ.get("BENCH_COMMIT", "?"),
        "batches": {
            str(b): {
                "control_ms": ctrl[b]["ms_per_tick"],
                "candidate_ms": cand[b]["ms_per_tick"],
                "control_agg_tok_s": ctrl[b]["aggregate_tok_s"],
                "candidate_agg_tok_s": cand[b]["aggregate_tok_s"],
                "gain_pct": 100 * (cand[b]["ms_per_tick"] / ctrl[b]["ms_per_tick"] - 1) * -1,
                "tokens_identical": tok[b],
                "graph_captured": cand[b]["graph_captured"],
            }
            for b in batches
        },
        "kernel_relerr": relerr,
    }
    print("\nJSON " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
