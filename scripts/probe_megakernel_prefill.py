"""Check MegaForward.prefill against the serial megakernel and the batched path."""

import argparse
import json
from dataclasses import replace

import torch
from tilerl_kernels.backend import Backend
from tilerl_kernels.mega import MegaForward

from tilerl.config import qwen38_27b
from tilerl.kv_cache import BatchKv, LinearStatePool, PagedKvPool
from tilerl.model import build_random, load_hf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--stream-load", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(23)
    backend, cfg = Backend("cuda"), qwen38_27b()
    if args.checkpoint:
        if args.stream_load:
            import os

            from tilerl_kernels.mega_loader import load_sm70
            os.environ["TILERL_SM70_NATIVE_FP8"] = "1"
            model = load_sm70(cfg, args.checkpoint, backend)
        else:
            model = load_hf(cfg, args.checkpoint, fuse_projections=True)
    else:
        cfg = replace(cfg, hidden_size=512, intermediate_size=512, num_layers=2,
                      num_attention_heads=4, num_kv_heads=1, head_dim=128, vocab_size=512,
                      full_attn_layers=(1,), linear_num_key_heads=1, linear_num_value_heads=4)
        model = build_random(cfg, seed=5, fuse_projections=True)
        for name, tensor in model.params.items():
            if name.endswith(".oscale") or tensor.ndim == 2 and not name.endswith((".wq", ".scale")):
                tensor.mul_(0.02)
    model.params = backend.materialize(model.params)
    model.materialized = True
    batch, tokens = args.batch, args.tokens
    capacity = max(tokens, 16)
    pages = (capacity + 15) // 16
    pool = PagedKvPool(batch * pages, cfg.num_kv_heads, cfg.head_dim, device=backend.device,
                       dtype=torch.float32, layer_map=cfg.full_attn_layers)
    state = LinearStatePool(batch, cfg.num_linear_layers, cfg.linear_num_value_heads,
                            cfg.linear_value_head_dim, device=backend.device, dtype=torch.float32,
                            conv_window=3, conv_dim=cfg.linear_qkv_dim)
    table = torch.arange(batch * pages, device=backend.device).reshape(batch, pages)
    kv = BatchKv(table, torch.full((batch,), tokens, device=backend.device, dtype=torch.long),
                 torch.arange(batch, device=backend.device), pool, state)
    ids = torch.randint(0, min(cfg.vocab_size, 512), (batch, tokens), device=backend.device)
    baseline = model.forward(ids, torch.arange(tokens, device=backend.device), kv, backend,
                             last_only=True)[:, -1]
    mega = MegaForward(model, backend, batch, capacity)
    serial = None
    for pos in range(tokens):
        serial = mega.step(ids[:, pos].contiguous()).clone()
    mega.reset()
    fused = mega.prefill(ids).clone()
    serial_diff = (fused - serial).abs().max().item()
    baseline_diff = (fused - baseline).abs().max().item()
    print(json.dumps({"serial_max_abs": serial_diff, "baseline_max_abs": baseline_diff,
                      "top1_equal": bool(torch.equal(fused.argmax(-1), baseline.argmax(-1)))}),
          flush=True)
    torch.testing.assert_close(fused, serial, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fused, baseline, rtol=1e-2, atol=1e-2)
    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        mega.reset()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            mega.prefill(ids)
            torch.cuda.synchronize()
        events = [e for e in profiler.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        model_events = [e for e in events if "megakernel" in e.name]
        print(json.dumps({"gpu_events": len(events), "model_events": len(model_events),
                          "setup_events": [e.name.split("(")[0] for e in events if e not in model_events]}),
              flush=True)
        assert len(model_events) == 1
        profiler.export_chrome_trace("megakernel-prefill-trace.json")
    print(json.dumps({"gate": "prefill_parity", "layers": cfg.num_layers, "batch": batch,
                      "tokens": tokens, "max_abs": serial_diff}), flush=True)


if __name__ == "__main__":
    main()
