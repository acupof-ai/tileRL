"""Compare every model layer against the existing sm70 path."""

import argparse
import json
from dataclasses import replace

import torch
from tilerl_kernels.backend import Backend
from tilerl_kernels.mega import MegaForward

from tilerl.config import qwen38_27b
from tilerl.engine import card_guard
from tilerl.kv_cache import BatchKv, LinearStatePool, PagedKvPool
from tilerl.model import build_random, load_hf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--negative", action="store_true")
    parser.add_argument("--stream-load", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(19)
    backend = Backend("c" if args.cpu else "cuda")
    cfg = qwen38_27b()
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
        model = build_random(cfg, seed=3, fuse_projections=True)
        for name, tensor in model.params.items():
            if name.endswith(".oscale") or tensor.ndim == 2 and not name.endswith((".wq", ".scale")):
                tensor.mul_(0.02)
    if args.mixed:
        import os

        from tilerl_kernels.reference import unpack_fp4
        os.environ["TILERL_SM70_NATIVE_FP8"] = "1"
        base = "layers.0.qkvz"
        value = unpack_fp4(model.params.pop(base+".wq"), model.params.pop(base+".scale"),
                           model.params.pop(base+".oscale"))
        osc = value.abs().amax(-1).clamp_min(1e-6) / 448
        model.params[base+".w8"] = (value/osc[:, None]).to(torch.float8_e4m3fn)
        model.params[base+".oscale"] = osc
        model.params[base+".wscale"] = torch.ones((value.shape[0]+127)//128, value.shape[1]//128)
    card_guard()
    model.params = backend.materialize(model.params)
    model.materialized = True
    if args.load_only:
        print(json.dumps({"gate":"load", "resident_bytes":sum(v.numel()*v.element_size() for v in model.params.values()),
                          "allocated_bytes":torch.cuda.memory_allocated()}), flush=True)
        return
    batch, capacity = args.batch, max(args.tokens, 16)
    pages = (capacity + 15) // 16
    pool = PagedKvPool(batch * pages, cfg.num_kv_heads, cfg.head_dim,
                       device=backend.device, dtype=torch.float32, layer_map=cfg.full_attn_layers)
    state = LinearStatePool(batch, cfg.num_linear_layers, cfg.linear_num_value_heads,
                            cfg.linear_value_head_dim, device=backend.device, dtype=torch.float32,
                            conv_window=3, conv_dim=cfg.linear_qkv_dim)
    table = torch.arange(batch * pages, device=backend.device).reshape(batch, pages)
    slots = torch.arange(batch, device=backend.device)
    length = torch.ones(batch, device=backend.device, dtype=torch.long)
    kv = BatchKv(table, length, slots, pool, state)
    mega = MegaForward(model, backend, batch, capacity, cpu_reference=args.cpu)
    ids = torch.randint(0, min(cfg.vocab_size, 512), (batch, args.tokens), device=backend.device)
    worst = 0.0
    if args.debug:
        from tilerl_kernels.kernels_mega import make_megakernel
        from tilerl_kernels.mega_reference import interpret_instruction
        data = [t.clone() for t in mega.buffers]
        mega.ids.copy_(ids[:, 0])
        data[mega.rows[0][4]].copy_(ids[:, 0])
        kernel = make_megakernel(80, 1, len(mega.buffers))
        for index, row in enumerate(mega.rows):
            program = torch.tensor([row], device=backend.device, dtype=torch.int64)
            data = [t.clone() for t in mega.buffers]
            if row[0] == 2 and row[11] > 0:
                source = data[row[4]]
                data[row[5]].copy_(backend.linear_fp4(source, mega.buffers[row[6]],
                                                     mega.buffers[row[7]], oscale=mega.buffers[row[8]]))
            else:
                interpret_instruction(row, data, 0)
            kernel(program, mega.pointers, mega.position)
            target = row[5]
            diff = (mega.buffers[target] - data[target]).abs().max().item()
            print(json.dumps({"instruction": index, "opcode": row[0], "max_abs": diff}), flush=True)
            torch.testing.assert_close(mega.buffers[target], data[target], rtol=1e-2, atol=1e-2)
        return
    baselines = []
    for pos in range(args.tokens):
        length.fill_(pos + 1)
        baseline = model.forward(ids[:, pos:pos+1], torch.tensor([pos], device=backend.device), kv, backend)
        candidate = mega.step(ids[:, pos].contiguous())
        baselines.append(baseline[:, 0].clone())
        diff = (candidate - baseline[:, 0]).abs().max().item()
        worst = max(worst, diff)
        print(json.dumps({"pos": pos, "max_abs": diff,
                          "top1_equal": bool(torch.equal(candidate.argmax(-1), baseline[:, 0].argmax(-1)))}), flush=True)
        torch.testing.assert_close(candidate, baseline[:, 0], rtol=1e-2, atol=1e-2)
    if args.negative:
        # State flow is the thing under test: zeroing it after token 0 must break the gate.
        mega.reset()
        diverged = 0.0
        for pos in range(args.tokens):
            candidate = mega.step(ids[:, pos].contiguous())
            if pos == 0:
                for tensor in mega.state_buffers:
                    tensor.zero_()
            if pos:
                diverged = max(diverged, (candidate - baselines[pos]).abs().max().item())
        print(json.dumps({"negative_control": "zeroed state", "max_abs": diverged}), flush=True)
        assert diverged > 1e-2, "negative control passed: state is not reaching the logits"
    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            mega.kernel(mega.program, mega.pointers, mega.position)
            torch.cuda.synchronize()
        events = [e for e in profiler.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        print(json.dumps({"gpu_events": len(events), "gpu_names": [e.name for e in events]}), flush=True)
        assert len(events) == 1 and "megakernel" in events[0].name
        profiler.export_chrome_trace("megakernel-launch-trace.json")
    print(json.dumps({"gate": "model_parity", "layers": cfg.num_layers,
                      "batch": batch, "tokens": args.tokens, "max_abs": worst}), flush=True)


if __name__ == "__main__":
    main()
