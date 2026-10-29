"""Aligned decode benchmark: same tokens, weights, precision, and context."""

import argparse
import json
import os
import time

import torch
from tilerl_kernels.backend import Backend
from tilerl_kernels.mega import MegaForward
from tilerl_kernels.mega_loader import load_sm70

from tilerl.config import qwen38_27b
from tilerl.kv_cache import BatchKv, LinearStatePool, PagedKvPool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["TILERL_SM70_NATIVE_FP8"] = "1"
    cfg, backend = qwen38_27b(), Backend("cuda")
    torch.manual_seed(77)
    model = load_sm70(cfg, args.checkpoint, backend)
    b, total = args.batch, args.context + args.steps + 1
    pages = (total + 15) // 16
    pool = PagedKvPool(b*pages, cfg.num_kv_heads, cfg.head_dim, device=backend.device,
                       dtype=torch.float32, layer_map=cfg.full_attn_layers)
    states = LinearStatePool(b, cfg.num_linear_layers, cfg.linear_num_value_heads,
                             cfg.linear_value_head_dim, device=backend.device, dtype=torch.float32,
                             conv_window=3, conv_dim=cfg.linear_qkv_dim)
    table = torch.arange(b*pages, device=backend.device).reshape(b,pages)
    length = torch.full((b,), args.context, device=backend.device, dtype=torch.long)
    kv = BatchKv(table,length,torch.arange(b,device=backend.device),pool,states)
    # Teacher-forced random inputs test a deterministic work shape, not language quality.
    ids = torch.randint(100,10000,(b,total),device=backend.device)
    for start in range(0,args.context,128):
        end = min(start+128,args.context)
        length.fill_(end)
        model.forward(ids[:,start:end],torch.arange(start,end,device=backend.device),kv,backend,last_only=True)
    mega = MegaForward(model,backend,b,total)
    full_idx = linear_idx = 0
    for row in mega.rows:
        if row[0] == 6:
            cache = mega.buffers[row[6]]
            cache[...,0].copy_(pool.k_pool[full_idx].reshape(b,pages,cfg.num_kv_heads,16,cfg.head_dim)
                               .permute(0,1,3,2,4).reshape(b,pages*16,cfg.num_kv_heads,cfg.head_dim)[:,:total])
            cache[...,1].copy_(pool.v_pool[full_idx].reshape(b,pages,cfg.num_kv_heads,16,cfg.head_dim)
                               .permute(0,1,3,2,4).reshape(b,pages*16,cfg.num_kv_heads,cfg.head_dim)[:,:total])
            full_idx += 1
        elif row[0] == 7:
            base = row[10]
            mega.buffers[base+2].copy_(states.states[:,linear_idx])
            # The pool's active plane is win_parity[slot]; the kernel picks its plane
            # from the absolute position, so land it in context%2 regardless.
            for slot in range(b):
                mega.buffers[base+3][slot,args.context%2].copy_(
                    states.conv_windows[slot,linear_idx,int(states.win_parity[slot])])
            linear_idx += 1
    mega.cursor = args.context
    snapshots = [(t,t.clone()) for t in mega.state_buffers]
    original_k, original_v = pool.k_pool.clone(), pool.v_pool.clone()
    original_s, original_w, original_p = states.states.clone(),states.conv_windows.clone(),states.win_parity.clone()

    def reset():
        pool.k_pool.copy_(original_k)
        pool.v_pool.copy_(original_v)
        states.states.copy_(original_s)
        states.conv_windows.copy_(original_w)
        states.win_parity.copy_(original_p)
        for dest,source in snapshots:
            dest.copy_(source)
        mega.cursor = args.context

    def run(arm, collect=False):
        values = []
        start = time.perf_counter()
        for pos in range(args.context,args.context+args.steps):
            length.fill_(pos+1)
            if arm == "baseline":
                out = model.forward(ids[:,pos:pos+1],torch.tensor([pos],device=backend.device),kv,backend)[:,0]
            else:
                out = mega.step(ids[:,pos].contiguous())
            if collect:
                values.append(out.detach().cpu())
        torch.cuda.synchronize()
        return (time.perf_counter()-start)*1000,values

    for arm in ("baseline", "mega"):
        reset()
        run(arm)
    torch.cuda.reset_peak_memory_stats()
    reset()
    _,ref = run("baseline",True)
    reset()
    _,candidate = run("mega",True)
    worst = 0.0
    for a,c in zip(ref,candidate):
        torch.testing.assert_close(c,a,rtol=1e-2,atol=1e-2)
        worst = max(worst,(a-c).abs().max().item())
    results = []
    for order in (("baseline","mega"),("mega","baseline")):
        for arm in order:
            reset()
            torch.cuda.synchronize()
            elapsed,_ = run(arm)
            results.append({"arm":arm,"order":list(order),"ms":elapsed,
                            "ms_per_step":elapsed/args.steps,"tokens_per_second":b*args.steps/(elapsed/1000)})
    data = {"checkpoint":args.checkpoint,"batch":b,"context":args.context,"steps":args.steps,
            "max_abs":worst,"results":results,"prefill":"not a megakernel; baseline used for common state",
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "resident_weight_bytes":sum(t.numel()*t.element_size() for t in model.params.values()),
            "kv_bytes":pool.k_pool.numel()*pool.k_pool.element_size()+pool.v_pool.numel()*pool.v_pool.element_size()}
    with open(args.output,"w") as stream:
        json.dump(data,stream,indent=2)
    print(json.dumps(data),flush=True)


if __name__ == "__main__":
    main()
