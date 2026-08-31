"""Where does a draft step's time actually go?

Speculative decode measured SLOWER than dense on this V100 (3.1 tok/s fp4-draft,
6.0 bf16-draft, vs 25.8 dense) even though acceptance is 97% and 5.33 tokens
commit per forward. So the cost is the draft itself, not the policy. This times
one draft step against the trunk forward, and prints which weight format each
draft projection ended up in — inference from end-to-end numbers had already
sent me down two wrong paths.

  TILERL_TARGET=cuda python3 scripts/prof_draft_step.py \
      --source /data00/.../Qwen3.8-27B-NVFP4 --draft <shard> [--fp4]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl.config import qwen38_27b
from tilerl.engine import BatchKv, _quantize_draft
from tilerl.kv_cache import BLOCK_TOKENS, LinearStatePool, PagedKvPool
from tilerl.model import load_hf
from tilerl.spec import load_draft
from tilerl_kernels.backend import get_backend


def timeit(fn, n=10):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--fp4", action="store_true", help="quantize draft to fp4 (else bf16 dense)")
    ap.add_argument("--fp8", action="store_true", help="quantize draft to fp8")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    cfg = qwen38_27b()
    backend = get_backend()
    print(f"arch={backend.arch} io={backend.io} "
          f"fp8_kernel={backend.has_kernel('linear_fp8')} "
          f"fp4_gemv={backend.has_kernel('linear_fp4_gemv')} "
          f"fp4_gemv_m={backend.has_kernel('linear_fp4_gemv_sm70_m')}")

    model = load_hf(cfg, args.source, fuse_projections=True)
    model.params = backend.materialize(model.params)
    draft = load_draft(model, args.draft)
    if args.fp4 or args.fp8:
        draft.params.update(
            backend.materialize(_quantize_draft(draft.params, fp4=args.fp4))
        )
        for k in [k for k, v in list(draft.params.items())
                  if v.ndim == 2 and v.shape[0] >= 128 and v.shape[1] >= 128]:
            del draft.params[k]  # drop the dense twin so Model._linear takes the quantized path
    else:
        draft.params.update(backend.materialize(draft.params))
    fmt = "fp4" if args.fp4 else "fp8" if args.fp8 else "bf16"
    q = [k for k in draft.params if "q_proj" in k]
    print(f"draft format={fmt}  q_proj keys={sorted(q)}")

    dev = backend.device
    n_lin = cfg.num_layers - len(cfg.full_attn_layers)
    kv = PagedKvPool(64, cfg.num_kv_heads, cfg.head_dim, device=dev,
                     layer_map=cfg.full_attn_layers)
    sp = LinearStatePool(1, n_lin, cfg.linear_num_value_heads, cfg.linear_value_head_dim,
                         device=dev, dtype=torch.float32,
                         conv_window=cfg.linear_conv_kernel_dim - 1, conv_dim=cfg.linear_qkv_dim)
    dkv_pool = PagedKvPool(64, draft.cfg.num_kv_heads, draft.cfg.head_dim, device=dev,
                           num_layers=draft.cfg.num_layers,
                           layer_map=tuple(range(draft.cfg.num_layers)))
    T = 8
    blocks = [kv.alloc_block() for _ in range((T + BLOCK_TOKENS - 1) // BLOCK_TOKENS)]
    dblocks = [dkv_pool.alloc_block() for _ in range(len(blocks))]
    slot = sp.alloc_slot()

    def mk(pool, blks, q_len):
        bt = torch.zeros(1, pool.num_blocks, dtype=torch.long, device=dev)
        bt[0, : len(blks)] = torch.tensor(blks, device=dev)
        return BatchKv(block_table=bt, seq_len=torch.tensor([T], device=dev),
                       state_slot=torch.tensor([slot], device=dev),
                       kv_pool=pool, state_pool=sp,
                       seq_q_lens=torch.tensor([q_len], device=dev))

    ids = torch.randint(0, 1000, (1, T), device=dev)
    pos = torch.arange(T, device=dev).unsqueeze(0)
    hid: list[torch.Tensor] = []
    with torch.no_grad():
        model.forward(ids, pos, mk(kv, blocks, T), backend, last_only=False, hidden_out=hid)
    h = hid[0]

    # M=1 decode shapes: one query position, which is what a draft step runs.
    id1, pos1, h1 = ids[:, :1], pos[:, :1], h[:, :1]
    with torch.no_grad():
        t_trunk = timeit(lambda: model.forward(id1, pos1, mk(kv, blocks, 1), backend))
        t_draft = timeit(lambda: draft.forward(h1, id1, pos1, mk(dkv_pool, dblocks, 1), backend))
    print(f"\ntrunk forward (M=1): {t_trunk:7.2f} ms")
    print(f"draft step    (M=1): {t_draft:7.2f} ms   ({t_draft / t_trunk:.2f}x the trunk)")
    print(f"depth 6 draft cost : {6 * t_draft:7.2f} ms")
    print(f"=> tick = {t_trunk + 6 * t_draft:.1f} ms for ~5.3 tokens "
          f"= {5.3 / (t_trunk + 6 * t_draft) * 1000:.1f} tok/s")
    # A 456 M-param head is 0.25 ms at fp4 / 1.01 ms at bf16 against 900 GB/s.
    print(f"   bandwidth floor for this head: fp4 0.25 ms, bf16 1.01 ms")
    if t_draft > 5:
        print(f"   -> {t_draft:.0f} ms is {t_draft / 1.01:.0f}x off even the bf16 floor: "
              f"the draft is NOT on a fused kernel")


if __name__ == "__main__":
    main()
