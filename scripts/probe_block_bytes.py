"""Measure a KV block's real byte cost, instead of trusting a comment's 0.92 MB.

Builds the pools the way build_engine does (engine.py:1191, :358) so the number is
the shipped shape, not a reconstruction.
"""
import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import torch

from tilerl.config import qwen38_27b
from tilerl.kv_cache import BLOCK_TOKENS, PagedKvPool


def main() -> None:
    cfg = qwen38_27b()
    n = 64
    for io in (torch.float32, torch.bfloat16):
        trunk = PagedKvPool(n, cfg.num_kv_heads, cfg.head_dim, device="cpu",
                            layer_map=cfg.full_attn_layers, dtype=io)
        tb = sum(t.numel() * t.element_size() for t in (trunk.k_pool, trunk.v_pool))
        # The draft mirrors num_blocks with its own 1-layer plane, same dtype.
        draft = PagedKvPool(n, cfg.num_kv_heads, cfg.head_dim, num_layers=1,
                            device="cpu", layer_map=(0,), dtype=io)
        db = sum(t.numel() * t.element_size() for t in (draft.k_pool, draft.v_pool))
        per = (tb + db) / n
        print(f"io={io}: planes={len(cfg.full_attn_layers)} kv_heads={cfg.num_kv_heads} "
              f"head_dim={cfg.head_dim} tok/blk={BLOCK_TOKENS}")
        print(f"  trunk {tb/n/2**20:.4f} + draft {db/n/2**20:.4f} = "
              f"{per/2**20:.4f} MiB/block = {per/1e6:.4f} MB/block")


if __name__ == "__main__":
    main()
