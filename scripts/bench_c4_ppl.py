"""Wikitext-103 perplexity through tileRL's model directly (teacher-forced, B=1).

C4 isn't cached on the pod; wikitext-103-raw-v1 test is the standard substitute.

  python3 scripts/bench_c4_ppl.py --source /data00/Qwen3.8-27B-NVFP4 --gpu 7 --n 50
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl.config import qwen38_27b
from tilerl.engine import BatchKv
from tilerl.kv_cache import BLOCK_TOKENS, LinearStatePool, PagedKvPool
from tilerl.model import load_hf
from tilerl.server import get_tokenizer
from tilerl_kernels.backend import get_backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--n", type=int, default=50, help="num C4 docs")
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    cfg = qwen38_27b()
    backend = get_backend()
    model = load_hf(cfg, args.source, fuse_projections=True)
    model.params = backend.materialize(model.params)
    tok = get_tokenizer(args.source)

    n_linear = cfg.num_layers - len(cfg.full_attn_layers)
    kv_pool = PagedKvPool(
        64, cfg.num_kv_heads, cfg.head_dim,
        device=backend.device, layer_map=cfg.full_attn_layers,
    )
    state_pool = LinearStatePool(
        1, n_linear, cfg.linear_num_value_heads, cfg.linear_value_head_dim,
        device=backend.device, dtype=torch.float32,
        conv_window=cfg.linear_conv_kernel_dim - 1,
        conv_dim=cfg.linear_qkv_dim,
    )

    num_blocks = (args.seq_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    blocks = [kv_pool.alloc_block() for _ in range(num_blocks)]
    slot = state_pool.alloc_slot()

    # Wikitext-103 test parquet, cached on the pod (HF hub cache).
    import glob

    pq = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/"
            "wikitext-103-raw-v1/test-*.parquet"
        )
    )
    if not pq:
        raise SystemExit("wikitext-103 test parquet not found in HF hub cache")
    import pyarrow.parquet as pq_mod

    text = "\n".join(pq_mod.read_table(pq[0]).column("text").to_pylist())

    total_loss = 0.0
    total_tokens = 0
    # Wikitext is line-per-row; concatenate and chunk into seq_len pieces.
    ids_all = tok.encode(text)
    n_docs = 0
    for start in range(0, len(ids_all) - 16, args.seq_len):
        if n_docs >= args.n:
            break
        ids = ids_all[start : start + args.seq_len]
        if len(ids) < 16:
            continue
        n_docs += 1
        T = len(ids)
        input_ids = torch.tensor([ids], dtype=torch.long, device=backend.device)
        positions = torch.arange(T, dtype=torch.long, device=backend.device).unsqueeze(0)
        bt = torch.zeros(1, kv_pool.num_blocks, dtype=torch.long, device=backend.device)
        bt[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.long, device=backend.device)
        kv = BatchKv(
            block_table=bt,
            seq_len=torch.tensor([T], dtype=torch.long, device=backend.device),
            state_slot=torch.tensor([slot], dtype=torch.long, device=backend.device),
            kv_pool=kv_pool,
            state_pool=state_pool,
            seq_q_lens=torch.tensor([T], dtype=torch.long, device=backend.device),
        )
        with torch.no_grad():
            logits = model.forward(input_ids, positions, kv, backend, last_only=False)
        targets = torch.tensor(ids[1:], dtype=torch.long, device=backend.device)
        log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        total_loss += torch.nn.functional.nll_loss(log_probs, targets, reduction="sum").item()
        total_tokens += T - 1
        state_pool.free_slot(slot)
        slot = state_pool.alloc_slot()
        if n_docs % 10 == 0:
            print(f"  {n_docs}/{args.n} chunks, ppl={math.exp(total_loss / total_tokens):.2f}", flush=True)

    ppl = math.exp(total_loss / total_tokens)
    print(f"Wikitext-103 perplexity: {ppl:.2f} ({total_tokens} tokens, {n_docs} chunks)")


if __name__ == "__main__":
    main()
