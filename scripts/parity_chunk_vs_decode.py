"""Does a mid-sequence multi-token forward agree with token-by-token decode?

Speculative decoding's verify step feeds depth+1 tokens at once from a state
built by decode. If that path disagrees with T=1 decode, every acceptance
number is noise and spec decode cannot be wired at all. No draft head here —
this isolates the engine.

  python scripts/parity_chunk_vs_decode.py /data00/Qwen3.8-27B-NVFP4 --gpu 7
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=5, help="tokens per multi-token forward")
    ap.add_argument("--text", default="Write a Python function that merges two sorted lists.")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    import numpy as np
    import torch

    from tilerl.config import qwen38_27b
    from tilerl.engine import BatchKv
    from tilerl.kv_cache import BLOCK_TOKENS, LinearStatePool, PagedKvPool
    from tilerl.model import load_hf
    from tilerl.ops.backend import get_backend
    from tilerl.server import get_tokenizer

    backend = get_backend()
    cfg = qwen38_27b()
    trunk = load_hf(cfg, args.source, fuse_projections=True)
    ids0 = list(get_tokenizer(args.source).encode(args.text))

    def arr(x):
        return np.asarray(x, dtype=np.int64)

    def fresh():
        nblk = -(-(len(ids0) + args.gen + 64) // BLOCK_TOKENS) + 2
        pool = PagedKvPool(nblk, cfg.num_kv_heads, cfg.head_dim,
                           num_layers=len(cfg.full_attn_layers), device=backend.device,
                           layer_map=cfg.full_attn_layers)
        states = LinearStatePool(
            1, cfg.num_linear_layers, cfg.linear_num_value_heads, cfg.linear_value_head_dim,
            device=backend.device,
            dtype=torch.float32 if backend.device.type == "cuda" else torch.bfloat16,
            conv_window=cfg.linear_conv_kernel_dim - 1, conv_dim=cfg.linear_qkv_dim)
        bt = torch.arange(nblk, dtype=torch.int32, device=backend.device).reshape(1, nblk)

        def kv(length, q):
            return BatchKv(
                block_table=bt, kv_pool=pool, state_pool=states,
                seq_len=torch.tensor([length], dtype=torch.int32, device=backend.device),
                state_slot=torch.zeros(1, dtype=torch.int32, device=backend.device),
                seq_q_lens=torch.tensor([q], dtype=torch.int32, device=backend.device))
        return kv

    # (a) reference: pure T=1 greedy decode.
    kv = fresh()
    lg = trunk.forward(arr([ids0]), arr(range(len(ids0))), kv(len(ids0), len(ids0)), backend)
    ref = [int(lg[0, -1].argmax())]
    pos = len(ids0)
    for _ in range(args.gen - 1):
        lg = trunk.forward(arr([[ref[-1]]]), arr([pos]), kv(pos + 1, 1), backend)
        ref.append(int(lg[0, -1].argmax()))
        pos += 1

    # (b) same tokens, fed back in chunks of --chunk from a decode-built state.
    kv = fresh()
    trunk.forward(arr([ids0]), arr(range(len(ids0))), kv(len(ids0), len(ids0)), backend)
    got: list[int] = []
    pos = len(ids0)
    while len(got) < args.gen - 1:
        blk = ref[len(got) : len(got) + args.chunk]
        lg = trunk.forward(arr([blk]), arr(range(pos, pos + len(blk))),
                           kv(pos + len(blk), len(blk)), backend)
        got.extend(int(x) for x in lg[0].argmax(-1))
        pos += len(blk)
    got = got[: args.gen - 1]
    want = ref[1:]
    n_ok = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), len(want))
    print(f"chunk={args.chunk}: {n_ok}/{len(want)} positions agree with T=1 decode")
    print("  T=1  :", want)
    print("  chunk:", got)
    assert n_ok == len(want), "multi-token forward disagrees with decode — verify path is broken"


if __name__ == "__main__":
    main()
