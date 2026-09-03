"""Does the speculative verify path reproduce plain greedy decode, no draft head?
  default: a mid-sequence multi-token forward vs token-by-token decode.
  --loop:  the block loop (snapshot / verify / roll back / re-absorb) with an
           always-right or always-wrong draft; committed tokens must equal greedy.
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
    ap.add_argument("--chunk", type=int, default=5, help="draft tokens per verify")
    ap.add_argument("--loop", choices=("accept", "reject"), default=None)
    ap.add_argument("--text", default="Write a Python function that merges two sorted lists.")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    import numpy as np
    import torch
    from tilerl_kernels.backend import get_backend

    from tilerl.config import qwen38_27b
    from tilerl.engine import BatchKv
    from tilerl.kv_cache import BLOCK_TOKENS, LinearStatePool, PagedKvPool
    from tilerl.model import load_hf
    from tilerl.server import get_tokenizer

    backend = get_backend()
    cfg = qwen38_27b()
    trunk = load_hf(cfg, args.source, fuse_projections=True)
    ids0 = list(get_tokenizer(args.source).encode(args.text))

    def arr(x):
        return np.asarray(x, dtype=np.int64)

    def fresh():
        nblk = -(-(len(ids0) + args.gen + args.chunk + 64) // BLOCK_TOKENS) + 2
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
        return kv, states

    # reference: T=1 greedy decode
    kv, _ = fresh()
    lg = trunk.forward(arr([ids0]), arr(range(len(ids0))), kv(len(ids0), len(ids0)), backend)
    ref = [int(lg[0, -1].argmax())]
    pos = len(ids0)
    for _ in range(args.gen - 1):
        lg = trunk.forward(arr([[ref[-1]]]), arr([pos]), kv(pos + 1, 1), backend)
        ref.append(int(lg[0, -1].argmax()))
        pos += 1

    if args.loop:
        kv, states = fresh()
        trunk.forward(arr([ids0]), arr(range(len(ids0))), kv(len(ids0), len(ids0)), backend)
        out = [ref[0]]
        pos = len(ids0)
        n_acc = 0
        while len(out) < args.gen:
            i = len(out) - 1
            drafts = (ref[i + 1 : i + 1 + args.chunk] if args.loop == "accept"
                      else [ids0[0]] * args.chunk)
            chain = [out[-1]] + drafts
            snap_state, snap_win = states.states.clone(), states.window_snapshot(0)
            lg = trunk.forward(arr([chain]), arr(range(pos, pos + len(chain))),
                               kv(pos + len(chain), len(chain)), backend)
            n_ok = 0
            for a, b in zip((int(x) for x in lg[0, :-1].argmax(-1)), chain[1:]):
                if a != b:
                    break
                n_ok += 1
            n_acc += n_ok
            out.extend(chain[1 : n_ok + 1])
            out.append(int(lg[0, n_ok].argmax()))
            pos += n_ok + 1
            states.states.copy_(snap_state)
            if snap_win is not None:
                states.window_restore(0, snap_win)
            keep = chain[: n_ok + 1]
            trunk.forward(arr([keep]), arr(range(pos - n_ok - 1, pos)),
                          kv(pos, len(keep)), backend)
        out = out[: args.gen]
        bad = next((i for i, (a, b) in enumerate(zip(ref, out)) if a != b), None)
        print(f"loop={args.loop} chunk={args.chunk}: accepted {n_acc}, "
              f"first divergence from greedy at {bad} (None = clean)")
        print("  greedy   :", ref)
        print("  committed:", out)
        assert bad is None, "the block loop does not reproduce greedy decode"
        return

    kv, _ = fresh()
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
