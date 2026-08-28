"""Acceptance rate of the bundled NextN/MTP draft head against the trunk.

The one number that decides whether speculative decoding is worth wiring: what
fraction of greedy drafts the trunk would have produced itself. agent-infer
records 33% for this checkpoint's head.

  python scripts/draft_probe.py /data00/Qwen3.8-27B-NVFP4 --gpu 7 --depth 4
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
    ap.add_argument("--depth", type=int, default=4, help="draft tokens per block")
    ap.add_argument("--gen", type=int, default=256,
                    help="tokens of continuation to score over — FIXED across depths, or "
                         "a deeper draft simply generates further into the model's "
                         "repetition regime and reads as a higher acceptance")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--text", default=None, help="real prompt (default: random ids)")
    ap.add_argument("--show", action="store_true", help="print the continuation")
    ap.add_argument("--embed-first", action="store_true", help="fc sees concat(embed, hidden)")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    import torch

    import benchkit as bk
    from tilerl.config import qwen38_27b
    from tilerl.engine import BatchKv
    from tilerl.kv_cache import BLOCK_TOKENS, PagedKvPool
    from tilerl.model import load_hf
    from tilerl.ops.backend import get_backend
    from tilerl.spec import load_draft

    backend = get_backend()
    cfg = qwen38_27b()
    trunk = load_hf(cfg, args.source, fuse_projections=True)
    draft = load_draft(trunk, Path(args.source) / "model_mtp.safetensors",
                       hidden_first=not args.embed_first)
    draft.params = backend.materialize(draft.params)
    print(f"draft: {draft.cfg.num_layers} layer(s), confidence head: "
          f"{draft.has_confidence}, fc order: "
          f"{'hidden,embed' if draft.hidden_first else 'embed,hidden'}")

    # Two pools: the trunk's full-attn planes, and one plane for the draft layer.
    nblk = -(-(args.prompt_len + 4 * args.gen + args.depth + 64) // BLOCK_TOKENS) + 4
    trunk_pool = PagedKvPool(nblk, cfg.num_kv_heads, cfg.head_dim,
                             num_layers=len(cfg.full_attn_layers), device=backend.device,
                             layer_map=cfg.full_attn_layers)
    draft_pool = PagedKvPool(nblk, cfg.num_kv_heads, cfg.head_dim, num_layers=1,
                             device=backend.device, layer_map=(0,))
    from tilerl.kv_cache import LinearStatePool

    states = LinearStatePool(1, cfg.num_linear_layers, cfg.linear_num_value_heads,
                             cfg.linear_value_head_dim, device=backend.device)
    bt = torch.arange(nblk, dtype=torch.int32, device=backend.device).reshape(1, nblk)

    def kv_for(pool, length, q):
        return BatchKv(
            block_table=bt, kv_pool=pool, state_pool=states,
            seq_len=torch.tensor([length], dtype=torch.int32, device=backend.device),
            state_slot=torch.zeros(1, dtype=torch.int32, device=backend.device),
            seq_q_lens=torch.tensor([q], dtype=torch.int32, device=backend.device),
        )

    import numpy as np

    if args.text:
        from tilerl.server import get_tokenizer

        ids = list(get_tokenizer(args.source).encode(args.text))
    else:  # random ids have no structure to predict — a floor, not the real rate
        ids = list(bk.rand_prompt(cfg.vocab_size, args.prompt_len, seed=0))
    hid: list = []
    def arr(x):  # backend ops want arrays, not python lists
        return np.asarray(x, dtype=np.int64)

    prompt_len = len(ids)
    logits = trunk.forward(arr([ids]), arr(range(len(ids))),
                           kv_for(trunk_pool, len(ids), len(ids)), backend, hidden_out=hid)
    accepted = total = blocks = 0
    per_pos = [0] * args.depth  # P(first j+1 drafts all accepted) — sets the depth
    nxt = int(logits[0, -1].argmax())  # the trunk's own next token
    h = hid[-1][:, -1:, :]  # trunk hidden at the position that predicts nxt
    pos = len(ids)  # position of nxt (not yet in the KV)
    ids.append(nxt)
    while len(ids) - prompt_len < args.gen:
        blocks += 1
        chain = [nxt]
        for j in range(args.depth):  # draft off the head's own hidden
            dh: list = []
            dl = draft.forward(h, arr([[chain[-1]]]), arr([pos + j]),
                               kv_for(draft_pool, j + 1, 1), backend, hidden_out=dh)
            chain.append(int(dl[0, -1].argmax()))
            h = dh[-1] if dh else h
        # verify: one trunk forward over [nxt, draft_1..draft_d]
        hid.clear()
        logits = trunk.forward(arr([chain]), arr(range(pos, pos + len(chain))),
                               kv_for(trunk_pool, pos + len(chain), len(chain)), backend,
                               hidden_out=hid)
        want = [int(x) for x in logits[0, :-1].argmax(-1)]
        n_ok = 0
        for j, (a, b) in enumerate(zip(want, chain[1:])):
            total += 1
            if a != b:
                break
            n_ok += 1
            accepted += 1
            per_pos[j] += 1
        # Commit the accepted prefix plus the trunk's own next token, and
        # DROP the rejected drafts — extending by the whole chain would score
        # the draft against text it wrote itself (a flat survival curve near
        # 0.8 that is not an acceptance rate).
        ids.extend(chain[1 : n_ok + 1])
        nxt = int(logits[0, n_ok].argmax())  # bonus token, from the last kept row
        ids.append(nxt)
        h = hid[-1][:, n_ok : n_ok + 1, :]
        pos += n_ok + 1
    n = blocks
    if args.show:
        from tilerl.server import get_tokenizer

        print("  continuation:", repr(get_tokenizer(args.source).decode(ids[prompt_len:])[:200]))
    print(f"depth {args.depth}, {args.gen} generated tokens ({n} blocks): "
          f"accepted {accepted}/{total} = {100 * accepted / max(total, 1):.1f}%")
    print("  survival by position:", " ".join(f"{c / n:.2f}" for c in per_pos))
    # tokens per trunk forward = 1 bonus + expected accepted drafts
    print(f"  tokens per verify: {1 + accepted / n:.2f}  (1.00 = no speculation)")


if __name__ == "__main__":
    main()
