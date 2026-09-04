"""Does the checkpoint's MTP head predict the trunk's next token?

The draft is only worth its verify rows if its top-1 agrees with the trunk often
enough. load_draft has a known trap — Qwen3_5RMSNorm is zero-centered, so a head
read straight from safetensors needs +1 folded into every norm, and without it
the logits come back ANTI-correlated (argmax ranked 248191/248320). This measures
the thing that matters: top-1 agreement and the trunk's rank of the draft pick.

  TILERL_TARGET=cuda python3 scripts/check_mtp_draft.py \
      --source /data00/home/chenkailun.c/models/Qwen3.8-27B-NVFP4 \
      --draft  /data00/home/chenkailun.c/models/Qwen3.8-27B-NVFP4/model-00018-of-00018.safetensors
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl_kernels.backend import get_backend

from tilerl.config import qwen38_27b
from tilerl.engine import BatchKv
from tilerl.kv_cache import BLOCK_TOKENS, LinearStatePool, PagedKvPool
from tilerl.model import load_hf
from tilerl.server import get_tokenizer
from tilerl.spec import load_draft

PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Japan is Tokyo. Large language models predict the next token "
    "given all previous tokens, which is why speculative decoding works: a small "
    "draft model proposes and the large model verifies."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    cfg = qwen38_27b()
    backend = get_backend()
    model = load_hf(cfg, args.source, fuse_projections=True)
    model.params = backend.materialize(model.params)
    tok = get_tokenizer(args.source)
    draft = load_draft(model, args.draft)
    draft.params = backend.materialize(draft.params)
    print(f"draft loaded: {len(draft.params)} params, has_confidence={draft.has_confidence}")

    ids = tok.encode(PROMPT)
    T = len(ids)
    dev = backend.device
    n_linear = cfg.num_layers - len(cfg.full_attn_layers)
    kv = PagedKvPool(64, cfg.num_kv_heads, cfg.head_dim, device=dev,
                     layer_map=cfg.full_attn_layers)
    sp = LinearStatePool(1, n_linear, cfg.linear_num_value_heads, cfg.linear_value_head_dim,
                         device=dev, dtype=torch.float32,
                         conv_window=cfg.linear_conv_kernel_dim - 1, conv_dim=cfg.linear_qkv_dim)
    blocks = [kv.alloc_block() for _ in range((T + BLOCK_TOKENS - 1) // BLOCK_TOKENS)]
    slot = sp.alloc_slot()
    bt = torch.zeros(1, kv.num_blocks, dtype=torch.long, device=dev)
    bt[0, : len(blocks)] = torch.tensor(blocks, device=dev)
    bkv = BatchKv(
        block_table=bt,
        seq_len=torch.tensor([T], device=dev),
        state_slot=torch.tensor([slot], device=dev),
        kv_pool=kv, state_pool=sp,
        seq_q_lens=torch.tensor([T], device=dev),
    )

    inp = torch.tensor([ids], dtype=torch.long, device=dev)
    pos = torch.arange(T, dtype=torch.long, device=dev).unsqueeze(0)
    hid: list[torch.Tensor] = []  # model.forward appends the pre-final-norm state
    with torch.no_grad():
        logits = model.forward(inp, pos, bkv, backend, last_only=False, hidden_out=hid)
    trunk_tok = logits[0].float().argmax(-1)  # trunk's pick at every position

    # The draft at position t predicts t+1 FROM the trunk's own token there, so
    # feed it the real ids: this measures the head, not error accumulation.
    # Its own KV plane — sharing the trunk's would overwrite the trunk's keys.
    dkv_pool = PagedKvPool(64, draft.cfg.num_kv_heads, draft.cfg.head_dim, device=dev,
                           num_layers=draft.cfg.num_layers,
                           layer_map=tuple(range(draft.cfg.num_layers)))
    dblocks = [dkv_pool.alloc_block() for _ in range(len(blocks))]
    dbt = torch.zeros(1, dkv_pool.num_blocks, dtype=torch.long, device=dev)
    dbt[0, : len(dblocks)] = torch.tensor(dblocks, device=dev)
    dkv = BatchKv(
        block_table=dbt,
        seq_len=torch.tensor([T], device=dev),
        state_slot=torch.tensor([slot], device=dev),
        kv_pool=dkv_pool, state_pool=sp,
        seq_q_lens=torch.tensor([T], device=dev),
    )
    with torch.no_grad():
        dlog = draft.forward(hid[0], inp, pos, dkv, backend)
    draft_tok = dlog[0].float().argmax(-1)

    n = T - 1
    agree = (draft_tok[:n] == trunk_tok[:n]).sum().item()
    # Where the draft disagrees, how highly does the trunk rank the draft's pick?
    rank = (logits[0, :n].float() > logits[0, :n].float()
            .gather(-1, draft_tok[:n, None])).sum(-1)
    print(f"positions={n}")
    print(f"top1 agreement : {agree}/{n} = {agree / n:.1%}")
    print(f"trunk rank of draft pick: median={rank.median().item():.0f} "
          f"mean={rank.float().mean().item():.1f}")
    print(f"draft pick in trunk top-5 : {(rank < 5).sum().item() / n:.1%}")
    # 248320 vocab: a rank near 248k means the norms were not folded (+1).
    assert rank.float().mean().item() < 1000, (
        f"draft is anti-correlated with the trunk (mean rank {rank.float().mean().item():.0f} "
        f"of {cfg.vocab_size}) — the Qwen3_5RMSNorm +1 fold is likely missing"
    )
    exp = agree / n
    print(f"\nexpected accepted tokens/forward at depth d (geometric, p={exp:.3f}):")
    for d in (1, 2, 3, 4):
        acc = sum(exp**k for k in range(1, d + 1))
        print(f"  depth {d}: {1 + acc:.2f} committed  -> {1000 / (39 + d * 0.25) * (1 + acc):.1f} tok/s")


if __name__ == "__main__":
    main()
