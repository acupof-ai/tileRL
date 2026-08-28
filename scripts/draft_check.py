"""Is the draft head wired correctly? One forward, no loop, no state.

Teacher-forced: run the trunk over a real prompt, then run the draft head ONCE
over every position with the true next token as its input. Top-1 agreement with
the trunk's own argmax is the head's quality; mean max-probability says whether
it is confident or emitting noise. Neither number depends on any rollout
bookkeeping, which is where five earlier acceptance readings went wrong.

  python scripts/draft_check.py /data00/Qwen3.8-27B-NVFP4 --gpu 7
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
    ap.add_argument("--swap-fc", action="store_true", help="concat(hidden, embed) instead")
    ap.add_argument("--text", default=(
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "Machine learning models are trained on large datasets to predict the "
        "next token in a sequence, and the quality of those predictions depends "
        "on both the data and the architecture."))
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
    from tilerl.spec import load_draft

    backend = get_backend()
    cfg = qwen38_27b()
    trunk = load_hf(cfg, args.source, fuse_projections=True)
    draft = load_draft(trunk, Path(args.source) / "model_mtp.safetensors")
    draft.params = backend.materialize(draft.params)
    if args.swap_fc:
        # The engine concatenates [embed, hidden]; swapping fc's two column
        # halves tests the other order without a flag in the engine.
        w = draft.params["fc"]
        h = w.shape[-1] // 2
        draft.params["fc"] = torch.cat([w[..., h:], w[..., :h]], dim=-1).contiguous()

    ids = list(get_tokenizer(args.source).encode(args.text))
    t = len(ids)
    nblk = -(-t // BLOCK_TOKENS) + 2

    def pools(layers, layer_map):
        return PagedKvPool(nblk, cfg.num_kv_heads, cfg.head_dim, num_layers=layers,
                           device=backend.device, layer_map=layer_map)

    states = LinearStatePool(
        1, cfg.num_linear_layers, cfg.linear_num_value_heads, cfg.linear_value_head_dim,
        device=backend.device,
        dtype=torch.float32 if backend.device.type == "cuda" else torch.bfloat16,
        conv_window=cfg.linear_conv_kernel_dim - 1, conv_dim=cfg.linear_qkv_dim)
    bt = torch.arange(nblk, dtype=torch.int32, device=backend.device).reshape(1, nblk)

    def kv(pool, length, q):
        return BatchKv(
            block_table=bt, kv_pool=pool, state_pool=states,
            seq_len=torch.tensor([length], dtype=torch.int32, device=backend.device),
            state_slot=torch.zeros(1, dtype=torch.int32, device=backend.device),
            seq_q_lens=torch.tensor([q], dtype=torch.int32, device=backend.device))

    def arr(x):
        return np.asarray(x, dtype=np.int64)

    hid: list = []
    tl = trunk.forward(arr([ids]), arr(range(t)), kv(pools(len(cfg.full_attn_layers),
                       cfg.full_attn_layers), t, t), backend, hidden_out=hid)
    # Position i drafts from trunk hidden_i plus the token it predicted, ids[i+1].
    dk = kv(pools(draft.cfg.num_layers, tuple(range(draft.cfg.num_layers))), t - 1, t - 1)
    dk.state_pool = None
    dl = draft.forward(hid[-1][:, : t - 1], arr([ids[1:]]), arr(range(1, t)), dk, backend)

    # Per-stage scale: a head that is wired right produces a final hidden of
    # roughly the trunk's magnitude. A blow-up or collapse localises the fault
    # better than any guess about which tensor is misnamed.
    import torch as _t

    e = backend.embedding(_t.as_tensor(arr([ids[1:]]), device=backend.device),
                          trunk.params["embed_tokens"])
    en = backend.rmsnorm(e, draft.params["pre_fc_norm_embedding"], cfg.rms_eps)
    hn = backend.rmsnorm(hid[-1][:, : t - 1], draft.params["pre_fc_norm_hidden"], cfg.rms_eps)
    xf = backend.linear(_t.cat([en, hn], dim=-1), draft.params["fc"])
    for nm, v in [("trunk hidden", hid[-1]), ("embed", e), ("norm(embed)", en),
                  ("norm(hidden)", hn), ("fc out", xf)]:
        print(f"  std {nm:<14} {v.float().std().item():>9.4f}  "
              f"absmax {v.float().abs().max().item():>9.2f}")
    print(f"  std {'trunk logits':<14} {tl.float().std().item():>9.4f}  "
          f"absmax {tl.float().abs().max().item():>9.2f}")
    print(f"  std {'draft logits':<14} {dl.float().std().item():>9.4f}  "
          f"absmax {dl.float().abs().max().item():>9.2f}")
    rank = (dl[0] > dl[0].gather(-1, tl[0, 1:].argmax(-1, keepdim=True))).sum(-1)
    print(f"  trunk argmax's rank in the draft's distribution: median "
          f"{rank.float().median().item():.0f} of {cfg.vocab_size}")

    # Bisect the readout before blaming the head. (1) the trunk's own final
    # norm + lm_head over the same hidden must reproduce the trunk's logits —
    # if it does not, `hidden_out` is not the tensor this head is fed. (2) the
    # same hidden through the HEAD's norm isolates whether the readout is sane.
    ln = trunk._linear(backend, backend.rmsnorm(hid[-1], trunk.params["final_norm"],
                                                cfg.rms_eps), "lm_head")
    mn = trunk._linear(backend, backend.rmsnorm(hid[-1], draft.params["norm"],
                                                cfg.rms_eps), "lm_head")
    print(f"  readout: trunk final_norm reproduces trunk logits "
          f"{100 * (ln[0].argmax(-1) == tl[0].argmax(-1)).float().mean():.1f}%; "
          f"through the head's norm {100 * (mn[0].argmax(-1) == tl[0].argmax(-1)).float().mean():.1f}%")

    want = tl[0, 1:].argmax(-1)            # the trunk's own next-token argmax
    got = dl[0].argmax(-1)                 # the draft's guess at the same place
    agree = (want == got).float().mean().item()
    conf = torch.softmax(dl[0].float(), -1).max(-1).values
    print(f"{t} tokens, fc order {'hidden,embed' if args.swap_fc else 'embed,hidden'}")
    print(f"  top-1 agreement with trunk : {100 * agree:.1f}%")
    print(f"  draft max-prob  mean {conf.mean():.4f}  median {conf.median():.4f}  "
          f"max {conf.max():.4f}")
    print(f"  uniform would be {1 / cfg.vocab_size:.2e}")


if __name__ == "__main__":
    main()
