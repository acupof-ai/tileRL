"""What a prefix-boundary state snapshot costs, and what caps it.

Each snapshot is a clone of one slot's recurrent state plus its conv window —
device memory, held until something evicts it. This drives N distinct
block-aligned prompts through an engine and reports how many snapshots survive
and what they cost, for both store kinds. Runs unchanged on a tree where the
snapshots live in ``Engine._prefix_state`` and on one where the store owns them.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/bench_prefix_state.py \
        --source /work/Qwen3.8-27B-NVFP4 --prompts 200
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from tilerl_kernels.backend import get_backend

from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.model import load_hf


def retained(engine) -> tuple[int, int]:
    """(snapshots held, bytes) however this tree holds them."""
    side = getattr(engine, "_prefix_state", None)
    if side is not None:  # pre-fix: an engine-side dict the store cannot evict
        n = len(side)
        b = sum(s.nbytes + (w.nbytes if w is not None else 0) for s, w in side.values())
        return n, b
    st = engine._prefix.stats() if hasattr(engine._prefix, "stats") else {}
    return st.get("entries", 0), st.get("state_bytes", 0)


def run(name, cfg, model, backend, store, n_prompts, plen, blocks, newtok=2) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    engine = build_engine(cfg, model, backend, num_blocks=blocks, num_slots=8,
                          decode_graph=False, prefix_store=store)
    rng = np.random.default_rng(11)
    for i in range(n_prompts):
        # Block-aligned prompt: _publish_prefix only fires at prompt_len % BLOCK_TOKENS == 0.
        p = rng.integers(10, cfg.vocab_size - 10, size=plen).tolist()
        wid = engine.submit(p, SamplingParams(temperature=0.0, max_new_tokens=newtok, seed=i))
        for _ in range(plen + newtok + 8):  # capped: a stalled engine must not spin forever
            if wid in engine.poll():
                break
            engine.step()
        else:
            raise RuntimeError(f"request {wid} never finished")
    torch.cuda.synchronize()
    n, b = retained(engine)
    st = engine._prefix.stats() if hasattr(engine._prefix, "stats") else {}
    print(f"{name:<26} snapshots {n:>4}  {b / 2**30:>6.2f} GiB   "
          f"published {engine.stats()['prefix_published']:>4}  evictions {st.get('evictions', 0):>4}  "
          f"peak {(torch.cuda.max_memory_allocated() - base) / 2**30:>6.2f} GiB "
          f"(abs {torch.cuda.max_memory_allocated() / 2**30:.2f})", flush=True)
    engine = None
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--prompts", type=int, default=200)
    p.add_argument("--plen", type=int, default=2 * BLOCK_TOKENS)
    p.add_argument("--blocks", type=int, default=2048)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--newtok", type=int, default=128)
    args = p.parse_args()

    backend = get_backend()
    cfg = qwen38_27b()
    model = load_hf(cfg, args.source)
    one = cfg.num_layers - len(cfg.full_attn_layers)
    print(f"\none snapshot = {one} linear layers x {cfg.linear_num_value_heads} heads x "
          f"{cfg.linear_value_head_dim}^2 state + conv window\n"
          f"{args.prompts} distinct {args.plen}-token prompts, {args.blocks} blocks\n", flush=True)
    run("PrefixStore", cfg, model, backend, None, args.prompts, args.plen, args.blocks)
    run("NoPrefixStore (training)", cfg, model, backend, NoPrefixStore(), args.prompts, args.plen,
        args.blocks)
    # The rollout shape: a training engine publishes at every 16 generated tokens too.
    print(f"\n{args.rollouts} rollouts x {args.newtok} new tokens, NoPrefixStore\n", flush=True)
    run("NoPrefixStore (rollout)", cfg, model, backend, NoPrefixStore(), args.rollouts, args.plen,
        args.blocks, newtok=args.newtok)


if __name__ == "__main__":
    main()
