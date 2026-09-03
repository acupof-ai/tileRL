"""What actually caps context at B=1: reserve, slots, or blocks.

Two findings this exists to separate. (1) Load leaves the caching allocator
holding 29.02 GiB reserved against 15.96 allocated on a 31.74 GiB card, and it
is RESERVED that a later pool has to fit beside -- `torch.cuda.mem_get_info`
reports 2.36 GiB free at that point, which is also the number build_engine sizes
PrefixStore from (engine.py:1103). (2) LinearStatePool.step_states scales
slots x width and NOT max_batch, so `--slots 8 --max-batch 2` pays 5.06 GiB for
six slots that can never be admitted.

One arm per process. An engine teardown does NOT return everything (measured:
3.18 GiB retained after `del e; empty_cache()`), so a second arm in the same
process is measured against less memory than it reports -- the same class of
error that made two ctx=512 rows disagree in bench_ctx_decode.

  scripts/v100.sh run kv1 '... probe_kv_ceiling.py --source $CKPT --slots 3'
  scripts/v100.sh run kv2 '... probe_kv_ceiling.py --source $CKPT --slots 3 --no-reclaim'
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS
from tilerl.spec import load_draft

GiB = 1024**3


def mem(tag: str) -> float:
    a = torch.cuda.memory_allocated() / GiB
    r = torch.cuda.memory_reserved() / GiB
    free, total = (x / GiB for x in torch.cuda.mem_get_info())
    print(f"  {tag:26s} alloc {a:6.2f}  reserved {r:6.2f}  free {free:6.2f} / {total:.2f}")
    return free


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--slots", type=int, default=3,
                    help="state slots; step_states scales with THIS, not max_batch")
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--no-reclaim", dest="reclaim", action="store_false",
                    help="skip empty_cache after load: the control arm, and what serve does today")
    ap.add_argument("--decode", type=int, default=8, help="decode this many tokens to prove it runs")
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    backend = get_backend()
    print(f"slots={args.slots} depth={args.depth} max_batch={args.max_batch} "
          f"reclaim={args.reclaim}")
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    # Draft BEFORE materialize, the order cli.py serves in: materialize rewrites the
    # fp4 lm_head into wq/scale/oscale, and read_head_params then rejects all three.
    draft = load_draft(model, args.draft) if args.draft else None
    model.params = backend.materialize(model.params)
    mem("after load")
    if args.reclaim:
        torch.cuda.empty_cache()
    free = mem("before pools")

    # Size the pool from what is visible NOW, which is the whole point of the A/B.
    # GDN is computed, not guessed: states + conv_windows scale with slots, and
    # step_states/step_windows with slots*width -- 79% of the pool at depth 3, and
    # the reason `--slots 8 --max-batch 2` wastes 5 GiB. build_engine's PrefixStore
    # then takes a quarter of what is free AFTER the pools (engine.py:1103), so the
    # blocks only get three quarters of the remainder. 1.0 GiB of headroom for the
    # attention partials, which are transient and scale with B*S.
    n_lin = cfg.num_layers - len(cfg.full_attn_layers)
    s, w = args.slots + 1, (1 + args.depth) if draft else 1
    per_state = n_lin * cfg.linear_num_value_heads * cfg.linear_value_head_dim**2 * 4 / GiB
    per_win = n_lin * (cfg.linear_conv_kernel_dim - 1) * cfg.linear_qkv_dim * 4 / GiB
    gdn = s * (per_state * (1 + w) + per_win * (2 + w))
    blocks = int(max(64, (free - gdn - 1.0) * 0.75 / (2.125 / 1024)))
    print(f"  GDN estimate {gdn:.2f} GiB -> {blocks} blocks "
          f"({blocks * 2.125 / 1024:.2f} GiB) = {blocks * BLOCK_TOKENS} tokens")
    try:
        e = build_engine(cfg, model, backend, num_blocks=blocks, num_slots=args.slots,
                         max_batch=args.max_batch, max_total_tokens=blocks * BLOCK_TOKENS,
                         draft=draft, spec_depth=args.depth if draft else 1)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"  OOM building {blocks} blocks: {str(exc)[:160]}")
        raise SystemExit(1) from None
    mem("after pools")
    sp = e._states
    step = sum(t.numel() * t.element_size() for t in (sp.step_states, sp.step_windows)
               if t is not None) / GiB
    gdn_real = step + sum(t.numel() * t.element_size() for t in (sp.states, sp.conv_windows)
                          if t is not None) / GiB
    print(f"  GDN actual {gdn_real:.2f} GiB (step_states {step:.2f} = "
          f"{100 * step / gdn_real:.0f}%)")

    # A pool that allocates but cannot decode is not capacity. Prefill is ~31 ms per
    # prompt token here, so prove it on a short prompt rather than at the ceiling.
    rid = e.submit(list(range(1, 65)), SamplingParams(temperature=0.0,
                                                      max_new_tokens=args.decode, seed=0))
    for _ in range(4096):
        e.step()
        if rid in e.poll():
            break
    else:
        raise SystemExit("the engine never returned the probe request")
    torch.cuda.synchronize()
    mem("after a decode")
    print(f"  OK: {blocks * BLOCK_TOKENS} token pool built and decoded "
          f"{args.decode} tokens at B={args.max_batch}")


if __name__ == "__main__":
    main()
