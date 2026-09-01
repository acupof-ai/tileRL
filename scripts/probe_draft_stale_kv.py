"""Does the draft head read stale KV over the prompt?

_draft_step only ever forwards positions [draft_pos+1 .. seq_len-1], so the
draft's own KV pool is never written for the PROMPT positions — yet its
attention reads [0, seq_len). On a recycled block those positions hold the
previous request's draft KV. Position 0 is zeroed by hand (engine.py:1033-1038);
nothing else is.

If that matters, acceptance should depend on prompt length in a way that
zeroing the whole prompt range removes. This runs the same generation twice on
one engine — once as-is, once with the draft pool zeroed before each request —
and compares tokens per trunk forward.

  scripts/v100.sh run sk 'CKPT=...; /usr/bin/python3 -u scripts/probe_draft_stale_kv.py \
      --source $CKPT --draft $CKPT/model-00018-of-00018.safetensors'
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.spec import load_draft
from tilerl_kernels.backend import get_backend

# Prompt lengths chosen to straddle the measured acceptance gap: coding
# (prompt 60) accepted 0.77, dialogue (prompt 158) only 0.57.
LENS = [32, 160, 512]


def run(e, prompt_len: int, tokens: int, wipe: bool) -> tuple[float, int, int]:
    if wipe:  # the whole draft pool, not just position 0
        e._draft_kv.k_pool.zero_()
        e._draft_kv.v_pool.zero_()
    s0 = e.stats()
    rid = e.submit(list(range(10, 10 + prompt_len)),
                   SamplingParams(temperature=0.0, max_new_tokens=tokens, seed=0))
    out = None
    while out is None:
        e.step()
        out = e.poll().get(rid)
    s1 = e.stats()
    fwd = s1["decode_forwards"] - s0["decode_forwards"]
    acc = s1["spec_accepted"] - s0["spec_accepted"]
    dr = s1["spec_drafted"] - s0["spec_drafted"]
    return len(out) / max(fwd, 1), acc, dr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=96)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft)
    e = build_engine(cfg, model, backend, num_blocks=512, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft, spec_depth=args.depth)
    run(e, 32, 16, False)  # warm the graphs
    torch.cuda.synchronize()

    print(f"{'prompt':>7} {'tok/fwd dirty':>14} {'tok/fwd wiped':>14} {'delta':>7}")
    for n in LENS:
        # Dirty first: it inherits whatever the previous request left behind,
        # which is exactly the condition being tested.
        d, _, _ = run(e, n, args.tokens, wipe=False)
        w, _, _ = run(e, n, args.tokens, wipe=True)
        print(f"{n:>7} {d:>14.2f} {w:>14.2f} {w - d:>+7.2f}")


if __name__ == "__main__":
    main()
