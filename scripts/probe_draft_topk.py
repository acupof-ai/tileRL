"""How much acceptance is in the draft's top-k that top-1 misses?

The proposal: keep top-k candidates per draft position instead of top-1 and
search for the longest path the trunk accepts. Whatever the verification
mechanism costs, the CEILING on that idea is one number — how often the trunk's
own next token sits in the draft's top-k but not at top-1. If top-16 coverage is
barely above top-1, no search pays for itself, and this is far cheaper to
measure than to implement.

Measurable with the head we already ship, no engine change beyond a diagnostic
flag. It also does not presuppose the DSpark integration: the top-1 -> top-k
coverage curve is a property of a draft head, and our 1-layer MTP head bounds
what searching runners-up could add.

  scripts/v100.sh run tk 'export TILERL_QWEN38_SOURCE=$HOME/models/Qwen3.8-27B-NVFP4;
      /usr/bin/python3 -u scripts/probe_draft_topk.py --source $TILERL_QWEN38_SOURCE \
        --draft $TILERL_QWEN38_SOURCE/model-00018-of-00018.safetensors'
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

os.environ.setdefault("TILERL_TARGET", "cuda")
from tilerl import cli  # noqa: E402
from tilerl.cli import _build_model  # noqa: E402
from tilerl.engine import _PHASE_DECODE, SamplingParams, build_engine  # noqa: E402
from tilerl.spec import load_draft  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

KS = (1, 2, 4, 8, 16, 64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()
    cli._QWEN38_SOURCE = args.source

    be = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft)
    e = build_engine(cfg, model, be, num_blocks=1024, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft, spec_depth=args.depth)
    e._keep_draft_logits = True

    rid = e.submit(list(range(10, 10 + args.ctx)),
                   SamplingParams(temperature=0.0, max_new_tokens=args.steps * 4, seed=0))
    req = None
    while req is None or req.phase != _PHASE_DECODE:
        e.step()
        req = next((r for r in e._running if r.req_id == rid), None)
        if req is None:
            raise SystemExit("finished during prefill")

    hits = dict.fromkeys(KS, 0)
    ranks: list[int] = []
    for _ in range(args.steps):
        e.step()
        if e.poll().get(rid) is not None:
            break
        tl, ch = e._trunk_logits, e._verify_chains
        if tl is None or not ch:
            continue
        # The accept test is got[j] == chains[0][j+1] (engine.py:1128): the
        # trunk's pick at chain position j against the draft's NEXT chain entry.
        # So rank the drafted token inside the TRUNK's ordering — that is the
        # quantity a path search would exploit, and comparing the other way
        # round (trunk's pick inside the draft's order) is off by one position
        # and reads ~9% where end-to-end tok/fwd implies >90%.
        for j in range(len(ch[0]) - 1):
            order = tl[0, j].float().argsort(descending=True)
            drafted = ch[0][j + 1]
            rank = int((order == drafted).nonzero()[0])
            ranks.append(rank)
            for k in KS:
                hits[k] += rank < k
        e._trunk_logits = e._verify_chains = None

    n = max(len(ranks), 1)
    print(f"\n# ctx={args.ctx} depth={args.depth}, {len(ranks)} verified positions")
    print("# rank of the DRAFTED token inside the trunk's own ordering")
    print("\n" + "".join(f"{'top-' + str(k):>9}" for k in KS))
    print("".join(f"{100 * hits[k] / n:>8.1f}%" for k in KS))
    if ranks:
        rt = torch.tensor(ranks, dtype=torch.float)
        print(f"\nrank: median {int(rt.median())}, mean {rt.mean():.1f}, "
              f"p90 {int(rt.quantile(0.9))}")
        gap = 100 * (hits[16] - hits[1]) / n
        print(f"\ntop-1 = today's accept rate. The top-1 -> top-16 gap is {gap:.1f} points: "
              "positions\nwhere the trunk would have taken a DIFFERENT token that the draft "
              "still ranked highly.\nThat is the ceiling a path search could recover, before "
              "paying for any extra verify rows.")


if __name__ == "__main__":
    main()
