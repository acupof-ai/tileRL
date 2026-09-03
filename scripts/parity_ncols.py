"""Does ncols=2 generate the same text as ncols=1, on a real prompt?

The prefill A/B's own parity signal was empty: it fed synthetic token ids
(`range(base, base+ctx)`) and every arm returned first-token id 0, including the
run whose kernels were deliberately wrong
(errors/2026-09-03-the-ab-measured-abl-not-ncols.md). A tell that reads the same
for a correct and an incorrect kernel is not a check.

ncols=2 pairs output column j with j + N/2 and gives one thread two accumulators,
so its failure mode is HALF the output columns being wrong -- which a real greedy
continuation catches immediately and a synthetic-id run does not.

Greedy, temperature 0, one engine, flipping backend._NCOLS between generations.
Prefill and decode both, since prefill is where M reaches the 32-row rung.

  scripts/v100.sh run pncols 'CKPT=...; /usr/bin/python3 -u scripts/parity_ncols.py \
      --source $CKPT'
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tilerl_kernels import backend as bk_mod
from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine

PROMPTS = (
    "Write a Python function that merges two sorted lists.",
    "Explain in three sentences why a GPU is faster than a CPU for matrix multiply.",
)


def gen(e, ids: list[int], n: int) -> list[int]:
    rid = e.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=n, seed=0))
    out = None
    while out is None:
        e.step()
        out = e.poll().get(rid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--gen", type=int, default=24)
    #: 600 tokens puts prefill above the 32-row rung (chunk budget 512), so the
    #: ncols kernel is exercised at the M it was accepted at, not only at M=1.
    ap.add_argument("--pad", type=int, default=600, help="repeat the prompt to this length")
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    from tilerl.server import get_tokenizer

    be = get_backend()
    tok = get_tokenizer(args.source)
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    e = build_engine(cfg, model, be, num_blocks=1024, num_slots=4, max_batch=4,
                     max_total_tokens=8192)

    bad = 0
    for text in PROMPTS:
        ids = list(tok.encode(text))
        for label, ids_in in (("short", ids), ("long", (ids * (args.pad // len(ids) + 1))[:args.pad])):
            outs = {}
            for nc in (1, 2):
                bk_mod._NCOLS = nc
                outs[nc] = gen(e, list(ids_in), args.gen)
            ok = outs[1] == outs[2]
            bad += not ok
            print(f"\n{label:>5} M{'>8' if len(ids_in) > 8 else '=1'} "
                  f"{len(ids_in):>4} tok  {'MATCH' if ok else 'DIFFER'}")
            print(f"  nc1: {tok.decode(outs[1])!r}")
            if not ok:
                print(f"  nc2: {tok.decode(outs[2])!r}")
                d = next((i for i, (a, b) in enumerate(zip(outs[1], outs[2])) if a != b), None)
                print(f"  first divergence at token {d}: {outs[1][d]} vs {outs[2][d]}")

    print(f"\n{'PARITY HOLDS' if not bad else f'{bad} PROMPT(S) DIVERGED'}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
