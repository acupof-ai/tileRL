"""How often does a row draft a SHORT chain, and what does it cost in draft tokens?

`spec.py:474` drops a row from the chain loop when `hi + j` would pass the blocks it
owns, and the `ponytail:` comment says so: "a row at a block boundary drafts shorter".
The engine grows blocks to cover `seq_len - 1` before drafting (engine.py:787) but the
chain needs up to `hi + depth - 1`, so the last positions of a chain can fall outside.

The clamp is monotone -- once a row fails at j it fails at every larger j -- so it
SHORTENS the chain rather than leaving a KV hole (which is the failure mode
engine.py:783 warns about). A short chain is not a correctness bug; it is lost
speculation, and this measures how much.

Reports, per row per step: chain positions granted vs depth-1 requested, and whether any
shortfall tracks `seq_len % BLOCK_TOKENS`.

**RESULT: the clamp never fires** -- 0 short chains in 156 row-steps, 312 of 312 positions
granted. So the odd draft batch sizes probe_draft_batch.py found (n=3, n=7) are NOT this
clamp: they come from spec.py:372 dropping a finished row from `plan` before the chain loop
runs. Kept as the negative control that separates the two, since the clamp's own comment
describes exactly the symptom and fits the data without being the cause.

    uv run python scripts/probe_short_chain.py
"""

from __future__ import annotations

import collections
import os
from dataclasses import replace

os.environ.setdefault("TILERL_TARGET", "cpu")

import torch  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.config import tiny  # noqa: E402
from tilerl.engine import BLOCK_TOKENS, SamplingParams, build_engine  # noqa: E402
from tilerl.model import build_random  # noqa: E402
from tilerl.spec import DraftHead  # noqa: E402

#: (granted, requested, seq_len % BLOCK_TOKENS, blocks_owned) per row per step.
ROWS: list[tuple[int, int, int, int]] = []


def _spy() -> None:
    """Count chain positions per row by re-deriving the loop's own predicate.

    Wraps `step` and inspects the rows it was handed, rather than counting kernel
    calls: `len(live)` alone cannot say WHICH rows were dropped, and the cost is
    per-row.
    """
    inner = DraftHead.step

    def step(self, rows):
        live = [r for r in rows if r.hidden is not None and not r.done]
        want = self.width - 2  # chain positions after the pooled prefill
        for r in live:
            hi = r.seq_len - 1
            cap = len(r.blocks) * BLOCK_TOKENS
            granted = sum(1 for j in range(1, self.width - 1) if hi + j < cap)
            ROWS.append((granted, want, r.seq_len % BLOCK_TOKENS, len(r.blocks)))
        return inner(self, rows)

    DraftHead.step = step


def _draft(cfg, trunk, seed: int) -> DraftHead:
    dcfg = replace(cfg, num_layers=1, full_attn_layers=(0,), fp4=False)
    params = {k: v for k, v in build_random(dcfg, seed=seed).params.items()
              if k.startswith("layers.")}
    h = cfg.hidden_size
    gen = torch.Generator().manual_seed(seed)
    params["fc"] = (torch.randn(h, 2 * h, generator=gen) * 0.02).to(torch.bfloat16)
    params["norm"] = torch.ones(h, dtype=torch.bfloat16)
    params["pre_fc_norm_hidden"] = torch.ones(h, dtype=torch.bfloat16)
    return DraftHead(trunk, params, num_layers=1)


def main() -> None:
    _spy()
    cfg = tiny()
    backend = get_backend()
    trunk = build_random(cfg, seed=0)
    eng = build_engine(
        cfg, trunk, backend,
        num_blocks=128, num_slots=16, max_batch=4, max_total_tokens=1024,
        draft=_draft(cfg, trunk, seed=1), spec_depth=3,
    )
    sp = SamplingParams(max_new_tokens=40, temperature=0.0)
    for i in range(4):
        eng.submit(list(range(2, 2 + 7 + 5 * i)), sp)
        if i % 2:
            eng.step()
    for _ in range(400):
        eng.poll()
        eng.step()

    if not ROWS:
        raise SystemExit("nothing recorded -- DraftHead.step moved")
    short = [r for r in ROWS if r[0] < r[1]]
    got, want = sum(r[0] for r in ROWS), sum(r[1] for r in ROWS)
    print(f"row-steps: {len(ROWS)}, short: {len(short)} ({len(short)/len(ROWS)*100:.1f}%)")
    print(f"chain positions granted {got} of {want} requested "
          f"({(want - got) / want * 100:.1f}% lost)")
    if not short:
        print("\nThe clamp never fired. So an odd draft n is NOT block-boundary alignment --\n"
              "it is spec.py:372 dropping a finished row from `plan` before the chain loop,\n"
              "which probe_draft_batch.py cannot distinguish because it only sees len(live).")
        return
    mods = collections.Counter(r[2] for r in short)
    allm = collections.Counter(r[2] for r in ROWS)
    print("\nseq_len % BLOCK_TOKENS, short / total at that residue:")
    for m in sorted(allm):
        s = mods.get(m, 0)
        mark = "  <-- always short" if s and s == allm[m] else ""
        print(f"  {m:>3}: {s:>4} / {allm[m]:<4}{mark}")
    print(f"\nBLOCK_TOKENS = {BLOCK_TOKENS}; a shortfall only at the top residues is the "
          f"boundary,\nnot the harness.")


if __name__ == "__main__":
    main()
