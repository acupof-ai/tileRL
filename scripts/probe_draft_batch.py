"""How many distinct batch sizes does the draft hand its kernels per request?

Last measurement (wins/2026-09-04-b-is-a-shape-axis-and-decode-already-buckets-it.md)
left one open scope: `engine.py:817` buckets B on the decode graph path, but `spec.py`
contains "graph" zero times, so the draft always runs eager and its own B is never
bucketed. `spec.py:412` takes `n = len(plan)` raw, and the chain loop at `:484` takes
`len(live)`, which SHRINKS mid-chain as rows hit block boundaries.

Counts the distinct n values `DraftHead.step` produces across a multi-row run. This is
a property of the plan/live arithmetic, not of the arch, so the CPU twin answers it --
unlike the kernel-shape question, which needed the pod (probe_b_axis.py records nothing
here because CPU dispatches neither B-baking kernel).

    uv run python scripts/probe_draft_batch.py
"""

from __future__ import annotations

import collections
import os

os.environ.setdefault("TILERL_TARGET", "cpu")

from dataclasses import replace  # noqa: E402

import torch  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.config import tiny  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.model import build_random  # noqa: E402
from tilerl.spec import DraftHead  # noqa: E402

PLAN_N: collections.Counter = collections.Counter()
LIVE_N: collections.Counter = collections.Counter()


def _spy() -> None:
    """Record the batch size the draft's two forward sites see.

    Delimits the sites by wrapping `step` rather than inferring from shapes: the
    first `forward` inside each step is the pooled prefill (spec.py:451), the rest
    are chain positions (spec.py:490). Guessing from `ids.shape[1] > 1` was wrong --
    a depth-1 tick's prefill is also 1 wide, so it counted as a chain step, which is
    how 22 chain calls appeared against 2 prefills when every step makes exactly one
    of each.
    """
    inner_f, inner_s = DraftHead.forward, DraftHead.step
    first = [True]

    def step(self, rows):
        first[0] = True
        return inner_s(self, rows)

    def forward(self, hidden, ids, positions, kv, backend, **kw):
        (PLAN_N if first[0] else LIVE_N)[ids.shape[0]] += 1
        first[0] = False
        return inner_f(self, hidden, ids, positions, kv, backend, **kw)

    DraftHead.step, DraftHead.forward = step, forward


def _draft(cfg, trunk, seed: int) -> DraftHead:
    """A random 1-layer head, same shape as tests/test_e2e.py:948."""
    dcfg = replace(cfg, num_layers=1, full_attn_layers=(0,), fp4=False)
    params = {k: v for k, v in build_random(dcfg, seed=seed).params.items()
              if k.startswith("layers.")}
    h = cfg.hidden_size
    gen = torch.Generator().manual_seed(seed)
    params["fc"] = (torch.randn(h, 2 * h, generator=gen) * 0.02).to(torch.bfloat16)
    params["norm"] = torch.ones(h, dtype=torch.bfloat16)
    params["pre_fc_norm_hidden"] = torch.ones(h, dtype=torch.bfloat16)
    return DraftHead(trunk, params, num_layers=1)


def _run(max_batch: int, n_reqs: int, depth: int = 3) -> tuple[dict, dict]:
    """One engine, staggered arrivals; returns the two sites' n histograms."""
    PLAN_N.clear()
    LIVE_N.clear()
    cfg = tiny()
    backend = get_backend()
    trunk = build_random(cfg, seed=0)
    eng = build_engine(
        cfg, trunk, backend,
        num_blocks=128, num_slots=16, max_batch=max_batch, max_total_tokens=1024,
        draft=_draft(cfg, trunk, seed=1), spec_depth=depth,
    )
    sp = SamplingParams(max_new_tokens=8, temperature=0.0)
    # Stagger arrivals so the batch forms and drains: that is what varies n.
    for i in range(n_reqs):
        eng.submit(list(range(2, 2 + 7 + 4 * i)), sp)
        if i % 2:
            eng.step()
    for _ in range(96):
        eng.poll()
        eng.step()
    return dict(PLAN_N), dict(LIVE_N)


def main() -> None:
    _spy()
    print(f"{'max_batch':>10} {'reqs':>5} {'prefill n':>22} {'chain n':>22} {'distinct':>9}")
    seen_all = set()
    for mb, nr in ((1, 3), (2, 4), (4, 6), (8, 10)):
        pl, lv = _run(mb, nr)
        seen_all |= set(pl) | set(lv)
        print(f"{mb:>10} {nr:>5} {str(dict(sorted(pl.items()))):>22} "
              f"{str(dict(sorted(lv.items()))):>22} {len(set(pl) | set(lv)):>9}")
        # Every step calls the prefill site once, then up to depth-1 chain positions:
        # spec.py:475 breaks when no row has block room left, so the chain count is
        # BOUNDED by 2x, not equal to it (21 prefills gave 40 chain calls, not 42).
        # An equality here is what a first version asserted, and it failed on the real
        # early break rather than on a broken delimiter.
        assert 0 < sum(lv.values()) <= sum(pl.values()) * 2, (
            f"depth 3 allows at most 2 chain calls per step: {sum(pl.values())} "
            f"prefills vs {sum(lv.values())} chain calls -- the delimiter is wrong")
    print(f"\ndistinct draft batch sizes across all arms: {sorted(seen_all)}")
    print("n tracks the live row count, so it is bounded by max_batch and nothing\n"
          "rounds it -- each value is a separate eager kernel specialization.")
    assert PLAN_N or LIVE_N, "the spy recorded nothing -- DraftHead.forward moved"
    assert PLAN_N or LIVE_N, "the spy recorded nothing -- DraftHead.forward moved"


if __name__ == "__main__":
    main()
