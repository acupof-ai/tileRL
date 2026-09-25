"""Skip the draft forward on prefill chunks whose draft KV decode never reads.

A chunked-prefill row paid one draft forward per interior chunk (6.5-10 s for
512 tokens on V100). The draft attention reads only the trailing W-window of
its OWN dense KV, so an interior chunk ending before the window's first page is
skipped; W=0 (full prefix, the default) skips nothing. The last chunk never
skips: it leaves the chain for the first decode tick.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

from dataclasses import replace

import torch
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import _PHASE_PREFILL, SamplingParams
from tilerl.model import build_random
from tilerl.spec import DraftHead


def _random_draft(cfg, seed, trunk, window=32):
    dcfg = replace(cfg, num_layers=1, full_attn_layers=(0,), fp4=False)
    params = {
        k: v for k, v in build_random(dcfg, seed=seed).params.items() if k.startswith("layers.")
    }
    h = cfg.hidden_size
    gen = torch.Generator().manual_seed(seed)
    params["fc"] = (torch.randn(h, 2 * h, generator=gen) * 0.02).to(torch.bfloat16)
    params["norm"] = torch.ones(h, dtype=torch.bfloat16)
    params["pre_fc_norm_hidden"] = torch.ones(h, dtype=torch.bfloat16)
    return DraftHead(trunk, params, num_layers=1, attn_window_tokens=window)


def _engine(window=32, chunk=8, blocks=64):
    cfg = tiny()
    model = build_random(cfg, seed=7)
    return build_engine(
        cfg=cfg,
        model=model,
        backend=get_backend(),
        num_blocks=blocks,
        num_slots=4,
        max_batch=4,
        max_total_tokens=4096,
        max_num_batched_tokens=chunk,
        draft=_random_draft(cfg, 7, model, window),
        spec_depth=1,
        sparse_k=0,
    )


def _pf_row(rid, n, prefill_from):
    from tilerl.engine import _Req

    return _Req(
        req_id=rid,
        params=SamplingParams(temperature=0.0, max_new_tokens=4, seed=0),
        tokens=[1] * n,
        blocks=[],
        state_slot=0,
        seq_len=prefill_from,
        prefill_from=prefill_from,
        phase=_PHASE_PREFILL,
        own_blocks=0,
    )


def test_window_zero_skips_nothing():
    e = _engine(window=0)
    r = _pf_row(1, n=5120, prefill_from=4608)
    assert e._draft_prefill_skip_ids([r], [512]) == set()
    e.shutdown()


def test_interior_chunks_before_window_are_skipped_last_is_kept():
    # n=80, W=32, chunk=16 -> window first page starts at floor((80-32)/16)*16=48.
    # A chunk ending at s writes positions through s-1: ends 16,32,48 land on or
    # before that page, 64 does not -> skips ids {1,2,3}.
    e = _engine(window=32)
    rows = [_pf_row(i + 1, n=80, prefill_from=pf) for i, pf in enumerate((0, 16, 32, 48))]
    assert e._draft_prefill_skip_ids(rows, [16, 16, 16, 16]) == {1, 2, 3}
    e.shutdown()


def test_last_chunk_is_never_skipped_even_inside_window():
    # The finishing chunk ends exactly at n; it carries the first chain.
    e = _engine(window=32)
    r = _pf_row(1, n=80, prefill_from=64)
    assert e._draft_prefill_skip_ids([r], [16]) == set()
    e.shutdown()


def test_short_prompt_inside_window_skips_nothing():
    e = _engine(window=32)
    r = _pf_row(1, n=24, prefill_from=0)
    assert e._draft_prefill_skip_ids([r], [24]) == set()  # single chunk, also last
    r2 = _pf_row(2, n=40, prefill_from=16)
    assert e._draft_prefill_skip_ids([r2], [8]) == set()  # ends 24 > window start 0
    e.shutdown()


def _drive(engine, prompt, chunk, new_tokens, patch_skip=False):
    """Run one request; return (output, per-tick row ids handed to draft.step)."""
    if patch_skip:
        engine._draft_prefill_skip_ids = lambda prefills, chunks: set()
    seen = []
    orig = engine._draft_step

    def rec(rows):
        seen.append(sorted(r.req_id for r in rows))
        return orig(rows)

    engine._draft_step = rec
    rid = engine.submit(
        [10 + (i % 200) for i in range(prompt)],
        SamplingParams(temperature=0.0, max_new_tokens=new_tokens, seed=0),
    )
    for _ in range(500):
        d = engine.poll()
        if rid in d:
            return d[rid], seen
        engine.step()
    raise AssertionError("request never finished")


def test_skip_calls_draft_step_without_the_prefilling_row_and_old_behavior_does_more():
    # n=40 chunk=8 W=32: interior ends 8..32, window start floor((40-32)/16)=0,
    # so NOTHING is skipped at this size -> use W=16 (window start 16): ends
    # 8 and 16 skipped, 24/32 kept, last 40 kept.
    e_new = _engine(window=16, chunk=8)
    out_new, ticks_new = _drive(e_new, 40, 8, 2)
    assert sum(1 for ids in ticks_new if not ids) == 2, ticks_new

    e_old = _engine(window=16, chunk=8)
    out_old, ticks_old = _drive(e_old, 40, 8, 2, patch_skip=True)
    assert all(ids for ids in ticks_old), ticks_old  # OLD drafts the row every tick
    assert len(ticks_old) == len(ticks_new)
    # The output is trunk-sampled; skipping drafts cannot change greedy tokens.
    assert out_old == out_new


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
