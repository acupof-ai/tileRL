"""#621: dense+d1 concurrent block-boundary defect.

The eager decode path grew trunk blocks AFTER commit (an inline loop immediately
before ``draft.step``); the captured-graph path assumed the PRE-fork growth was
enough. When a commit lands exactly on a 16-token boundary (seq_len = k*16+1),
the draft's post-commit write ``hi = seq_len-1 = k*16`` needs the (k+1)th block
the graph row did not hold, so ``spec.DraftHead.step`` asserted and failed the
whole batched tick (the V100 4x2k warmup 500, ~1/3 of boots).

The fix is one shared planner, ``Engine.ensure_draft_write_blocks``, called from
BOTH paths right before ``draft.step``. These are pure planning gates on rows
constructed at the boundary (no forward, no graph): the multi-row alignment is
set explicitly instead of being scheduled into it.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

from dataclasses import replace

import torch
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import _PHASE_DECODE, BLOCK_TOKENS, SamplingParams, _Req
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import build_random
from tilerl.spec import DraftHead


def _random_draft(cfg, seed: int, trunk):
    dcfg = replace(cfg, num_layers=1, full_attn_layers=(0,), fp4=False)
    params = {k: v for k, v in build_random(dcfg, seed=seed).params.items()
              if k.startswith("layers.")}
    h = cfg.hidden_size
    gen = torch.Generator().manual_seed(seed)
    params["fc"] = (torch.randn(h, 2 * h, generator=gen) * 0.02).to(torch.bfloat16)
    params["norm"] = torch.ones(h, dtype=torch.bfloat16)
    params["pre_fc_norm_hidden"] = torch.ones(h, dtype=torch.bfloat16)
    return DraftHead(trunk, params, num_layers=1)


def _engine(num_blocks=64, slots=4, batch=4, sparse_k=0, kv_cold=0):
    cfg = tiny()
    model = build_random(cfg, seed=7)
    kw = dict(
        cfg=cfg, model=model, backend=get_backend(), num_blocks=num_blocks,
        num_slots=slots, max_batch=batch, max_total_tokens=4096,
        max_num_batched_tokens=512, prefix_store=NoPrefixStore(),
        draft=_random_draft(cfg, 7, model), spec_depth=1, sparse_k=sparse_k,
    )
    if kv_cold:
        kw["kv_cold_bytes"] = kv_cold
    return build_engine(**kw)


def _row(engine, rid, seq_len, held, sparse=False) -> _Req:
    """A d1 decode row owning exactly ``held`` trunk blocks at materialized
    ``seq_len``, with a real slot and blocks so _release works. A boundary row
    has seq_len = k*16+1 held on k blocks; a q=2 row k*16+2 on k blocks."""
    slot = engine._states.alloc_slot()
    blocks = [engine._kv.alloc_block() for _ in range(held)]
    r = _Req(
        req_id=rid,
        params=SamplingParams(temperature=0.0, max_new_tokens=4, seed=0),
        tokens=[0] * seq_len,
        blocks=blocks,
        state_slot=slot,
        seq_len=seq_len,
        phase=_PHASE_DECODE,
        prefill_from=seq_len,
        own_blocks=held,
        sparse_on=sparse,
    )
    r.output = [0]
    engine._running.append(r)
    return r


def test_q1_boundary_rows_are_grown_one_block_each():
    """q=1 minimal trigger. Four rows committed onto k*16 each holding k blocks;
    ensure_draft_write_blocks grows every one to k+1 and returns all four."""
    e = _engine()
    k = 3
    rows = [_row(e, rid=i + 1, seq_len=k * BLOCK_TOKENS + 1, held=k) for i in range(4)]
    free0 = e._kv.free_blocks
    kept = e.ensure_draft_write_blocks(rows)
    assert kept is rows
    assert len(kept) == 4
    for r in kept:
        # strict > hi = k*16, so k+1 blocks (covers (k+1)*16-1) is the minimum
        assert len(r.blocks) * BLOCK_TOKENS > r.seq_len - 1
        assert len(r.blocks) == k + 1
    assert e._kv.free_blocks == free0 - 4
    for r in rows:
        e._release(r)
    e.shutdown()


def test_q2_verifier_tail_row_is_grown_after_a_two_token_accept():
    """q=2 verifier-tail. Pre-tick seq_len was k*16; verify accepts BOTH the
    anchor and the draft (run length 2), so post-commit seq_len = k*16+2 and the
    draft's hi = k*16+1. Pre-fork growth for q=2 covers only index
    seq_len+q-2 = k*16, so k held blocks (cover k*16-1) are one short even off
    an exact-boundary commit. The planner, keyed on the POST-verify seq_len,
    must add one block; the old graph path did not."""
    e = _engine()
    k = 3
    r = _row(e, rid=1, seq_len=k * BLOCK_TOKENS + 2, held=k)
    free0 = e._kv.free_blocks
    kept = e.ensure_draft_write_blocks([r])
    assert kept == [r]
    assert len(r.blocks) == k + 1
    assert len(r.blocks) * BLOCK_TOKENS > r.seq_len - 1  # covers hi = k*16+1
    assert e._kv.free_blocks == free0 - 1
    e._release(r)
    e.shutdown()


def test_a_row_the_pool_cannot_fit_is_finished_alone_not_raised():
    """The shared helper keeps the pre-fork contract: a row whose next block the
    pool cannot supply fails by itself (_finish, pool_exhausted) and its held
    blocks/slot return to the pools; the other rows are kept and grown. No
    exception escapes to fail the whole tick."""
    e = _engine(num_blocks=8, slots=4, batch=4)
    k = 1
    # take the pool to exactly the blocks the three surviving rows need consumed,
    # leaving one block: row0 needs one (fits), rows1-3 get constructed tight.
    rows = [_row(e, rid=i + 1, seq_len=k * BLOCK_TOKENS + 1, held=k) for i in range(4)]
    # exhaust every free block AFTER construction so no growth can allocate
    spare = [e._kv.alloc_block() for _ in range(e._kv.free_blocks)]
    kept = e.ensure_draft_write_blocks(rows)
    # at least the rows with no free block are finished, and none raised
    assert len(kept) < 4
    for r in rows:
        if r in kept:
            assert len(r.blocks) * BLOCK_TOKENS > r.seq_len - 1
        else:
            assert r.failed
            assert r not in e._running
            assert r.req_id in e._failed
    # the exhausted rows returned their held blocks (k each) before dying
    for b in spare:
        e._kv.free_block(b)
    for r in kept:
        e._release(r)
    assert e._kv.free_blocks == e._kv.num_blocks
    e.shutdown()


def test_sparse_row_uses_reserved_draft_blocks_and_is_not_trunk_grown():
    """Sparse+spec rows draft into a separate DENSE pool reserved at admit; the
    helper must never allocate a trunk block for them, and must accept a row
    whose draft_blocks reservation covers the verifier tail (seq_len-1+width-1)."""
    e = _engine(num_blocks=64, slots=4, batch=4, sparse_k=2, kv_cold=1 << 30)
    assert e._sparse is not None
    k = 3
    r = _row(e, rid=1, seq_len=k * BLOCK_TOKENS + 1, held=k, sparse=True)
    # simulate admit's dense draft-pool reservation: enough for the verifier tail
    end = r.seq_len - 1 + e._width - 1
    need = end // BLOCK_TOKENS + 1
    r.draft_blocks = [e._draft.kv.alloc_block() for _ in range(need)]
    trunk_before = list(r.blocks)
    free0 = e._kv.free_blocks
    kept = e.ensure_draft_write_blocks([r])
    assert kept == [r]
    assert r.blocks == trunk_before  # no trunk growth for a sparse row
    assert e._kv.free_blocks == free0
    e._release(r)
    e.shutdown()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
