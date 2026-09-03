"""Eager vs captured decode parity (CUDA only).

The captured decode tick (_DecodeGraph in engine.py) must produce the same
token stream as the eager path on the same inputs: same weights, same prompt,
greedy sampling => identical tokens. The ``verify`` arm runs the same check on
a width-2 tick, where the fused GDN and paged-attention decode kernels
carry the whole chain in one graph. Runs on the pod CUDA target; skips on
CPU/metal (no CUDA graphs there).

Run: TILERL_TARGET=cuda uv run pytest tests/test_decode_graph.py -v
"""

from __future__ import annotations

import os

# Hermetic default: auto maps to cpu on this Mac; the test skips off-CUDA.
os.environ.setdefault("TILERL_TARGET", "cpu")

from dataclasses import replace

import pytest
import torch

from tilerl.config import tiny
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import build_random
from tilerl.spec import DraftHead
from tilerl_kernels.backend import get_backend


def _draft(cfg, trunk):
    seed = 21
    dcfg = replace(cfg, num_layers=1, full_attn_layers=(0,), fp4=False)
    params = {k: v for k, v in build_random(dcfg, seed=seed).params.items()
              if k.startswith("layers.")}
    gen = torch.Generator().manual_seed(seed)
    h = cfg.hidden_size
    params["fc"] = (torch.randn(h, 2 * h, generator=gen) * 0.02).to(torch.bfloat16)
    params["norm"] = torch.ones(h, dtype=torch.bfloat16)
    params["pre_fc_norm_hidden"] = torch.ones(h, dtype=torch.bfloat16)
    return DraftHead(trunk, params, num_layers=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph decode is CUDA-only")
@pytest.mark.parametrize("spec", [False, True], ids=["decode", "verify"])
def test_decode_graph_matches_eager(spec):
    backend = get_backend()
    cfg = tiny()
    prompt = torch.randint(
        0, cfg.vocab_size, (16,), generator=torch.Generator().manual_seed(11)
    ).tolist()
    params = SamplingParams(temperature=0.0, max_new_tokens=6, seed=3)

    def engine(decode_graph):
        model = build_random(cfg, seed=7)
        return build_engine(
            cfg, model, backend, num_blocks=8, num_slots=2, decode_graph=decode_graph,
            draft=_draft(cfg, model) if spec else None, spec_depth=1,
        )

    eager, captured = engine(False), engine(True)
    we = eager.submit(prompt, params)
    wc = captured.submit(prompt, params)
    for _ in range(64):
        eager.step()
        captured.step()
        pe, pc = eager.poll(), captured.poll()
        if we in pe or wc in pc:
            assert pe.get(we) == pc.get(wc), f"eager {pe.get(we)} vs captured {pc.get(wc)}"
            # A capture failure degrades to eager with a warning — that would
            # make the parity check vacuous. Require the graph to exist, and at
            # the verify width require the wide one, not just the W=1 fallback.
            widths = {w for _, w in captured._decode_graphs}
            assert captured._decode_graph_on and widths, "decode graph capture fell back to eager"
            assert not spec or max(widths) > 1, f"no verify-width graph captured: {widths}"
            return
    raise AssertionError("requests did not finish")


def test_the_graphs_padding_row_is_not_taken_from_the_callers_capacity():
    """``num_slots=N`` must serve N concurrent requests with the graph on.

    A replay's padding rows write to the state and KV pools, so they need a slot
    and a block. Taking those from the pools the caller sized left N slots
    serving N-1, and the N-th ``submit`` raised from ``alloc_slot`` with no
    fallback. Runs on any target: what is gated is the reservation and the
    accounting, not the capture, which only CUDA does.
    """
    cfg, backend = tiny(), get_backend()
    n = 3

    def engine(decode_graph):
        return build_engine(cfg, build_random(cfg, seed=7), backend, num_blocks=16,
                            num_slots=n, max_batch=n, decode_graph=decode_graph)

    on, off = engine(True), engine(False)
    # The pad row is engine overhead: the pool grows by it, the reported capacity does not.
    assert on._states.num_slots == n + 1 and on._kv.num_blocks == 16 + 1
    assert on._pad_slot is not None and on._pad_block is not None
    assert on.stats()["slots_total"] == n and on.stats()["blocks_total"] == 16
    # Negative control: with the graph off nothing is reserved and nothing is added.
    assert off._states.num_slots == n and off._kv.num_blocks == 16
    assert off._pad_slot is None and off.stats()["slots_total"] == n

    prompt = torch.randint(0, cfg.vocab_size, (8,),
                           generator=torch.Generator().manual_seed(5)).tolist()
    params = SamplingParams(temperature=0.0, max_new_tokens=2, seed=0)
    ids = [on.submit(prompt, params) for _ in range(n)]  # the N-th used to raise
    assert len(set(ids)) == n and on.stats()["slots_used"] == n


def test_the_kv_guard_measures_usable_capacity_not_the_pool():
    """``submit``'s KV guard must compare against capacity net of the pad row.

    The pools are sized one larger when the captured tick is on, so a guard
    reading ``self._kv.num_blocks`` admits the one request sized to the whole
    pool and then fails on the allocation behind it — the same shape as the pad
    row itself, one level down.
    """
    cfg, backend = tiny(), get_backend()
    nb = 16  # 16 blocks x BLOCK_TOKENS 16 = 256 tokens usable, 272 gross

    def engine(decode_graph):
        return build_engine(cfg, build_random(cfg, seed=7), backend, num_blocks=nb,
                            num_slots=2, max_batch=2, max_total_tokens=512,
                            decode_graph=decode_graph)

    on, off = engine(True), engine(False)
    assert on.usable_blocks == nb and on._kv.num_blocks == nb + 1
    assert off.usable_blocks == nb and off._kv.num_blocks == nb

    # 260 + spec_depth needs 17 blocks: over the 16 usable, inside the 17 gross.
    big = torch.randint(0, cfg.vocab_size, (260,),
                        generator=torch.Generator().manual_seed(2)).tolist()
    params = SamplingParams(temperature=0.0, max_new_tokens=1, seed=0)
    for eng in (on, off):  # graph off is the control: same rejection, no pad row
        with pytest.raises(ValueError, match="exceeds KV pool capacity"):
            eng.submit(big, params)
