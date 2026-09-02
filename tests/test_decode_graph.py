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
