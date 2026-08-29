"""Eager vs captured decode parity (CUDA only).

The captured decode tick (_DecodeGraph in engine.py) must produce the same
token stream as the eager path on the same inputs: same weights, same prompt,
greedy sampling => identical tokens. Runs on the pod CUDA target; skips on
CPU/metal (no CUDA graphs there).

Run: TILERL_TARGET=cuda uv run pytest tests/test_decode_graph.py -v
"""

from __future__ import annotations

import os

# Hermetic default: auto maps to cpu on this Mac; the test skips off-CUDA.
os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
import torch

from tilerl.config import tiny
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import build_random
from tilerl_kernels.backend import get_backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph decode is CUDA-only")
def test_decode_graph_matches_eager():
    backend = get_backend()
    cfg = tiny()
    prompt = torch.randint(
        0, cfg.vocab_size, (16,), generator=torch.Generator().manual_seed(11)
    ).tolist()
    params = SamplingParams(temperature=0.0, max_new_tokens=6, seed=3)
    eager = build_engine(
        cfg,
        build_random(cfg, seed=7),
        backend,
        num_blocks=8,
        num_slots=2,
        decode_graph=False,
    )
    captured = build_engine(
        cfg,
        build_random(cfg, seed=7),
        backend,
        num_blocks=8,
        num_slots=2,
        decode_graph=True,
    )
    we = eager.submit(prompt, params)
    wc = captured.submit(prompt, params)
    for _ in range(64):
        eager.step()
        captured.step()
        pe, pc = eager.poll(), captured.poll()
        if we in pe or wc in pc:
            assert pe.get(we) == pc.get(wc), f"eager {pe.get(we)} vs captured {pc.get(wc)}"
            # A capture failure degrades to eager with a warning — that would
            # make the parity check vacuous. Require the graph to exist.
            assert captured._decode_graph_on and captured._decode_graphs.get(1) is not None, (
                "decode graph capture fell back to eager"
            )
            return
    raise AssertionError("requests did not finish")
