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




def test_graph_keys_covers_what_a_decode_tick_keys_on():
    """`graph_keys` is what `precapture` builds, so it must contain every key
    `_run_decode_graph` would look up — otherwise warming succeeds, reports N
    graphs, and a real request captures anyway (~14 s on the 27B).

    That is exactly what a generate-and-hope warmup did: chain width depends on
    the draft's confidence, so no number of generated tokens guarantees a width
    appears, and two were left uncaptured. Runs off CUDA because it checks keys,
    not captures; capture parity is the CUDA test above.
    """
    backend = get_backend()
    cfg = tiny()
    for max_batch in (1, 2, 4, 8):
        e = build_engine(cfg, build_random(cfg, seed=21), backend, num_blocks=16,
                         num_slots=max_batch + 1, max_batch=max_batch,
                         max_total_tokens=256)
        keys = e.graph_keys()
        for rows in range(1, max_batch + 1):
            assert (e._graph_bucket(rows), 1) in keys, (
                f"max_batch={max_batch}: a {rows}-row tick keys on "
                f"{(e._graph_bucket(rows), 1)}, which precapture would not build"
            )
        assert e._graph_bucket(max_batch) <= max_batch, "a bucket may not exceed max_batch"
