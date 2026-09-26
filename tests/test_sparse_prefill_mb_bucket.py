"""The eager sparse prefill attention table width (Mb) must be one fixed cap per
(S, B), not the batch's data width: a data-sized width moves by a page with
every distinct prompt length and re-JITs paged_attention on the first request
of each length. See awb impl#E.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import numpy as np

from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import SamplingParams
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import build_random
from tilerl.sparse_engine import SparseForward
from tilerl.testing import RefBackend

# Two prompts whose FINAL prefill chunk lands in the same S=64 query bucket but
# occupies a different own-page count (16 tokens -> 1 page, 48 -> 3 pages).
PROMPT_A = np.arange(7, 7 + 2 * 64 + 16, dtype=np.int64)  # last chunk 16
PROMPT_B = np.arange(7, 7 + 2 * 64 + 48, dtype=np.int64)  # last chunk 48


def _engine():
    return build_engine(
        cfg=tiny(),
        model=build_random(tiny(), seed=11),
        backend=RefBackend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        max_num_batched_tokens=64,
        prefix_store=NoPrefixStore(),
        sparse_k=2,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
    )


def _drain(engine, prompt, n):
    widths = []
    orig = SparseForward.attention_args

    def rec(self, plane, q, h=None):
        table, sl = orig(self, plane, q, h)
        if plane == 0 and self.rows and not self.rows[0]["decoding"]:
            widths.append(table.shape[1])  # Mb on each prefill tick
        return table, sl

    SparseForward.attention_args = rec
    try:
        rid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=n, seed=0))
        for _ in range(512):
            done = engine.poll()
            if rid in done and len(done[rid]) >= n:
                return done[rid][:n], widths
            engine.step()
        raise TimeoutError("engine did not finish")
    finally:
        SparseForward.attention_args = orig


def test_eager_mb_is_one_fixed_cap_within_an_s_bucket():
    e = _engine()
    cap = e._sparse.tracker.eager_mb_cap
    # k_pages(2) + WINDOW_PAGES(8) + ceil(64/16)+1(=5)
    assert cap == 2 + 8 + 5
    _, w_a = _drain(e, PROMPT_A, 6)
    _, w_b = _drain(e, PROMPT_B, 6)
    e.shutdown()
    # The FINAL prefill tick (last prefill width recorded) is the same Mb for
    # both prompt lengths even though their own-page counts are 1 vs 3.
    assert w_a[-1] == cap and w_b[-1] == cap
    assert len({w_a[-1], w_b[-1]}) == 1


def test_uncapped_data_width_differs_and_padded_output_matches():
    # Negative control: with the cap removed the table takes its data width and
    # the two lengths produce a DIFFERENT Mb; this is the per-length compile the
    # cap removes. It must fail the one-Mb assertion.
    e0 = _engine()
    e0._sparse.tracker.eager_mb_cap = 0
    out_a0, wa0 = _drain(e0, PROMPT_A, 6)
    out_b0, wb0 = _drain(e0, PROMPT_B, 6)
    e0.shutdown()
    assert wa0[-1] != wb0[-1], (wa0, wb0)

    # And padding the table to the cap must not move a single output token.
    e1 = _engine()
    out_a1, _ = _drain(e1, PROMPT_A, 6)
    out_b1, _ = _drain(e1, PROMPT_B, 6)
    e1.shutdown()
    assert out_a1 == out_a0
    assert out_b1 == out_b0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
