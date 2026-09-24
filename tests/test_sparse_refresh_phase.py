"""A new request must not inherit the sparse refresh phase.

``SparseRuntime.ticks_since_refresh`` is engine-wide: initialised 0 once at
construction (``sparse_runtime.py:120``), advanced every decode tick, and reset
only when a refresh itself fires (``:307``). Nothing reset it at admission, so a
new request's eager-refresh ticks -- and therefore its served tokens -- depended
on the traffic that ran before it. Confirmed against the device:
``errors/2026-09-24-slot-output-cycle.md``.

Gate: the phase after a new call's FIRST decode tick is 1 (reset to 0 at
admission, then one increment), whatever phase the predecessor left behind. A
small sparse engine on the CPU is enough -- the quantity asserted is the counter
itself, not a token that depends on the model being sensitive to it.
"""

from __future__ import annotations

import numpy as np

from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import _PHASE_DECODE, SamplingParams
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.model import build_random
from tilerl.testing import RefBackend

# tiny's vocab is small, so ids must stay inside it.
_FIRST = (np.arange(8 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
_SECOND = (np.arange(5 * BLOCK_TOKENS, dtype=np.int64) % 280) + 9


def _engine():
    return build_engine(
        cfg=tiny(),
        model=build_random(tiny(), seed=11),
        backend=RefBackend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        prefix_store=NoPrefixStore(),
        sparse_k=2,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        # The phase only advances on the device-select decode path
        # (``sparse_runtime.py:301``); without this the counter stays 0 and the
        # assertion cannot tell a reset from no reset.
        sparse_device_select=True,
    )


def _drain(e, rid, n):
    for _ in range(512):
        done = e.poll()
        if rid in done and len(done[rid]) >= n:
            return done[rid][:n]
        e.step()
    raise AssertionError(f"rid {rid} produced < {n} tokens")


def _phase_after_first_decode_tick(e, prompt, n_tok):
    """Submit one call and return the refresh phase after its first PURE decode
    tick. The tick that first leaves a row in DECODE is still a prefill tick (it
    is the chunk that produced the first token), so the test steps until a decode
    row exists BEFORE the step -- that step is the one that advances the counter.
    """
    params = SamplingParams(temperature=0.0, max_new_tokens=n_tok, seed=0)
    rid = e.submit(prompt, params)
    for _ in range(512):
        decoding_already = any(r.phase == _PHASE_DECODE for r in e._running)
        e.step()
        if decoding_already:
            return e._sparse_ticks_since_refresh
        if rid in e.poll():
            break
    raise AssertionError("call never reached a pure decode tick")


def test_first_decode_tick_sees_a_zeroed_phase():
    e = _engine()
    try:
        # A predecessor long enough to leave a NON-zero phase behind. Without one
        # the test cannot discriminate: 0 is what the unfixed tree reports too.
        _drain(e, e.submit(_FIRST, SamplingParams(temperature=0.0,
                                                  max_new_tokens=6, seed=0)), 6)
        leftover = e._sparse_ticks_since_refresh
        assert leftover != 0, (
            f"predecessor left phase {leftover}; the test needs a non-zero leftover "
            "to tell a reset from no reset"
        )

        # The call under test: its first decode tick must start from the reset.
        assert _phase_after_first_decode_tick(e, _SECOND, 4) == 1
    finally:
        e.shutdown()
