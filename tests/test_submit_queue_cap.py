"""Finding 17: submit() must bound the waiting queue with a typed overload.

Today ``submit`` appends to ``_waiting`` unconditionally, so an unbounded client
burst grows per-request host RAM (each waiting row pins its full prompt token
list) and hides head-of-line latency until the caller's own deadline. The bound
is on TOTAL in-flight rows (running + waiting), keyed to the state-slot
capacity the rows actually need: one running wave plus ONE queued wave is
enough to keep every slot saturated, since a freed slot is refilled from the
queue in the same step. A second queued wave cannot raise throughput.

Over-capacity submission raises ``EngineOverloaded`` synchronously and never
touches the queue. It is a RuntimeError so an existing 503 mapping contains it;
the routes later narrow it to 429.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import EngineOverloaded, SamplingParams
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import build_random

_AUTO = object()  # use build_engine's default (two-wave derived cap)


def _engine(num_slots=4, batch=4, max_inflight=_AUTO, num_blocks=128):
    kw = dict(
        cfg=tiny(), model=build_random(tiny(), seed=7), backend=get_backend(),
        num_blocks=num_blocks, num_slots=num_slots, max_batch=batch,
        max_total_tokens=4096, max_num_batched_tokens=512,
        prefix_store=NoPrefixStore(), sparse_k=0,
    )
    if max_inflight is not _AUTO:
        kw["max_inflight"] = max_inflight
    return build_engine(**kw)


_PARAMS = SamplingParams(temperature=0.0, max_new_tokens=2, seed=0)
_PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]


def test_the_overload_type_is_a_runtime_error():
    """A RuntimeError subclass is caught by the existing 503 mapping until the
    routes narrow it to 429."""
    assert issubclass(EngineOverloaded, RuntimeError)


def test_submit_accepts_one_running_wave_plus_one_queued_wave():
    """usable_slots=4: up to 2*4=8 in flight (4 running-capacity + 4 queued) are
    accepted without admitting anything; the 9th raises and is not enqueued."""
    e = _engine(num_slots=4, batch=4)
    ids = [e.submit(_PROMPT, _PARAMS) for _ in range(8)]
    assert len(ids) == 8
    assert len(e._waiting) == 8 and len(e._running) == 0  # nothing admitted pre-step
    with pytest.raises(EngineOverloaded) as exc:
        e.submit(_PROMPT, _PARAMS)
    assert "in-flight" in str(exc.value)
    # refused request is not counted anywhere
    assert len(e._waiting) == 8
    stats = e.stats()
    assert stats["waiting"] == 8
    e.shutdown()


def test_a_finished_row_frees_capacity_for_a_new_submit():
    """The cap counts LIVE in-flight rows; draining a slot lets a refused caller
    immediately resubmit."""
    e = _engine(num_slots=2, batch=2, num_blocks=64)
    cap = e.limits.max_inflight
    assert cap == 4  # 2 * usable_slots
    for _ in range(cap):
        e.submit(_PROMPT, _PARAMS)
    with pytest.raises(EngineOverloaded):
        e.submit(_PROMPT, _PARAMS)
    for _ in range(256):
        done = e.poll()
        if len(done) >= 1:
            break
        e.step()
    # one row finished -> one slot of capacity returned -> a submit succeeds
    rid = e.submit(_PROMPT, _PARAMS)
    assert isinstance(rid, int)
    e.shutdown()


def test_zero_new_tokens_requests_do_not_consume_inflight_capacity():
    """An embedding/score request is finished synchronously in submit and never
    enters running/waiting, so it is not charged against the concurrency cap."""
    e = _engine(num_slots=2, batch=2, num_blocks=64)
    for _ in range(e.limits.max_inflight):
        e.submit(_PROMPT, _PARAMS)
    # the cap is full, but a max_new_tokens<=0 request still goes through
    rid = e.submit(_PROMPT, SamplingParams(max_new_tokens=0))
    assert rid in e._finished
    with pytest.raises(EngineOverloaded):
        e.submit(_PROMPT, _PARAMS)
    e.shutdown()


def test_none_max_inflight_is_unbounded_not_the_two_wave_default():
    """The two meanings must not merge: None = no cap at all (submit accepts
    beyond 2*slots), while build_engine's default derives 2*slots."""
    e = _engine(num_slots=2, batch=2, max_inflight=None, num_blocks=64)
    assert e.limits.max_inflight is None
    for _ in range(4 * 2 + 3):  # well past the two-wave default of 4
        e.submit(_PROMPT, _PARAMS)
    assert len(e._waiting) == 4 * 2 + 3
    e.shutdown()


def test_an_explicit_max_inflight_overrides_the_two_wave_default():
    e = _engine(num_slots=4, batch=4, max_inflight=5)
    assert e.limits.max_inflight == 5
    for _ in range(5):
        e.submit(_PROMPT, _PARAMS)
    with pytest.raises(EngineOverloaded):
        e.submit(_PROMPT, _PARAMS)
    e.shutdown()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
