"""Deterministic CPU gate for prefill-cancel preemption (#667).

UI symptom (H20): aborting a request BEFORE its first token closes the client
socket immediately, but ``slots_used`` stays full until the long prefill fill
runs to completion (~80 s for a 24k fill); decode-phase Stop cancels promptly.

Two seams behave differently, and the gate pins which one is red:

* BETWEEN ticks (chunked prefill) — green control: ``step()`` releases
  ``engine._lock`` after every forward, so a cancel issued in that gap removes
  the row before the next chunk runs. It must NOT run the remaining chunks.
* INSIDE one long forward — the red seam: a fill planned as a single
  ``model.forward`` holds ``engine._lock`` for its whole duration. An abort's
  ``engine.cancel()`` (a worker thread, exactly like the HTTP ``to_thread``
  cancel) blocks acquiring that lock until the forward returns, so the slot is
  held for the whole fill. Marked ``xfail(strict=True)``: an intra-fill
  preemption fix makes it xpass, which fails and asks to remove the marker.

Both drive the real tiny engine; only ``model.forward`` is instrumented.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import threading

import pytest
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import SamplingParams
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import build_random

_PHASE_PREFILL = 1


def _engine(max_num_batched_tokens: int, num_blocks: int, prompt_len: int,
            sparse: bool = False):
    cfg = tiny()
    eng = build_engine(
        cfg=cfg, model=build_random(cfg, seed=0), backend=get_backend(),
        num_blocks=num_blocks, num_slots=4, max_batch=4,
        max_total_tokens=prompt_len + 256,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_store=NoPrefixStore(), decode_graph=False,
        **(dict(sparse_k=16, scorer="bounds", kv_cold_bytes=1 << 30,
                sparse_min_tokens=8192, sparse_prefill_tokens=128) if sparse else {}),
    )
    rid = eng.submit(list(range(1, prompt_len + 1)),
                     SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    return eng, rid


def _prefill_forwards(eng, rid, lock, count):
    """Wrap model.forward to count PREFILL forwards (records the planned width)."""
    real = eng._model.forward

    def counting(input_ids, positions, *a, **k):
        req = next((r for r in eng._running if r.req_id == rid), None)
        if req is not None and req.phase == _PHASE_PREFILL:
            with lock:
                count["n"] += 1
                count["max_width"] = max(count["max_width"], input_ids.shape[1])
        return real(input_ids, positions, *a, **k)

    eng._model.forward = counting


def test_chunked_prefill_cancel_stops_before_the_remaining_chunks():
    """GREEN CONTROL: cancel in the lock-free gap after chunk 5 frees the row.

    24k at the served 512-token budget is ~47 prefill forwards. The cancel is
    issued from the main thread BETWEEN manual steps (where step() has dropped
    the lock), exactly the window a daemon loop exposes each tick. The row must
    be gone and must not run the other ~42 chunks.
    """
    eng, rid = _engine(max_num_batched_tokens=512, num_blocks=8192, prompt_len=24_000)
    lock = threading.Lock()
    count = {"n": 0, "max_width": 0}
    _prefill_forwards(eng, rid, lock, count)

    ran_after_cancel = 0
    for tick in range(60):
        eng.step()
        with lock:
            n = count["n"]
        gone = not any(r.req_id == rid for r in eng._running)
        if n == 5 and not gone:
            # The real abort arrives on a to_thread worker; issue it here in the
            # between-tick gap and let it complete before the next step.
            assert eng.cancel(rid) is True
        if n > 5:
            ran_after_cancel += 1
        if gone:
            break

    try:
        assert not any(r.req_id == rid for r in eng._running), "row still running"
        assert eng.stats()["slots_used"] == 0, "slot not released"
        assert rid in eng._failed, "cancelled row must be recorded failed"
        assert ran_after_cancel == 0, f"{ran_after_cancel} chunks ran after cancel"
    finally:
        eng.shutdown()


def test_chunked_sparse_prefill_cancel_stops_between_capped_ticks():
    """Sparse-hybrid rows (capped at sparse_prefill_cap) preempt between ticks too."""
    prompt = 9000  # > sparse_min_tokens 8192 -> sparse_on
    eng, rid = _engine(max_num_batched_tokens=8192, num_blocks=2048,
                       prompt_len=prompt, sparse=True)
    lock = threading.Lock()
    count = {"n": 0, "max_width": 0}
    _prefill_forwards(eng, rid, lock, count)

    ran_after_cancel = 0
    for tick in range(200):
        eng.step()
        with lock:
            n = count["n"]
        gone = not any(r.req_id == rid for r in eng._running)
        if n == 3 and not gone:
            assert eng.cancel(rid) is True
        if n > 3:
            ran_after_cancel += 1
        if gone:
            break

    try:
        assert not any(r.req_id == rid for r in eng._running)
        assert eng.stats()["slots_used"] == 0
        assert rid in eng._failed
        assert count["max_width"] <= 128, "sparse prefill did not run capped"
        assert ran_after_cancel == 0, f"{ran_after_cancel} sparse chunks ran after cancel"
    finally:
        eng.shutdown()


@pytest.mark.xfail(strict=True, reason="#667: a fill whose budget>=prompt runs as ONE "
                                      "model.forward holding engine._lock; cancel blocks "
                                      "until it returns. The served 24k row is NOT this "
                                      "shape (47x512 dense / 125x192 sparse, multi-tick); "
                                      "this only fires if such a one-forward fill exists.")
def test_cancel_preempts_a_single_long_prefill_forward():
    """RED SEAM: a one-forward fill must become cancellable before it returns.

    The forward is held open (a device prefill kernel can run tens of seconds);
    the abort's engine.cancel() races on a worker thread, as the HTTP
    to_thread cancel does. Today it blocks on engine._lock (held by step across
    the forward), so mid-forward the row is still running and the slot still
    occupied — the UI ~80 s hold. An intra-fill cancel-checkpoint fix makes the
    row disappear here (xpass -> strict xfail fails -> remove the marker).
    """
    # Budget covering the prompt plans the whole fill as ONE prefill forward.
    eng, rid = _engine(max_num_batched_tokens=4096, num_blocks=512, prompt_len=2048)
    observed = {}
    lock = threading.Lock()
    proceed = threading.Event()
    real = eng._model.forward

    def one_forward(input_ids, positions, *a, **k):
        req = next((r for r in eng._running if r.req_id == rid), None)
        if req is not None and req.phase == _PHASE_PREFILL and req.prefill_from == 0:
            with lock:
                observed["n"] = observed.get("n", 0) + 1
            t = threading.Thread(target=lambda: eng.cancel(rid), daemon=True)
            t.start()
            t.join(timeout=0.5)  # a preempting cancel returns well inside this
            observed["cancel_returned"] = not t.is_alive()
            observed["row_gone"] = not any(r.req_id == rid for r in eng._running)
            observed["slots"] = eng._slots_used
            proceed.wait(2.0)  # hold the lock: emulate the uninterruptible fill
        return real(input_ids, positions, *a, **k)

    eng._model.forward = one_forward
    eng.step()
    try:
        assert observed.get("n") == 1, "setup: fill must be one forward"
        assert observed["cancel_returned"] is True, "cancel blocked until the forward ended"
        assert observed["row_gone"] is True, "row still running inside its own fill"
        assert observed["slots"] == 0, "slot not released inside the cancelled fill"
    finally:
        proceed.set()
        eng.shutdown()
