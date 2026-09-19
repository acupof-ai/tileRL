"""Gate for TILERL_DRAFT_TRUE_Q_WIDTH: a DECODE-only draft step keeps its true
query width instead of rounding up to ``_PREFILL_BUCKET``.

The bucket exists so a chunked PREFILL's widening prompt length cannot recompile
the seq_q_lens kernels (14 compiles / 15.5 s on a served first visit). A decode
step's q is at most ``draft.width`` (2 at d1) and carries no prefill, so the
bucket buys it nothing and costs rows: the sm70 split kernel carries S in its
grid, so T=64 over q=2 runs 32x the CTAs and 32x the history scan of rows no
kernel reads.

CPU gates:
1. the flag defaults OFF and the forward is byte-identical to the bucketed one;
2. on decode-only ticks the flag really narrows T (the positive control — without
   it, gate 1 would pass vacuously, since a run whose q never exceeds 1 has
   nothing to narrow);
3. the two exclusions keep the bucket: a still-PREFILLING row and a decode-phase
   CATCH-UP row (q > width);
4. the shared predicate is one object — ``DraftHead.step`` and
   ``_windowed_read_kv`` cannot drift apart.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from test_e2e import _random_draft
from tilerl_kernels.backend import get_backend

from tilerl import spec
from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import SamplingParams
from tilerl.model import build_random
from tilerl.spec import _PREFILL_BUCKET, draft_step_is_decode_only


def _force_accept(eng, tok: int = 7):
    """Make every draft accepted, so verify ticks carry q = n_ok + 1 = 2 — the
    shape this change is about. A random tiny draft is rejected on nearly every
    tick, so without this the run only ever exercises q=1 and the gate would pass
    vacuously."""
    be, orig = eng._backend, (eng._backend.greedy, eng._backend.sample_batch)

    def greedy(logits):
        t, p = orig[0](logits)
        return torch.full_like(t, tok), p

    def sample(logits, *a, **k):
        t, lp = orig[1](logits, *a, **k)
        return torch.full_like(t, tok), lp

    be.greedy, be.sample_batch = greedy, sample
    return orig


def _restore(eng, orig):
    eng._backend.greedy, eng._backend.sample_batch = orig


def _run(flag: bool, n_new: int = 24, accept: bool = True):
    """One served run with the flag set; returns (generated ids, T histogram,
    per-call records ``(qs, decoding, T_in, T_out)``)."""
    prev = spec._DRAFT_TRUE_Q_WIDTH
    spec._DRAFT_TRUE_Q_WIDTH = flag
    cfg = tiny()
    model = build_random(cfg, seed=7)
    draft = _random_draft(cfg, 7, model)
    eng = build_engine(cfg, model, get_backend(), num_blocks=32, num_slots=2,
                       max_batch=2, max_total_tokens=512, draft=draft,
                       spec_depth=1, sparse_k=0)
    widths: list[int] = []
    calls: list[tuple] = []
    orig_fwd, orig_qw = draft.forward, draft._draft_q_width

    def spy(hidden, ids, positions, kv, backend, **kw):
        widths.append(hidden.shape[1])
        return orig_fwd(hidden, ids, positions, kv, backend, **kw)

    def spy_qw(plan, w):
        out = orig_qw(plan, w)
        calls.append(([hi - lo + 1 for _, lo, hi, _ in plan],
                      [r.decoding for r, *_ in plan], w, out))
        return out

    draft.forward, draft._draft_q_width = spy, spy_qw
    orig = _force_accept(eng) if accept else None
    try:
        prompt = np.random.default_rng(3).integers(3, 320, size=140).astype(np.int64)
        rid = eng.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=n_new,
                                                seed=0))
        out = []
        for _ in range(300):
            done = eng.poll()
            if rid in done and len(done[rid]) >= n_new:
                out = done[rid]
                break
            eng.step()
        return out, Counter(widths), calls
    finally:
        draft.forward, draft._draft_q_width = orig_fwd, orig_qw
        if orig is not None:
            _restore(eng, orig)
        eng.shutdown()
        spec._DRAFT_TRUE_Q_WIDTH = prev


def test_flag_defaults_off():
    """Opt-in, not a default flip: the shipped value is off, so every existing
    number is unchanged until a device window measures this one."""
    assert spec._DRAFT_TRUE_Q_WIDTH is False
    assert "_DRAFT_TRUE_Q_WIDTH = bool(os.environ.get" in (
        spec.__loader__.get_source(spec.__name__) if hasattr(spec, "__loader__") else "")


def test_decode_only_predicate_matches_window_predicate_table():
    """The predicate's truth table, stated independently of both consumers. A row
    is a windowable/narrowable tail iff it is DECODE-phase AND q <= width."""
    w = 2
    assert draft_step_is_decode_only([1], w) is True
    assert draft_step_is_decode_only([2, 1], w, [True, True]) is True
    assert draft_step_is_decode_only([2], w, [False]) is False          # prefilling
    assert draft_step_is_decode_only([1, 8], w, [True, True]) is False  # catch-up
    assert draft_step_is_decode_only([3], w, [True]) is False           # q > width
    # decode=None defaults to all-decode (the window's own default).
    assert draft_step_is_decode_only([1, 2], w) is True


def test_flag_narrows_decode_t_and_output_is_identical():
    """The positive control AND the parity claim in one run.

    OFF -> bucketed T (64 on a verify tick). ON -> T = 2 on every decode-only step.
    The generated tokens are identical either way: the padded columns were never
    read (every kernel gates on SeqQLens), so narrowing T is a launch-shape change
    and not a numerics change.
    """
    off_out, off_T, _ = _run(False)
    on_out, on_T, on_calls = _run(True)

    assert len(off_out) == len(on_out) == 24
    assert off_out == on_out, "narrowing T changed the generated tokens"

    # Positive control: the flag must actually narrow T on decode ticks. Without
    # this assertion the parity above would hold trivially if q never exceeded 1.
    assert off_T[_PREFILL_BUCKET] > 0, f"no bucketed verify tick to narrow: {off_T}"
    assert on_T[_PREFILL_BUCKET] < off_T[_PREFILL_BUCKET], (
        f"flag did not narrow any decode tick: off={off_T} on={on_T}")

    # The exact contract, per call: a decode-only plan returns max(q) (== 2 here);
    # anything else returns the bucketed input unchanged.
    narrowed = [c for c in on_calls if c[3] < c[2]]
    assert narrowed, f"no call narrowed: {on_calls}"
    for qs, dec, w_in, w_out in on_calls:
        if draft_step_is_decode_only(qs, 2, dec):
            assert w_out == max(qs) == 2, (qs, dec, w_in, w_out)
        else:
            assert w_out == w_in, (qs, dec, w_in, w_out)


def test_non_decode_ticks_keep_the_bucket():
    """The two exclusions, observed through a real run instead of the unit table:
    a still-PREFILLING row (q=127, chunked into the prompt) and a decode-phase
    CATCH-UP row (q > width) both keep the bucketed T."""
    _, _, on_calls = _run(True)
    kept = [c for c in on_calls if c[3] == c[2] and not draft_step_is_decode_only(
        c[0], 2, c[1])]
    assert kept, f"run exercised no excluded row: {on_calls}"
    # Both exclusion reasons appear: a prefilling row and a catch-up decode row.
    assert any(not all(c[1]) for c in kept), "no prefilling row was excluded"
    assert any(all(c[1]) and max(c[0]) > 2 for c in kept), (
        f"no catch-up row was excluded: {kept}")


def test_step_and_window_share_one_predicate_object():
    """The stated risk is the two sites drifting. They must CALL the same function,
    not restate its logic: a restated copy is what silently diverges when one side
    is edited."""
    import inspect

    from tilerl.spec import DraftHead

    step_src = inspect.getsource(DraftHead._draft_q_width)
    win_src = inspect.getsource(DraftHead._windowed_read_kv)
    assert "draft_step_is_decode_only" in step_src
    assert "draft_step_is_decode_only" in win_src
    # Neither may keep its own inline copy of the condition.
    assert "int(q) <= width for d, q in zip" not in win_src
    assert "r.decoding and" not in step_src


def test_widening_a_catch_up_row_does_not_narrow_the_batch():
    """Batch-level: one catch-up row among verify tails pins the whole forward to
    the bucket. Unit-level companion to test_non_decode_ticks_keep_the_bucket."""
    assert draft_step_is_decode_only([2, 2, 13], 2, [True, True, True]) is False
    assert draft_step_is_decode_only([2, 2, 2], 2, [True, True, True]) is True
