"""#805 parity probe hook gates (device harness for scripts/probe_sparse_graph_cmax_bucket.py).

Two regressions the first V100 window hit, both invisible to the synthetic
verdict-table checks because they live in the hooks that drive a real engine:

1. srow lineage asymmetry. SparseRuntime.build_rows rows (every eager tick,
   every prefill chunk, and the graph arm's refresh ticks) are dicts carrying
   only req_id; decode_rows rows additionally carry req=r but those never reach
   sf.rows. The forward hook wrote rw.get("req", rw) and then read .req_id on
   the fallback, so a build_rows-shaped row crashed with
       'dict' object has no attribute 'req_id'
   before any comparison ran (rc13 in BOTH arms; graph is merely spawned
   first). A decode_rows-shaped-only fixture passes the old code, which is
   exactly how this slipped through -- the gate must build the build_rows
   shape.

2. selection observer bias. Reading sf.selected(bi, g) at the forward boundary
   forces a device->host .tolist() on the graph arm (fill() clears the device
   cache before every replay) while the eager arm reads the host _chosen map:
   two mechanisms at two times, able to fabricate TOKEN_DIVERGE_INPUT_DIFF_H3.
   The hook must observe the selection at ONE point shared by both arms --
   AFTER SparseRuntime.finalize, via sf.selected_pages -- and never call
   sf.selected.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_sparse_graph_cmax_bucket.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("h2_probe_805", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Sf:
    """Minimal SparseForward stand-in. rows are srow dicts in the
    build_rows shape (req_id only, no req object) -- the shape that crashed."""

    def __init__(self, rows, device_select=False):
        self.rows = rows
        self.device_select = device_select
        self.n_groups = 1
        self.calls = []

    def selected(self, bi, g):  # must never be called from the hook
        self.calls.append(("selected", bi, g))
        raise AssertionError("hook must not read sf.selected (D2H observer bias)")

    def selected_pages(self, bi):
        self.calls.append(("selected_pages", bi))
        r = self.rows[bi]
        return set(r["own"]) | set(r.get("_chosen", ()))


class _Model:
    def __init__(self, out):
        self.out = out

    def forward(self, input_ids, positions, kv, backend, **kw):
        return self.out


class _Sparse:
    def __init__(self, sf):
        self.sf = sf
        self.order = []

    def finalize(self, sf, rows, hidden=None):
        self.order.append("finalize")
        return []

    def run_decode_graph(self, reqs, chains=None):
        return False


class _Engine:
    def __init__(self, sf, out, state_slot):
        self._sparse = _Sparse(sf)
        self._model = _Model(out)
        self._kv = _Kv(sf, state_slot)
        self._states = _States()
        self._commit = lambda req, toks, lps=None: None

    def step(self):
        pass


class _Kv:
    def __init__(self, sf, state_slot):
        self.sparse = sf
        self.state_slot = torch.tensor([state_slot], dtype=torch.long)
        self.seq_q_lens = torch.tensor([1], dtype=torch.long)


class _States:
    def __init__(self):
        # [slots, layers, H, D, D] shaped; sum/sumsq only need a tensor.
        self.states = torch.zeros(1, 1, 1, 4, 4)
        self.conv_windows = None
        self.win_parity = torch.zeros(1, dtype=torch.int32)


def _job():
    return {
        "rid": 7,
        "ticks": [],
        "commits": [],
        "prefill_logits": [],
        "prefill_boundary": None,
        "immutable": {},
        "cur": None,
        "path": "eager",
        "arm": "eager",
        "bucket": 512,
        "W": 1,
    }


def test_build_rows_shaped_srow_does_not_crash_forward_hook():
    """build_rows rows carry req_id but no req object. The prefill forward hook
    must hash last-position logits without touching a .req attribute."""
    probe = _load_probe()
    srow = {  # build_rows shape: keys copied from SparseRuntime.build_rows
        "req_id": 7,
        "own": [10, 11],
        "own_len": 200,
        "q_hi": 200,
        "tq": 1,
        "decoding": False,
        "cand": [0, 1],
        "force_window": 8,
        "resolve": lambda p: p,
        "reserved": set(),
    }
    sf = _Sf([srow], device_select=False)
    # The engine passes last_only=seq_q to model.forward, so for a prefill
    # chunk the RETURNED logits are sliced to [B, 1, V] (last valid position
    # per row), even though seq_q_lens is the full chunk width (512). The
    # hook must read out[bi, -1], not out[bi, seq_q_lens - 1] -> the latter
    # raised "IndexError: index 511 out of bounds for dimension 1 size 1"
    # on-card in window2 before any decode ran.
    out = torch.zeros(1, 1, 8)
    eng = _Engine(sf, out, state_slot=0)
    eng._kv.seq_q_lens = torch.tensor([512], dtype=torch.long)
    job = _job()
    probe._install_parity_hooks(eng, job)

    ids = torch.zeros(1, 512, dtype=torch.long)
    pos = torch.zeros(1, 512, dtype=torch.long)
    eng._model.forward(ids, pos, eng._kv, backend=None)  # must not raise

    assert len(job["prefill_logits"]) == 1
    rec = job["prefill_logits"][0]
    assert set(rec) == {"sha1", "argmax", "n"}
    assert rec["n"] == 8


def test_selection_observed_once_after_finalize_via_selected_pages():
    """Both arms' selection must be observed AFTER SparseRuntime.finalize and
    must come from sf.selected_pages (one mechanism, one time point), never
    sf.selected (per-group device .tolist read at the forward boundary, which
    fabricated H3). With empty own/candidate sets no K/V gather runs, so this
    isolates the observation mechanism itself."""
    probe = _load_probe()
    srow = {  # build_rows shape, empty geometry so _pages_fp is a no-op
        "req_id": 7,
        "own": [],
        "own_len": 176,
        "q_hi": 176,
        "tq": 1,
        "decoding": True,
        "cand": [],
        "force_window": 0,
        "resolve": lambda p: p,
        "reserved": set(),
    }
    sf = _Sf([srow], device_select=True)
    eng = _Engine(sf, torch.zeros(1, 1, 8), state_slot=0)
    sparse = eng._sparse

    def pages(bi):
        sparse.order.append("selected_pages")
        return set()

    sf.selected_pages = pages
    job = _job()
    probe._install_parity_hooks(eng, job)

    class R:
        req_id = 7
        phase = 2
        output = [100]
        seq_len = 176
        state_slot = 0

    eng._sparse.finalize(sf, [R()])  # installed wrapper

    calls = [c[0] for c in sf.calls]
    assert "selected" not in calls, "hook read sf.selected (D2H observer bias)"
    assert sparse.order == ["finalize", "selected_pages"], sparse.order
    assert job["ticks"][0]["selected_pages"] == []


def test_page_content_comparator_source_aware():
    """fixmisc req #2/#4 follow-up: earlier-page K/V is compared by stable
    logical content. blob/blob = exact byte hash; frame/frame = the 1e-6 f32
    sketch (page never demoted); mixed source or a miss is UNDECIDABLE (None),
    not equal and not a corruption verdict -- it routes to the byte-dump rerun.
    A global widened tolerance must never be used."""
    probe = _load_probe()
    blob1 = {"src": "blob", "k": "aa", "v": "bb"}
    blob2 = {"src": "blob", "k": "aa", "v": "bb"}
    blob_diff = {"src": "blob", "k": "xx", "v": "bb"}
    frame1 = {"src": "frame", "k": [1.0, 2.0], "v": [3.0, 4.0]}
    assert probe._page_rec_equal(blob1, blob2) is True
    assert probe._page_rec_equal(blob1, blob_diff) is False
    assert probe._page_rec_equal(frame1, dict(frame1)) is True
    assert probe._page_rec_equal(blob1, frame1) is None  # mixed basis
    assert probe._page_rec_equal({"src": "miss"}, blob1) is None
    assert probe._sel_fp_diff({"1": blob1}, {"1": blob2}) is None
    assert probe._sel_fp_diff({"1": blob1}, {"1": blob_diff}) == ("sel_fp", "1", "differ")
    assert probe._sel_fp_diff({"1": blob1}, {"1": frame1}) == ("sel_fp", "1", "undecidable")
    assert probe._sel_fp_diff({"1": blob1}, {"2": blob1})[2] == "missing"


def test_undecidable_input_is_harness_not_h3():
    """fixmisc 5775711783 blocker: at a token-divergent tick, if the page K/V
    cannot be compared on the same basis (one arm cold blob, other resident
    frame), the verdict must be a HARNESS UNDECIDABLE -- not H3 and not a BAD
    rc10 conclusion -- so it routes to the byte-dump rerun. Real inequality on
    the same basis is still H3; equal inputs is still H1."""
    probe = _load_probe()

    def cells(sel1, st=None):
        def tick():  # one tick, token 99 vs eager 11
            return {
                "out_before": 0,
                "seq_before": 100,
                "cmax": 512,
                "path": "graph",
                "device_select": True,
                "ids": [10],
                "pos": [99],
                "state": {"x": 1},
                "own": [5, 6],
                "selected_pages": [1, 5, 6],
                "own_fp": {},
            }

        tg, te = tick(), tick()
        tg["tokens"] = [99]
        te["tokens"] = [11]
        commit_g = {"out_before": 0, "seq_before": 100, "toks": [99]}
        commit_e = {"out_before": 0, "seq_before": 100, "toks": [11]}

        def mk(tick, commit, graph):
            c = {
                "bucket": 512,
                "W": 1,
                "n_tokens": 8311,
                "n_graph": 1,
                "prefill_logits": {"sha1": "a" * 40, "n": 8},
                "prefill_own_fp": {},
                "ticks": [tick],
                "commits": [commit],
                "head": [commit["toks"][0]],
            }
            c["immutable"] = {"1": sel1} if graph else {"1": sel1}
            return [c]

        return mk(tg, commit_g, True), mk(te, commit_e, False)

    # mixed basis at the diverging tick -> UNDECIDABLE harness, never H3
    g, e = cells({"src": "blob", "k": "x", "v": "y"})
    e[0]["immutable"]["1"] = {"src": "frame", "k": [9.0], "v": [9.0]}
    seq = iter([{"arm": "graph", "W": 1, "cells": g}, {"arm": "eager", "W": 1, "cells": e}])
    probe.spawn_worker = lambda *a: next(seq)
    r = probe.compare_parity("s", "d", 0)
    assert r["verdict"] == "PROBE", r
    cell = r["rows"][0]
    assert cell["verdict"] == "TOKEN_DIVERGE_INPUT_UNDECIDABLE", cell["verdict"]
    assert "H3" not in cell["verdict"]
    assert cell["first"]["input_diff"][0] == "undecidable"
    assert cell["first"]["dump_cell"] == "512:1"

    # same basis, real inequality -> H3 / BAD
    g2, e2 = cells({"src": "blob", "k": "x", "v": "y"})
    e2[0]["immutable"]["1"] = {"src": "blob", "k": "different", "v": "y"}
    seq = iter([{"arm": "graph", "W": 1, "cells": g2}, {"arm": "eager", "W": 1, "cells": e2}])
    probe.spawn_worker = lambda *a: next(seq)
    r2 = probe.compare_parity("s", "d", 0)
    assert r2["verdict"] == "BAD", r2
    assert r2["rows"][0]["verdict"].startswith("TOKEN_DIVERGE_INPUT_DIFF_H3")


def test_cell_must_cross_cmax_bucket_boundary():
    """fixmisc 5774812215: a cell is valid only if the decode actually crossed
    a cmax doubling boundary (lazy recapture + steady replay after), not merely
    produced >=N tokens. prime lands cmax exactly on the bucket (n%16=7); the
    next page completes ~9 tokens in and doubles it. A run that ends still in
    the prime bucket must be rejected even if it emitted 8+ tokens — that was
    the window3 length-gate hole."""
    probe = _load_probe()

    def ticks_with(*cmaxs):
        return [{"cmax": c} for c in cmaxs]

    # prime bucket 512; cmax 512 sits inside 512 (cmax_bucket(512)=512)
    assert probe._cell_crossed_bucket(ticks_with(512, 512), 512) is False
    # 8-token run entirely below the next bucket (cmax_bucket(513)=1024 is the
    # crossing); 512..512 never crosses -> the hole the length>=8 gate missed
    assert probe._cell_crossed_bucket(ticks_with(*([512] * 8)), 512) is False
    # crossing observed at tick 9 (513 -> bucket 1024), then steady
    seq = [512] * 8 + [513] + [520] * 38
    assert probe._cell_crossed_bucket(ticks_with(*seq), 512) is True
    # cross early then cmax relaxes back: still crossed (any-tick semantics)
    assert probe._cell_crossed_bucket(ticks_with(512, 513, 512), 512) is True


def test_product_forward_exception_propagates_untagged(monkeypatch, capsys):
    """The other branch of fixmisc's fork: a PRODUCT forward exception
    (tilelang/attention/write_tokens) must bypass the probe observability try
    entirely. Three assertions, each of which goes red if orig_fwd is moved
    inside the try:
      * the original product exception type propagates;
      * _probe_die is NEVER called (structural: outside the try);
      * no [PROBE-EXC] is printed (worker tagging is probe-frame only)."""
    probe = _load_probe()
    calls = []
    monkeypatch.setattr(probe, "_probe_die", lambda where: calls.append(where))
    srow = {
        "req_id": 7,
        "own": [10, 11],
        "own_len": 200,
        "q_hi": 200,
        "tq": 1,
        "decoding": False,
        "cand": [0, 1],
        "force_window": 8,
        "resolve": lambda p: p,
        "reserved": set(),
    }
    sf = _Sf([srow], device_select=False)

    class ProductKernelError(RuntimeError):
        pass

    class FailingModel:
        def forward(self, input_ids, positions, kv, backend, **kw):
            raise ProductKernelError("simulated tilelang/attention failure")

    eng = _Engine(sf, torch.zeros(1, 1, 8), state_slot=0)
    eng._model = FailingModel()  # BEFORE hooks so fwd wraps the failing forward
    job = _job()
    probe._install_parity_hooks(eng, job)

    ids = torch.zeros(1, 512, dtype=torch.long)
    pos = torch.zeros(1, 512, dtype=torch.long)
    with pytest.raises(ProductKernelError):
        eng._model.forward(ids, pos, eng._kv, backend=None)
    assert calls == [], f"product exception routed through _probe_die: {calls}"
    assert "[PROBE-EXC]" not in capsys.readouterr().err
    assert job["prefill_logits"] == []


def test_probe_observation_error_is_tagged_exit13_in_worker(monkeypatch, capsys):
    """Worker-branch positive control: when the PROBE's own observation code
    fails (here a malformed srow missing a key), _probe_die prints [PROBE-EXC]
    and hard-exits 13. This is the harness half of the mechanical fork.
    os._exit cannot be caught, so stub it to raise SystemExit in-test."""
    probe = _load_probe()
    monkeypatch.setattr(probe.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    probe._IN_WORKER = True
    try:
        # entry loop subscripts rw0["req_id"] for EVERY row, so a malformed
        # srow lacking it raises inside the probe observation try.
        sf = _Sf([{}], device_select=False)
        eng = _Engine(sf, torch.zeros(1, 1, 8), state_slot=0)
        job = _job()
        probe._install_parity_hooks(eng, job)
        ids = torch.zeros(1, 1, dtype=torch.long)
        with pytest.raises(SystemExit) as ei:
            eng._model.forward(ids, ids, eng._kv, backend=None)
        assert ei.value.code == 13
        assert "[PROBE-EXC]" in capsys.readouterr().err
    finally:
        probe._IN_WORKER = False
