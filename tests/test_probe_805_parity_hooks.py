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
        "imm_conflict": [],
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
