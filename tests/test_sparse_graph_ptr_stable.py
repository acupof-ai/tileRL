import numpy as np
import torch

from tilerl import sparse_engine
from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import SamplingParams
from tilerl.model import build_random
from tilerl.testing import RefBackend

# Static tensors a captured sparse decode graph reads: their storage must never
# be REPLACED after capture (only in-place zero_/copy_), or a CUDA-graph replay
# reads the baked-in old address after the allocator recycles it (V100 mixed
# prefill+decode ScatterGather OOB, 2026-09-25). CpuSparseGraph keeps the same
# reuse SparseForward, so the ptr invariant is enforceable without a card.
STATIC_ATTRS = [
    "own_table",
    "page_base",
    "cand_idx",
    "own_log",
    "own_valid",
    "n_cand",
    "win",
    "own_len_t",
    "s_l2p",
    "s_nsel",
    "s_phys",
    "s_chosen",
]


def _snapshot(g):
    sf = g.sf
    snap = {}
    for a in STATIC_ATTRS:
        t = getattr(sf, a, None)
        if isinstance(t, torch.Tensor):
            snap[a] = t.data_ptr()
    if sf.s_bounds is not None:
        snap["s_bounds"] = [t.data_ptr() for t in sf.s_bounds]
    return snap


def test_captured_sparse_graph_static_tensors_are_never_reallocated():
    # apply_window_pages mutates a module global in BOTH sparse_index and the
    # by-value copy in sparse_engine, shared by every test in this worker;
    # restore the 128-token default or later tests see window 1024.
    sparse_engine.apply_window_pages(1024)  # production WINDOW_PAGES=64
    try:
        _run()
    finally:
        sparse_engine.apply_window_pages(128)


def _run():
    cfg = tiny(16384)
    e = build_engine(
        cfg=cfg,
        model=build_random(cfg, seed=11),
        backend=RefBackend(),
        num_blocks=512,
        num_slots=4,
        max_batch=4,
        max_total_tokens=32768,
        max_num_batched_tokens=512,
        sparse_k=128,
        sparse_min_tokens=0,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        decode_graph=True,
    )
    try:
        pA = (np.arange(6000, dtype=np.int64) % 300) + 7
        idA = e.submit(pA, SamplingParams(temperature=0.0, max_new_tokens=400, seed=1))

        def step(rid, nmin, cap=20000):
            n = 0
            while n < cap:
                d = e.poll()
                if rid in d and len(d[rid]) >= nmin:
                    return
                e.step()
                n += 1

        step(idA, 1)
        for _ in range(5):
            e.step()
        # at least one captured graph now exists
        assert e._sparse.graphs, "no sparse graph captured"
        g = next(iter(e._sparse.graphs.values()))
        # The selection OUTPUTS the recorded forward reads must live in the
        # persistent, outside-capture staging slices, not tensors lazily
        # allocated inside capture (graph-private storage that replays read as
        # garbage after a mixed eager tick). The cached dict is repopulated
        # pointing at those slices on every forward.
        sf = g.sf
        for gg in sf._dnsel:
            assert sf._dnsel[gg].data_ptr() == sf.s_nsel[gg].data_ptr(), (
                f"group {gg} nsel cache is not the persistent s_nsel slice"
            )
            assert sf._dphys[gg].data_ptr() == sf.s_phys[gg].data_ptr()
            assert sf._dchosen[gg].data_ptr() == sf.s_chosen[gg].data_ptr()
        snap0 = _snapshot(g)
        # drive a second long prompt through mixed eager ticks while row A decodes
        pB = (np.arange(8000, 13215, dtype=np.int64) % 300) + 7
        idB = e.submit(pB, SamplingParams(temperature=0.0, max_new_tokens=1, seed=2))
        step(idB, 1)
        for _ in range(4):
            e.step()
        snap1 = _snapshot(g)
        moved = {k: (snap0[k], snap1[k]) for k in snap0 if snap0[k] != snap1[k]}
        assert not moved, f"captured-graph static storage reallocated after mixed ticks: {moved}"
    finally:
        e.shutdown()
