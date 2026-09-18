"""Gates for the env-gated _StepTiming probe (PR #639 review).

Covers the three things the probe can silently get wrong:

* default OFF: an Engine built with TILERL_STEP_TIMING unset carries no timer,
  so the hot path makes zero extra perf_counter calls in production;
* ON: a real CPU-tiny run produces the documented segment structure, the
  "forward" envelope reconciles with its inner segments and the segments
  reconcile with the tick total, and the atexit average report does not crash;
* source: every ADDED perf_counter read (the `_t = time.perf_counter()` shape)
  sits under an `if _tm is not None` guard, so enabling is the only way to pay
  for them. The timer's own reads live inside the _StepTiming class.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("TILERL_TARGET", "cpu")

import numpy as np
from test_e2e import build_serving_engine

from tilerl import engine as engine_mod
from tilerl.engine import SamplingParams
from tilerl.kv_cache import BLOCK_TOKENS

#: The three release sub-segments every request end must charge.
_RELEASE_SEGMENTS = ("release_close_request", "release_cold_forget", "release_blocks")

#: The per-page publish costs, split by fix (see transfer_to_shared). Each has
#: a different remedy, which is why one close_request bucket is not enough.
_PUBLISH_SEGMENTS = ("pub_bounds_d2h", "pub_draft_clone", "pub_ssd_transfer",
                     "pub_frame_d2h", "pub_share_hold")

#: Every segment name the probe may emit. "forward" is an envelope (see its
#: docstring); graph ticks carry "graph" instead of the eager inner set.
_SEGMENTS = {
    "plan", "stats", "forward", "charge", "graph",
    "sparse_select", "prep", "model", "sparse_finalize", "sample",
    "draft_blocks", "draft_step", "offers_pub",
    "release_close_request", "release_cold_forget", "release_blocks",
    "pub_bounds_d2h", "pub_draft_clone", "pub_ssd_transfer", "pub_frame_d2h",
    "pub_share_hold",
}
_INNER = {
    "sparse_select", "prep", "model", "sparse_finalize", "sample",
    "draft_blocks", "draft_step", "offers_pub",
}

_ENGINE_PY = Path(engine_mod.__file__)


def test_timer_absent_without_env(monkeypatch):
    monkeypatch.delenv("TILERL_STEP_TIMING", raising=False)
    eng = build_serving_engine(seed=1)
    try:
        assert eng._step_timing is None
    finally:
        eng.shutdown()


def test_timing_on_segments_reconcile(monkeypatch, capsys):
    monkeypatch.setenv("TILERL_STEP_TIMING", "1")
    monkeypatch.setenv("TILERL_STEP_TIMING_SLOW_MS", "0")  # print path exercised too
    eng = build_serving_engine(seed=1)
    try:
        tm = eng._step_timing
        assert tm is not None
        prompt = np.random.default_rng(0).integers(3, 320, size=16).astype(np.int64)
        rid = eng.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=12))

        rows: list[tuple[float, dict[str, float]]] = []
        for _ in range(64):
            done = eng.poll()
            if rid in done and len(done[rid]) >= 12:
                break
            eng.step()
            rows.append((tm.last_total, dict(tm.cur)))
        else:
            raise AssertionError("request did not finish")

        assert rows, "no tick ran"
        eager = [r for r in rows if "model" in r[1]]
        assert eager, "no eager (model-bearing) tick observed"
        for total, cur in rows:
            assert set(cur) <= _SEGMENTS, f"unknown segments: {set(cur) - _SEGMENTS}"
            assert total > 0 and cur.get("forward", 0.0) > 0.0
            # Top level: the tick total is plan + stats + forward + charge.
            top = sum(cur.get(k, 0.0) for k in ("plan", "stats", "forward", "charge"))
            assert abs(total - top) < 2e-3 * total + 1e-3, (total, cur)
        for total, cur in eager:
            # The forward envelope equals its inner segments (no hidden time).
            inner = sum(cur[k] for k in _INNER if k in cur)
            assert abs(cur["forward"] - inner) < 2e-3 * cur["forward"] + 1e-3, cur
        assert tm.n == len(rows)

        # Hollow-tick tail: the bracket wraps _run_forward. fwd_host_ms must track
        # the "forward" envelope (same region), not seconds-since-epoch — that fails
        # if fwd_start is dropped and fwd_t0 stays 0.0.
        last_fwd = rows[-1][1].get("forward", 0.0) * 1000
        assert last_fwd > 0.0
        assert tm.fwd_host_ms > 0.0 and abs(tm.fwd_host_ms - last_fwd) < 5.0, (
            tm.fwd_host_ms, last_fwd)
        assert tm.cuda is False and tm.fwd_gpu_ms is None
        assert tm.fwd_path == "eager"
        assert tm.last_why == "cpu"
        err = capsys.readouterr().err
        slow = [ln for ln in err.splitlines() if ln.startswith("[step-timing] tick ")]
        assert slow, "SLOW_MS=0 must print at least one slow tick line"
        assert "fwd_host=" in slow[-1] and "why=cpu" in slow[-1]
        assert "fwd_gpu=" not in slow[-1]  # device span omitted off CUDA
        tm.report()  # the atexit callback: must not crash with real data
    finally:
        eng.shutdown()


def test_release_subsegments_are_inside_sample(monkeypatch):
    """A request ending must charge its release to a named sub-segment, and the
    sub-segments must stay inside the tick they were charged in. Guards both
    halves of the split: the marks exist at all, and a mark left outside the
    tick's own accounting would show up as a sub-segment sum exceeding its
    parent tick. Containment is asserted against the TICK total, not against
    "sample": a prefill that ends also releases, and that path runs after the
    sample mark (see _finish_prefills)."""
    monkeypatch.setenv("TILERL_STEP_TIMING", "1")
    monkeypatch.setenv("TILERL_STEP_TIMING_SLOW_MS", "0")
    eng = build_serving_engine(seed=1)
    try:
        tm = eng._step_timing
        prompt = np.random.default_rng(0).integers(3, 320, size=16).astype(np.int64)
        rid = eng.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=12))
        ended: list[dict[str, float]] = []
        for _ in range(64):
            done = eng.poll()
            eng.step()
            # tick_start cleared cur, so this dict IS this tick's segments.
            if tm.cur.get("release_blocks", 0.0) > 0.0:
                ended.append(dict(tm.cur))
            if rid in done and len(done[rid]) >= 12:
                break
        else:
            raise AssertionError("request did not finish")
        assert ended, "no tick charged a release_blocks: the end-tick split is gone"
        for cur in ended:
            assert cur.get("release_blocks", 0.0) > 0.0
            assert all(cur.get(k, 0.0) >= 0.0 for k in _RELEASE_SEGMENTS)
            assert sum(cur.get(k, 0.0) for k in _RELEASE_SEGMENTS) <= tm.last_total + 1e-3, cur
    finally:
        eng.shutdown()


def test_release_subsegments_charge_on_a_sparse_request_end(monkeypatch):
    """Every release and publish sub-segment must charge on a real sparse end.

    The dense test above only ever exercises `release_blocks`: with no sparse
    row the `close_request`/`cold_forget` marks are never reached, so deleting
    either one left the gate green. Two shapes are needed beyond that:

    * a DRAFT row, or `pub_draft_clone` charges ~1 us of timer noise on the
      skipped `if draft_block is not None` branch and the assertion passes
      vacuously (measured 0.8 us without, 23.2 us with);
    * at least 12 prompt pages, or the private blob is still under the host
      budget and the SSD transfer branch never runs.
    """
    from test_sparse_engine import _sparse_engine

    monkeypatch.setenv("TILERL_STEP_TIMING", "1")
    monkeypatch.setenv("TILERL_STEP_TIMING_SLOW_MS", "0")
    eng = _sparse_engine(2, draft=True)
    try:
        tm = eng._step_timing
        prompt = (np.arange(16 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
        rid = eng.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
        peak: dict[str, float] = {}
        for _ in range(512):
            done = eng.poll()
            eng.step()
            # Per-tick segments (tick_start cleared cur): a mark charges the
            # release work of exactly the requests that ended in this tick.
            for k, v in tm.cur.items():
                peak[k] = peak.get(k, 0.0) + v
            if rid in done and len(done[rid]) >= 2:
                break
        else:
            raise AssertionError("sparse request did not finish")
        missing = [k for k in _RELEASE_SEGMENTS if peak.get(k, 0.0) <= 0.0]
        assert not missing, f"never charged on a sparse request end: {missing}"
        assert eng._sparse.prefix.published == 1, eng._sparse.prefix.published
        # Measured 2026-09-18, k=2, 16 pages, draft=True, 2 generated:
        # bounds 39.5us, draft_clone 64.1, frame_d2h 121.5, share_hold 14.6,
        # ssd_transfer 17.7.
        missing = [k for k in _PUBLISH_SEGMENTS if peak.get(k, 0.0) <= 0.0]
        assert not missing, f"publish sub-segment never charged: {missing}"
        assert set(peak) <= set(_SEGMENTS)
    finally:
        eng.shutdown()


def test_added_perf_counter_reads_are_guarded():
    """Every `_t = time.perf_counter()` outside _StepTiming is inside an
    `if _tm is not None:` block. The always-on pre-existing reads use other
    statement shapes (_hybrid_t0 / t_fwd / the hybrid accounting) and are not
    what this probe added."""
    tree = ast.parse(_ENGINE_PY.read_text())

    class _Finder(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[ast.AST] = []
            self.bad: list[int] = []
            self.in_timer_class = 0

        def visit_ClassDef(self, node):
            saved = self.in_timer_class
            if node.name == "_StepTiming":
                self.in_timer_class += 1
            self.generic_visit(node)
            self.in_timer_class = saved

        def visit_If(self, node):
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Assign(self, node):
            if (
                self.in_timer_class == 0
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_t"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "perf_counter"
            ):
                guarded = any(
                    isinstance(n, ast.If) and "_tm" in ast.unparse(n.test)
                    for n in self.stack
                )
                if not guarded:
                    self.bad.append(node.lineno)
            self.generic_visit(node)

    f = _Finder()
    f.visit(tree)
    assert not f.bad, f"unguarded perf_counter reads at lines {f.bad}"


def test_hollow_tick_probe_is_sync_free_and_uses_async_reads():
    """The hollow-tick tail must read the device span WITHOUT draining the queue:
    memory_stats is a host counter read and the end event is read with non-blocking
    query(); a synchronize() call would perturb the allocator reuse it measures.
    Source-only (the CUDA branches never execute under the CPU RefBackend), so
    assert on AST CALL nodes — substring presence would be satisfied by a comment
    and stay green with the real call deleted."""
    text = _ENGINE_PY.read_text()
    tree = ast.parse(text)
    timer = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "_StepTiming")

    calls: set[str] = set()

    def _chain(n: ast.AST) -> str:
        parts = []
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
        return ".".join(reversed(parts))

    class _Walk(ast.NodeVisitor):
        def visit_Call(self, node):
            calls.add(_chain(node.func))
            self.generic_visit(node)

    _Walk().visit(timer)
    # The real async reads are present as call nodes (not just comment text).
    assert "torch.cuda.memory_stats" in calls
    assert any(c.endswith(".query") for c in calls)
    assert any(c.endswith(".elapsed_time") for c in calls)
    # No device-draining synchronize call anywhere in the timer code.
    assert not any(c.endswith(".synchronize") for c in calls), calls


def test_hollow_tick_classifier_decisions():
    """Pure decision table for the CUDA-only tail (never reached on the CPU cell,
    so without this its branches are unexecuted in CI). Gauge keys use the same
    names _slow_tail computes deltas under."""
    from tilerl.engine import _MEM_KEYS, _StepTiming

    t = _StepTiming(None)
    t.cuda = True
    t.fwd_host_ms = 1000.0
    zero = {k: 0 for k in _MEM_KEYS if k != "reserved_bytes.all.current"}

    def cls(dev, cur_finalize=0.002, **over):
        d = dict(zero)
        d.update(over)
        t.cur = {"sparse_finalize": cur_finalize}
        return t._classify(1000.0, dev, d)

    assert cls(900, num_alloc_retries=1) == "alloc_reclaim"
    assert cls(200, num_sync_all_streams=2) == "alloc_reclaim"
    assert cls(900, **{"segment.all.allocated": 1}) == "dev_malloc"
    assert cls(900, num_device_alloc=1) == "dev_malloc"
    assert cls(None) == "unknown"                       # end event still queued
    assert cls(900, cur_finalize=0.7) == "finalize"     # >half the span
    assert cls(850) == "gpu_drain"                      # >=0.85 host
    assert cls(499) == "sync_wait"                     # <0.5 host
    assert cls(500) == "host"                          # ==0.5 host (strict <)
    t.cuda = False
    assert t._classify(1000.0, 900, zero) == "cpu"
