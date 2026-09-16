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

#: Every segment name the probe may emit. "forward" is an envelope (see its
#: docstring); graph ticks carry "graph" instead of the eager inner set.
_SEGMENTS = {
    "plan", "stats", "forward", "charge", "graph",
    "sparse_select", "prep", "model", "sparse_finalize", "sample",
    "draft_blocks", "draft_step", "offers_pub",
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


def test_timing_on_segments_reconcile(monkeypatch):
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
        tm.report()  # the atexit callback: must not crash with real data
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
