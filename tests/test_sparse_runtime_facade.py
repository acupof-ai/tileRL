"""Step 11 facade gates: SparseRuntime seam (docs/design-architecture.md row 11).

The sparse-tick machinery lives in src/tilerl/sparse_runtime.py behind the SAME
Engine facade the rest of the tree already uses. These gates pin:

- dense engines keep ``_sparse is None``; sparse engines expose a SparseRuntime
  whose ``.tracker`` is the SparseTracker, with the tracker attributes the
  tree reads proxied through (prefix/resident/scorer/k_pages/...);
- the moved state (graphs/flags/counters) is reachable under its old Engine
  spellings, so probes and the schedule see one object;
- the runtime module never names the Engine (no import, no ``engine.`` ref);
- a sparse tick driven through the Engine facade lands on the runtime method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tilerl.config import tiny
from tilerl.engine import SamplingParams
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.model import build_random
from tilerl.sparse_engine import SparseTracker
from tilerl.testing import RefBackend

_SRC = Path(__file__).resolve().parents[1] / "src" / "tilerl"


def _engine(sparse: bool, **kw):
    from tilerl.build import build_engine

    args = dict(
        cfg=tiny(),
        model=build_random(tiny(), seed=11),
        backend=RefBackend(),
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        prefix_store=NoPrefixStore(),
    )
    if sparse:
        args.update(sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30)
    args.update(kw)
    return build_engine(**args)


def test_dense_engine_has_no_sparse_runtime():
    e = _engine(False)
    assert e._sparse is None
    assert e._sparse_graphs == {}
    assert not e._sparse_device_select
    assert e._sparse_graph_on is False
    assert e._sparse_min_tokens == 0
    e.shutdown()


def test_sparse_engine_sparse_attribute_is_the_runtime_with_a_tracker():
    from tilerl.sparse_runtime import SparseRuntime

    e = _engine(True)
    rt = e._sparse
    assert isinstance(rt, SparseRuntime)
    assert isinstance(rt.tracker, SparseTracker)
    # proxied tracker reads the tree relies on
    assert rt.k_pages == 2
    assert rt.scorer == "bounds"
    assert rt.resident == {}
    assert rt.prefix is None  # NoPrefixStore disables sharing
    assert rt.last_selected == {}
    assert callable(rt.attach) and callable(rt.bounds_bytes)
    # the moved state under old engine spellings is the runtime's own object
    assert e._sparse_graphs is rt.graphs
    assert e._sparse_ticks_since_refresh == rt.ticks_since_refresh == 0
    assert e._sparse_device_select == rt.device_select
    assert e._sparse_graph_on == rt.graph_on
    assert e._sparse_k == 2 and e._sparse_min_tokens == 0
    e.shutdown()


def test_sparse_graph_on_setter_reaches_the_runtime():
    """Tests and capture-failure fallback flip the flag via the Engine facade."""
    e = _engine(True, sparse_device_select=True)
    assert e._sparse_graph_on is e._sparse.graph_on
    e._sparse_graph_on = False
    assert e._sparse.graph_on is False
    e.shutdown()


def test_runtime_module_has_zero_engine_references():
    import ast

    src = (_SRC / "sparse_runtime.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("engine"):
            if node.module.split(".")[-1] != "sparse_engine":
                raise AssertionError(f"imports engine module {node.module}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[-1] == "engine":
                    raise AssertionError(f"imports {a.name}")
        elif isinstance(node, ast.Attribute):
            v = node.value
            if isinstance(v, ast.Name) and v.id == "engine":
                raise AssertionError(f"engine.{node.attr} reference at line {node.lineno}")


def test_a_sparse_decode_tick_goes_through_the_runtime_method():
    e = _engine(True, sparse_device_select=True)
    prompt = np.arange(7, 7 + 12 * BLOCK_TOKENS, dtype=np.int64)
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    calls = {"n": 0}
    orig = e._sparse.run_decode_graph

    def counting(reqs, chains):
        calls["n"] += 1
        return orig(reqs, chains)

    e._sparse.run_decode_graph = counting
    out: list = []
    for _ in range(300):
        e.step()
        out = e.poll().get(rid, out)
        if len(out) >= 4:
            break
    assert calls["n"] >= 1, "the facade tick never reached SparseRuntime.run_decode_graph"
    assert len(out) == 4
    e.shutdown()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
