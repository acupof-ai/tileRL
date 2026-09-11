"""Census of the host-bound work in ONE sparse vs dense decode tick.

The sparse decode path is eager: every tick rebuilds block tables and scores
candidate pages in Python, so the graph cannot be captured until that host work
is known and removed. This script counts, on the CPU engine at B=1 and B=8 with
a pool big enough that selection is real (sparse_k=2):

- aten ops dispatched inside one decode ``step`` and inside each full-attn
  plane's ``SparseForward.attention_args`` (one call per plane);
- host<->framework syncs: ``_local_scalar_dense`` (``.item()``/``bool(t)``/
  ``float(t)``) and ``.tolist()`` (Tensor.data.tolist, a host copy);
- Python-built tensors entering the graph: ``lift_fresh*`` (``torch.tensor``
  from a python list/int/scalar — a host allocation handed to aten).

Run it on three heads and diff:

    python scripts/census_sparse_decode_host_ops.py main
    git checkout <#534> && python scripts/census_sparse_decode_host_ops.py pin
    git checkout <#527> && python scripts/census_sparse_decode_host_ops.py bounds

The counter is read-only; no engine or kernel code changes. CPU only — the
COUNTS are what matter for graph-capturability, and they are identical on a
card (the sync/python-tensor sites are device-independent).
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from torch.utils._python_dispatch import TorchDispatchMode

from tilerl.config import tiny
from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.model import build_random
from tilerl.testing import RefBackend

# aten -> host-work classification.
_SCALAR_DENSE = "aten._local_scalar_dense.default"  # .item()/bool()/float()/int()
_TOLIST = "aten._tolist.data"  # tensor.tolist() host copy
# torch.tensor([...]) / torch.tensor(n) from a Python object is a host
# allocation handed into aten via lift_fresh{_copy}; these are the python-built
# tensors a captured graph cannot re-read from Python each replay.
_PY_BUILT = ("aten.lift_fresh.default", "aten.lift_fresh_copy.default")
# explicit host<->device movement on a cuda engine (none expected on CPU, kept
# so the same script classifies a card run correctly).
_H2D_D2H = ("aten._to_copy.default", "aten.copy_.default")


@dataclass
class Census:
    ops: Counter = field(default_factory=Counter)
    # first tilerl caller frame -> count, for every dispatched aten op
    sites: Counter = field(default_factory=Counter)

    def record(self, name: str) -> None:
        self.ops[name] += 1
        self.sites[_first_tilerl_site()] += 1

    # ---- derived buckets ----
    @property
    def aten_total(self) -> int:
        return sum(self.ops.values())

    @property
    def sync_scalar(self) -> int:
        return self.ops.get(_SCALAR_DENSE, 0)

    @property
    def sync_tolist(self) -> int:
        return self.ops.get(_TOLIST, 0)

    @property
    def py_built_tensors(self) -> int:
        return sum(self.ops.get(n, 0) for n in _PY_BUILT)

    @property
    def host_device_moves(self) -> int:
        return sum(self.ops.get(n, 0) for n in _H2D_D2H)

    def as_row(self, label: str) -> dict:
        return {
            "source": label,
            "aten_ops": self.aten_total,
            "sync_item_bool_float": self.sync_scalar,
            "sync_tolist": self.sync_tolist,
            "python_built_tensors": self.py_built_tensors,
            "host_device_moves": self.host_device_moves,
        }


class _Count(TorchDispatchMode):
    def __init__(self, census: Census) -> None:
        self.c = census

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.c.record(str(func))
        return func(*args, **(kwargs or {}))


def _first_tilerl_site() -> str:
    """file:function:line of the first tilerl frame on the dispatch stack — the
    engine/kernel site that issued the aten op (skips torch internals)."""
    import inspect
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.realpath(root)  # /tmp -> /private/tmp on macOS
    for fr in inspect.stack()[2:]:
        fn = os.path.realpath(fr.filename)
        if fn.startswith(root) and os.sep + "tilerl" + os.sep in fn:
            rel = fn.replace(root + os.sep, "")
            return f"{rel}:{fr.function}:{fr.lineno}"
    return "<non-tilerl>"


def _engine(sparse: bool, b: int):
    kw = dict(
        cfg=tiny(),
        model=build_random(tiny(), seed=11),
        backend=RefBackend(),
        num_blocks=128,
        num_slots=b + 1,
        max_batch=b,
        max_total_tokens=8192,
        max_num_batched_tokens=512,
        prefix_store=NoPrefixStore(),
    )
    if sparse:
        kw.update(sparse_k=2, scorer="bounds", kv_cold_bytes=1 << 30)
    return build_engine(**kw)


def _run_to_decode_then_census(sparse: bool, b: int, tag: str) -> dict:
    """Drain prefill, then count ONE decode step with b active decode rows.

    Selection is real: the prompt spans many more pages than k, so every decode
    tick has a non-empty candidate set to score and promote."""
    e = _engine(sparse, b)
    prompts = [(np.arange(7 + i, 7 + i + 6 * BLOCK_TOKENS, dtype=np.int64)) for i in range(b)]
    rids = []
    for i, p in enumerate(prompts):
        rids.append(e.submit(p, SamplingParams(temperature=0.0, max_new_tokens=8, seed=i)))
    # advance until all rows are in decode (prefill drained)
    for _ in range(512):
        dec = [r for r in e._running if r.phase == 2]
        if len(dec) == b:
            break
        e.step()

    tick = Census()
    per_plane: list[Census] = []

    # SparseForward lives on the BatchKv built per step; patch the CLASS method
    # so every tick's instance is covered, capturing one labeled window per call.
    if sparse:
        from tilerl.sparse_engine import SparseForward

        orig = SparseForward.attention_args

        def counted(self, plane, q):  # type: ignore[no-untyped-def]
            pc = Census()
            with _Count(pc):
                out = orig(self, plane, q)
            per_plane.append(pc)
            return out

        SparseForward.attention_args = counted

    with _Count(tick):
        e.step()

    if sparse:
        SparseForward.attention_args = orig  # noqa: F821 (defined iff sparse)

    e.shutdown()
    row = tick.as_row(tag)
    if sparse and per_plane:
        # attention_args fires once per source/mate plane; report per-plane mean
        # and the source-plane subset (groups cache, so only the first per group
        # does the scoring work).
        row["attn_planes_called"] = len(per_plane)
        row["plane_aten_mean"] = round(sum(p.aten_total for p in per_plane) / len(per_plane), 1)
        row["plane_sync_mean"] = round(
            sum(p.sync_scalar + p.sync_tolist for p in per_plane) / len(per_plane), 1
        )
        row["plane_pybuilt_mean"] = round(
            sum(p.py_built_tensors for p in per_plane) / len(per_plane), 1
        )
        row["plane_top_ops"] = Counter()
        for p in per_plane:
            row["plane_top_ops"].update(p.ops)
    # whole-tick site attribution (sparse AND dense, for the delta map)
    row["tick_sites"] = tick.sites
    return row


def main(tag: str) -> None:
    rows = []
    for b in (1, 8):
        rows.append(_run_to_decode_then_census(False, b, f"dense B={b} {tag}"))
        rows.append(_run_to_decode_then_census(True, b, f"sparse B={b} k=2 {tag}"))
    print(f"\n=== host-op census: one decode tick ({tag}) ===")
    hdr = (
        "source",
        "aten_ops",
        "item/bool/float",
        "tolist",
        "py-built tensors",
        "H<->D",
        "planes",
        "aten/plane",
        "sync/plane",
        "pybuilt/plane",
    )
    print("{:<22} {:>8} {:>14} {:>7} {:>16} {:>6} {:>6} {:>10} {:>11} {:>13}".format(*hdr))
    for r in rows:
        print(
            "{:<22} {:>8} {:>14} {:>7} {:>16} {:>6} {:>6} {:>10} {:>11} {:>13}".format(
                r["source"],
                r["aten_ops"],
                r["sync_item_bool_float"],
                r["sync_tolist"],
                r["python_built_tensors"],
                r["host_device_moves"],
                r.get("attn_planes_called", "-"),
                r.get("plane_aten_mean", "-"),
                r.get("plane_sync_mean", "-"),
                r.get("plane_pybuilt_mean", "-"),
            )
        )

    # top host-work sites by total count across the two sparse ticks
    print("\n=== top aten ops in the two SPARSE decode ticks ===")
    tot = Counter()
    for r in rows:
        if r["source"].startswith("sparse") and "plane_top_ops" in r:
            tot.update(r["plane_top_ops"])
    for name, n in tot.most_common(12):
        print(f"{n:>5}  {name}")

    # Top 5 tilerl sites by aten-op count, summed over the two sparse ticks
    # (file:function:line — the host-work hot spots to make graph-capturable).
    by_b = {1: Counter(), 8: Counter()}
    for r in rows:
        b = 1 if "B=1" in r["source"] else 8
        by_b[b].update(r["tick_sites"])
    for b in (1, 8):
        print(f"\n=== top 5 tilerl sites by aten ops (B={b}) ===")
        for name, n in by_b[b].most_common(5):
            print(f"{n:>5}  {name}")

    # The actionable map: sparse-MINUS-dense per site at B=8 — the host work the
    # sparse selection adds on top of a dense decode tick.
    print("\n=== top 5 sparse-minus-dense sites, B=8 (the selection host work) ===")
    d8 = next(r["tick_sites"] for r in rows if r["source"].startswith("dense B=8"))
    s8 = next(r["tick_sites"] for r in rows if r["source"].startswith("sparse B=8"))
    delta = Counter(s8)
    delta.subtract(d8)
    for name, n in delta.most_common(5):
        if n <= 0:
            break
        print(f"{n:>+5}  {name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "head")
