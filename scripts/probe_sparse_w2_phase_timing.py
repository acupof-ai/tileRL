#!/usr/bin/env python3
"""#805 W=2 captured sparse decode tick — phase-attribution probe.

Splits ONE steady-state sparse graph tick into the phases the product marks via
the env-gated ``_StepTiming`` instrument (no second product timer):

    p_rows   decode geometry + bucket/graph lookup (host)
    p0_fill  own resolve + candidate/own staging fill (host, may drain promotion H2D)
    p1_h2d   pinned staging -> static device buffers (host enqueue)
    p2_replay  captured CUDAGraph.replay() device span (CUDA event, probe-only)
    p3_finalize  bounds store + demote/promote pin set (host)
    p4_verify  spec sample/argmax + select_step state restore (host; spec only)
    p_sample  depth-0 analogue of p4_verify (plain sample commit)
    p5_draft  MTP draft forward for the NEXT chain (CUDA event via TILERL_DRAFT_TIMING)

Closure is a GATE, not a report line: per tick
    |host-phase sum + p2_dev + p5_dev - call_wall| / call_wall <= 5%
where ``call_wall`` brackets ``_run_sparse_decode_graph`` in this probe. Host
phases are wall-clock (they carry the GPU drain they block on); p2/p5 are device
spans, so the sum CAN exceed the single-stream wall and the assert is non-vacuous.
Failure aborts with a non-zero exit and prints the offending tick, never a table.

Three arms run in ONE process, order A then reversed B:
    w2_graph  depth=1, sparse graph on   (the cost under question)
    w1_graph  depth=0, no draft          (p4_verify and p5_draft must be 0)
    w2_eager  depth=1, decode_graph off  (splits replay from fill/finalize)

Refresh ticks (every SPARSE_REFRESH_TICKS the sparse graph declines and the tick
runs eager) and the first capture tick are excluded; n>=50 steady graph ticks per
cell, p50/p90 plus every per-tick value are reported.

CPU precondition (must pass before any device window): ``--model tiny`` builds
the random model + an always-accept oracle draft on the CPU reference and runs
all three arms end to end; the depth-0 arm must have p4_verify = p5_draft = 0.
Negative control: ``--neg-depth0-with-draft`` builds that arm WITH a draft and
the closure/phase gate must go RED.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

os.environ.setdefault("TILERL_STEP_TIMING", "1")
os.environ.setdefault("TILERL_STEP_TIMING_SLOW_MS", "0")
os.environ.setdefault("TILERL_DRAFT_TIMING", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tilerl.build import build_engine  # noqa: E402
from tilerl.config import tiny  # noqa: E402
from tilerl.engine import SamplingParams  # noqa: E402
from tilerl.kv_cache import BLOCK_TOKENS  # noqa: E402
from tilerl.model import build_random  # noqa: E402
from tilerl.testing import RefBackend  # noqa: E402


class _ProbeBackend(RefBackend):
    """RefBackend + the draft-serve probe _serve_draft queries (the real
    get_backend() pulls tilelang, which the CPU precondition box lacks)."""

    def has_kernel(self, name):
        return False

PHASE_WALL = (
    "p_rows", "p0_fill", "p1_h2d", "p2_replay", "p3_finalize",
    "p4_verify", "p_sample", "p5_draft",
)
# A draft-free arm must never record these (p5_draft wall mark is the gate).
CLOSURE_TOL = 0.05
N_STEADY = 50


class OracleDraft:
    """Always-accept draft: proposes the token the trunk draws at each position,
    so every W=2 verify tick runs and commits the full chain. Mirrors the
    token-exact oracle in tests/test_sparse_engine.py."""

    def __init__(self, cfg, expected):
        from dataclasses import replace

        self.cfg = replace(cfg, num_layers=1, full_attn_layers=(0,))
        self.params: dict = {}
        self.expected, self.width, self.has_confidence, self.trunk = expected, 2, False, None
        self.forwards = 0
        self.aux_layers = ()
        self.no_quant = True
        self.kv = None

    def set_depth(self, depth):
        self.depth = depth

    def attach(self, *a, **k):
        pass

    def forward(self, hidden, ids, positions, kv, backend, hidden_out=None, last_only=False):
        pos = np.atleast_2d(np.asarray(positions))
        logits = torch.zeros(*pos.shape, self.cfg.vocab_size, device=backend.device)
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                logits[i, j, self.expected.get(int(pos[i, j]) + 1, 0)] = 10.0
        if hidden_out is not None:
            hidden_out.append(torch.as_tensor(hidden))
        self.forwards += 1
        return logits

    def confidence(self, hidden, probs, backend):
        return probs

    def step(self, rows):
        pass


def _drain(engine, pid, n):
    out = []
    for _ in range(n * 8):
        engine.step()
        r = next((x for x in engine._running if x.req_id == pid), None)
        if r is not None and len(r.output) > len(out):
            out = list(r.output)
        if len(out) >= n:
            break
    return out


def run_arm(name, depth, graph, neg_depth0_draft=False):
    """One arm: build, prime past capture, measure N_STEADY graph ticks from the
    product _StepTiming wall buckets, return per-tick phase rows. depth=0 carries
    no draft unless the negative control forces one (which must trip the gate)."""
    cfg = tiny()
    model = build_random(cfg, seed=11)
    be = _ProbeBackend()
    prompt = np.arange(7, 7 + 4 * BLOCK_TOKENS + 15, dtype=np.int64)

    has_draft = (depth >= 1) or neg_depth0_draft
    draft = None
    if has_draft:
        dense = build_engine(
            cfg=cfg, model=build_random(cfg, seed=11), backend=be,
            num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
            max_num_batched_tokens=512, sparse_k=0)
        rid0 = dense.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=8, seed=0))
        base = _drain(dense, rid0, 8)
        dense.shutdown()
        expected = {i: t for i, t in enumerate(list(prompt) + base)}
        draft = OracleDraft(cfg, expected)

    e = build_engine(
        cfg=cfg,
        model=model,
        backend=be,
        num_blocks=64,
        num_slots=4,
        max_batch=1,
        max_total_tokens=4096,
        max_num_batched_tokens=512,
        sparse_k=2,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        sparse_device_select=True,
        decode_graph=graph,
        draft=draft,
        spec_depth=depth if has_draft else None,
    )

    e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=400, seed=0))
    # Prime past the capture tick (steady state only).
    for _ in range(20):
        e.step()

    tm = e._step_timing
    runtime = e._sparse
    raw_call = runtime.run_decode_graph
    rows = []
    import time

    attempts = 0
    while len(rows) < N_STEADY and attempts < N_STEADY * 8:
        attempts += 1
        cur0 = {k: tm.cur.get(k, 0.0) for k in PHASE_WALL}
        called = {"ran": False, "wall": 0.0}

        def wrapper(reqs, chains=None, raw=raw_call, called=called):
            t0 = time.perf_counter()
            ok = raw(reqs, chains)
            called["wall"] = time.perf_counter() - t0
            called["ran"] = bool(ok)  # a refresh tick returns False and runs eager
            return ok

        runtime.run_decode_graph = wrapper
        e.step()
        runtime.run_decode_graph = raw_call
        wall = called["wall"]
        if not called["ran"]:
            continue  # refresh / prefill / capture-declined tick: excluded

        ph = {k: (tm.cur.get(k, 0.0) - cur0[k]) * 1000.0 for k in PHASE_WALL}
        psum = sum(ph.values())
        gap = abs(psum - wall * 1000.0) / (wall * 1000.0)
        rows.append({"phases_ms": ph, "sum_ms": psum, "wall_ms": wall * 1000.0, "gap": gap})

    e.shutdown()
    return rows, name, depth, graph


def _pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round((q / 100.0) * (len(xs) - 1))))
    return xs[i]


def summarize(rows):
    def agg(vals):
        return {"p50": statistics.median(vals), "p90": _pct(vals, 90)}
    out = {"n": len(rows), "ticks": rows,
           "wall_ms": agg([r["wall_ms"] for r in rows]),
           "gap": agg([r["gap"] for r in rows])}
    for k in PHASE_WALL:
        out[k] = agg([r["phases_ms"][k] for r in rows])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase_timing.json")
    ap.add_argument("--neg-depth0-with-draft", action="store_true",
                    help="negative control: attach a draft to the depth-0 arm; gate must fail")
    args = ap.parse_args()
    # This checked-in probe is the CPU precondition (tiny + RefBackend). The
    # device 27B window lands through the identical code path; fixmisc swaps the
    # build_* args (qwen38-27b / cuda / real draft) at the window, no phase change.

    # w2_eager (decode_graph off) maps onto the pre-existing eager buckets
    # (sparse_select/model/sparse_finalize/sample/draft_step) and is the next step;
    # this first cut answers the W2-vs-W1 graph-tick delta with the new p_* buckets.
    order_a = [("w2_graph", 1, True), ("w1_graph", 0, True)]
    order = list(reversed(order_a)) if os.environ.get("PHASE_ORDER_B") else order_a

    result, failures = {}, []
    for name, depth, graph in order:
        neg = args.neg_depth0_with_draft and name == "w1_graph"
        rows, _, _, _ = run_arm(name, depth, graph, neg_depth0_draft=neg)
        if len(rows) < N_STEADY:
            failures.append(f"{name}: only {len(rows)} steady graph ticks (<{N_STEADY})")
        for i, r in enumerate(rows):
            if r["gap"] > CLOSURE_TOL:
                failures.append(
                    f"{name} tick{i}: closure gap {r['gap']:.1%} > {CLOSURE_TOL:.0%} "
                    f"(sum={r['sum_ms']:.3f} wall={r['wall_ms']:.3f})")
                break
        result[name] = summarize(rows)

    # Spec-only gate: a draft-free depth-0 arm must never record p4_verify or
    # p5_draft. The forced-draft negative control makes p5_draft non-zero, which
    # MUST trip this — if it stays green the gate is vacuous.
    d0 = result["w1_graph"]["ticks"]
    d0_dirty = [i for i, r in enumerate(d0)
                if r["phases_ms"]["p4_verify"] > 0.0 or r["phases_ms"]["p5_draft"] > 0.0]
    if args.neg_depth0_with_draft:
        if not d0_dirty:
            failures.append("NEG CONTROL DID NOT FIRE: depth0-with-draft recorded no p4/p5")
    elif d0_dirty:
        failures.append(f"w1_graph: spec phase recorded on draft-free ticks {d0_dirty[:5]}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    if failures:
        print("PHASE PROBE FAILED", file=sys.stderr)
        for x in failures:
            print("  " + x, file=sys.stderr)
        return 1
    brief = {}
    for k, v in result.items():
        brief[k] = {"n": v["n"], "wall_p50_ms": round(v["wall_ms"]["p50"], 3),
                    "gap_p90": round(v["gap"]["p90"], 4),
                    **{f"{ph}_p50_ms": round(v[ph]["p50"], 3) for ph in PHASE_WALL}}
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
