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

Two arms run in ONE process, order A then reversed B:
    w2_graph  depth=1, sparse graph on   (the cost under question)
    w1_graph  depth=0, no draft          (p4_verify and p5_draft must be 0)
The device window runs three cmax buckets (512/1024/2048) per arm.

Refresh ticks (every SPARSE_REFRESH_TICKS the sparse graph declines and the tick
runs eager) and the first capture tick are excluded; n>=50 steady graph ticks per
cell, p50/p90 plus every per-tick value are reported.

CPU precondition (must pass before any device window): ``--model tiny`` builds
the random model + an always-accept oracle draft on the CPU reference and runs
all three arms end to end; the depth-0 arm must have p4_verify = p5_draft = 0.
Negative control: ``--neg-depth0-with-draft`` builds that arm WITH a draft and
the spec-phase gate must go RED.
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
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.build import build_engine  # noqa: E402
from tilerl.config import tiny  # noqa: E402
from tilerl.engine import SamplingParams  # noqa: E402
from tilerl.kv_cache import BLOCK_TOKENS  # noqa: E402
from tilerl.model import build_random  # noqa: E402

PHASE_WALL = (
    "p_rows", "p0_fill", "p1_h2d", "p2_replay", "p3_finalize",
    "p4_verify", "p_sample", "p5_draft",
)
# A draft-free arm must never record these (p5_draft wall mark is the gate).
CLOSURE_TOL = 0.05
N_STEADY = 50
CMAX_BUCKETS = [512, 1024, 2048]
PHASE_DECODE = 2

class ProbeFail(Exception):
    """A hard instrument/precondition failure. Raised with an exit-class tag so
    main maps a missing/incomplete observation to rc=14 (INSUFFICIENT), distinct
    from a closure-gate failure (rc=1)."""

    def __init__(self, msg: str, rc: int = 14):
        super().__init__(msg)
        self.rc = rc


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

    def attach(self, backend, num_blocks, dtype=None):
        # The engine capacity check reads self.kv.num_blocks; give the oracle a
        # real dense draft pool (it is never populated — step() is a no-op).
        from tilerl.kv_cache import PagedKvPool

        self.kv = PagedKvPool(
            num_blocks, self.cfg.num_kv_heads, self.cfg.head_dim,
            num_layers=self.cfg.num_layers, device=backend.device)

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
        # Leave a one-token chain so the NEXT tick is a real W=2 verify
        # ([last, draft]). The token matches what the trunk draws (expected), so
        # the chain is accepted. The phase precondition does not assert token
        # correctness (that is test_sparse_graph_verify_tick_w2_*); it only
        # needs the spec phases (p4_verify/p5_draft) to actually execute.
        for r in rows:
            if getattr(r, "done", False):
                continue
            r.drafts = [self.expected.get(r.seq_len, 0)]
            r.draft_pos = max(getattr(r, "draft_pos", r.seq_len - 1), r.seq_len - 1)


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


def measure_graph_ticks(e, want=N_STEADY, max_attempts=None):
    """Step one engine and collect `want` STEADY sparse-graph ticks from the
    product _StepTiming buckets. Each returned row carries every PHASE_WALL value
    (missing keys are a hard error, never silently 0), the product 'graph'
    envelope as wall, and the per-tick closure gap. Refresh/prefill/mixed ticks
    and ticks that did not run exactly one sparse graph are SKIPPED, not counted.
    Raises ProbeFail(rc=14) if `want` steady ticks are not observed."""
    max_attempts = max_attempts or want * 16
    tm = e._step_timing
    if tm is None:
        raise ProbeFail("engine built without _StepTiming (set TILERL_STEP_TIMING=1)")
    runtime = e._sparse
    raw_call = runtime.run_decode_graph
    track = PHASE_WALL + ("graph",)

    def snap():
        # cur is moved to tot in tick_end() before step() returns, so diff the
        # cumulative totals one tick at a time.
        return {k: tm.tot.get(k, 0.0) for k in track}

    rows, attempts, prev, prev_fwd = [], 0, snap(), e._decode_forwards
    while len(rows) < want and attempts < max_attempts:
        attempts += 1
        ran = {"v": False}

        def wrapper(reqs, chains=None, raw=raw_call, ran=ran):
            ok = raw(reqs, chains)
            ran["v"] = bool(ok)
            return ok

        runtime.run_decode_graph = wrapper
        e.step()
        runtime.run_decode_graph = raw_call
        cur = snap()
        d = {k: (cur[k] - prev[k]) * 1000.0 for k in track}
        graph_ticks = e._decode_forwards - prev_fwd
        prev, prev_fwd = cur, e._decode_forwards
        if not ran["v"] or graph_ticks != 1 or d["graph"] <= 0.0:
            continue  # refresh / prefill / capture-declined: excluded

        ph = {k: d[k] for k in PHASE_WALL}  # every p_* key must exist
        psum = sum(ph.values())
        wall = d["graph"]
        rows.append({"phases_ms": ph, "sum_ms": psum, "wall_ms": wall,
                     "gap": abs(psum - wall) / wall})

    if len(rows) < want:
        raise ProbeFail(f"only {len(rows)} steady sparse-graph ticks (<{want}) "
                        f"after {attempts} steps", rc=14)
    return rows


def run_arm(name, depth, graph, neg_depth0_draft=False):
    """Tiny CPU arm: build, prime past capture, measure N_STEADY graph ticks."""
    cfg = tiny()
    model = build_random(cfg, seed=11)
    be = get_backend()
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

    rows = measure_graph_ticks(e)
    e.shutdown()
    return {0: rows}  # one synthetic cell for the CPU precondition


def _tokens_for_bucket(bucket: int) -> int:
    # Same geometry as scripts/probe_sparse_graph_cmax_bucket.py tokens_for_bucket:
    # linear 1 cmax per 16 tokens; midpoint of the 16-wide window so a one-token
    # scheduling slip cannot push the tick into the adjacent bucket.
    return (bucket + 7) * 16 + 7


def _prime_bucket_27b(e, tok, bucket, depth, gen):
    """Submit the counting prompt whose FIRST decode tick lands in `bucket` and
    wait until decode begins (measure-and-correct, mirroring prime_counting).
    Returns the rid. Does NOT drain decode — measure_graph_ticks owns that."""
    from tilerl.sparse_engine import cmax_bucket

    instr = tok.encode(" Count aloud from one to forty, one number per line:")
    fid = tok.encode(" z")[-1:]
    nfill = _tokens_for_bucket(bucket) - len(instr)
    for _ in range(6):
        ids = fid * nfill + instr
        rid = e.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=gen, seed=0))
        for _ in range(40000):
            e.step()
            row = next((r for r in e._running if r.req_id == rid), None)
            if row is None:
                raise ProbeFail(f"b{bucket}: row vanished before decode")
            if row.phase == PHASE_DECODE:
                srows = e._sparse.decode_rows([row], [1 + depth])
                cmax = max((len(x["cand"]) for x in srows), default=0)
                if cmax_bucket(cmax) != bucket:
                    raise ProbeFail(
                        f"b{bucket}: first decode tick in cmax bucket {cmax_bucket(cmax)} "
                        f"(cmax={cmax}), expected {bucket}", rc=14)
                return rid
            if getattr(row, "phase", None) > PHASE_DECODE:
                raise ProbeFail(f"b{bucket}: request DONE before decode", rc=14)
        nfill += bucket * 16  # never reached: loop above returns on first decode
    raise ProbeFail(f"b{bucket}: counting prompt never entered decode", rc=14)


def run_arm_27b(name, depth, graph, source, draft_path, neg=False):
    """Device arm on qwen38-27b: ONE engine, the three cmax buckets as three
    sequential requests in one process; >=N_STEADY steady graph ticks per bucket.
    depth=0 builds without a draft (the neg control is not used on device)."""
    import torch

    from tilerl import build as build_mod
    from tilerl.build import build_model
    from tilerl.cli import _qwen38_tokenizer
    from tilerl.spec import load_draft

    build_mod.QWEN38_SOURCE = source
    be = get_backend()
    if be.device.type != "cuda":
        raise ProbeFail("qwen38-27b arm requires a CUDA backend", rc=14)
    cfg, model = build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, draft_path) if depth >= 1 else None

    e = build_engine(
        cfg, model, be,
        num_slots=4, max_batch=4, max_total_tokens=131072,
        max_num_batched_tokens=512,
        sparse_k=128, sparse_min_tokens=0, sparse_device_select=True,
        scorer="bounds",
        kv_cold_bytes=int(os.environ.get("H2_COLD_BYTES", str(1 << 30))),
        cold_ssd_path=os.environ.get("H2_COLD_SSD", ""),
        cold_ssd_bytes=int(os.environ.get("H2_COLD_SSD_BYTES", "0")),
        cold_format="f16",
        decode_graph=graph, draft=draft,
        spec_depth=depth if draft is not None else None,
    )
    tok = _qwen38_tokenizer()
    cells = {}
    try:
        for bucket in CMAX_BUCKETS:
            rid = _prime_bucket_27b(e, tok, bucket, depth, gen=max(N_STEADY * 4, 240))
            cells[bucket] = measure_graph_ticks(e, want=N_STEADY)
            # drain/finish this request before priming the next bucket
            for _ in range(40000):
                if not any(r.req_id == rid for r in e._running):
                    break
                e.step()
    finally:
        e.shutdown()
        if be.device.type == "cuda":
            torch.cuda.synchronize()
    return cells


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


def _check_cells(arm, cells, neg=False):
    """Apply the gates to one arm's per-cell tick rows. Returns failure strings:
    per-cell steady-count + every-phase-present (enforced at collection) + the
    per-tick 5% closure gate (numerator and denominator printed), and the
    draft-free spec-phase gate for w1_graph."""
    out = []
    for bucket, rows in sorted(cells.items()):
        tag = f"{arm} b{bucket}"
        if len(rows) < N_STEADY:
            out.append(f"{tag}: {len(rows)} steady ticks < {N_STEADY}")
            continue
        for i, r in enumerate(rows):
            if r["gap"] > CLOSURE_TOL:
                out.append(
                    f"{tag} tick{i}: closure gap {r['gap']:.1%} > {CLOSURE_TOL:.0%} "
                    f"(numerator sum={r['sum_ms']:.4f}ms denominator graph={r['wall_ms']:.4f}ms)")
                break
    if arm == "w1_graph":
        # Depth-0 is spec-free. The negative control forces a draft in, which
        # must trip the SAME assertion (no neg branch -> cannot pass vacuously).
        dirty = [(b, i) for b, rows in sorted(cells.items()) for i, r in enumerate(rows)
                 if r["phases_ms"]["p4_verify"] > 0.0 or r["phases_ms"]["p5_draft"] > 0.0]
        if dirty:
            kind = "forced-draft NEGATIVE CONTROL" if neg else "draft-free"
            out.append(f"w1_graph: spec phase recorded on the {kind} arm at "
                       f"{[(b, i) for b, i in dirty[:5]]}")
    return out


def _brief_cells(cells):
    brief = {}
    for bucket, rows in sorted(cells.items()):
        s = summarize(rows)
        brief[str(bucket)] = {
            "n": s["n"], "wall_p50_ms": round(s["wall_ms"]["p50"], 3),
            "wall_p90_ms": round(s["wall_ms"]["p90"], 3),
            "gap_p90": round(s["gap"]["p90"], 4),
            **{f"{ph}_p50_ms": round(s[ph]["p50"], 3) for ph in PHASE_WALL}}
    return brief


def _assert_tree(expected: str):
    """Device precondition: the checked-out tree must equal the sha the window
    command passed (PROBE_SHA), not a constant baked into the script — the probe
    head advances and a hardcoded sha would rc14 on the first line."""
    import subprocess

    if not expected:
        raise ProbeFail("--expect-tree PROBE_SHA is required for --model qwen38-27b", rc=14)
    sha = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if sha != expected[:8]:
        raise ProbeFail(
            f"tree is {sha}, window expected {expected[:8]} (PROBE_SHA); "
            f"fetch/checkout the expected probe head before running", rc=14)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny", choices=["tiny", "qwen38-27b"])
    ap.add_argument("--source", default="", help="qwen38-27b model source dir")
    ap.add_argument("--draft", default="", help="draft MTP path (27b)")
    ap.add_argument("--expect-tree", default="",
                    help="required checked-out short sha (PROBE_SHA); mandatory for 27b")
    ap.add_argument("--out", default="phase_timing.json")
    ap.add_argument("--neg-depth0-with-draft", action="store_true",
                    help="CPU negative control only: force a draft on the depth-0 arm")
    args = ap.parse_args()

    order_a = [("w2_graph", 1, True), ("w1_graph", 0, True)]
    order = list(reversed(order_a)) if os.environ.get("PHASE_ORDER_B") else order_a

    result, failures = {}, []
    try:
        if args.model == "qwen38-27b":
            _assert_tree(args.expect_tree)
        for arm, depth, graph in order:
            neg = args.neg_depth0_with_draft and arm == "w1_graph"
            if args.model == "qwen38-27b":
                cells = run_arm_27b(arm, depth, graph, args.source, args.draft)
            else:
                cells = run_arm(arm, depth, graph, neg_depth0_draft=neg)
            failures += _check_cells(arm, cells, neg=neg)
            result[arm] = _brief_cells(cells)
    except ProbeFail as exc:
        print(f"PHASE PROBE INSUFFICIENT (rc14): {exc}", file=sys.stderr)
        return exc.rc

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    if failures:
        print("PHASE PROBE FAILED", file=sys.stderr)
        for x in failures:
            print("  " + x, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
