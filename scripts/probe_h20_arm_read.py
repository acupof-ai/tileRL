#!/usr/bin/env python3
"""Read one H20 (sm90) serve arm out of its step-timing log + /health pair.

Plan B for the H20 32k measurement: ops serves one arm at a time (`SERVE_DEPTH` /
`SERVE_SPARSE_K` / `SERVE_DECODE_GRAPH`, from scripts/serve_h20.sh), and this
reads the SHIPPED serve's own artifacts rather than building a second engine on
the same card. The log format is the engine's `_StepTiming` line, identical on
sm90 and sm70, so the steady-set filter below is the one proven on the V100
four-arm run.

    serve_h20: tree <dir> sha <10> boot N depth=<d> sparse_k=<k> decode_graph=<on|off> ctx=.. slots=.. at <ts>
    [step-timing] tick N total=..ms dec=<0|1> pre=<0|1> model=..ms sample=..ms ... path=eager|graph sparse=<0|1> ...

Two subcommands, one per artifact:

  log <serve_log> --arm-sparse <0|1> [--from-line N] [--to-line M]
      Steady-set statistics for one arm's line window. The classification is
      `steady_filter`'s -- imported, not reimplemented: that module owns the
      standard set, the tail split and the two percentile CONVENTIONS, and a
      second copy is exactly how the two conventions drifted apart before it was
      written (its docstring: sweep reports a tick-duration median and a
      per-prompt tok/s p50 on different rules). What this adds over it is the
      line window (one arm of a multi-boot log) and the boot-line self-cert.
      `--arm-sparse 0` selects the DENSE population; because `is_standard`
      already pins `sparse == 1`, the dense arm cannot be reached by adding a
      clause on top of it, so it is expressed directly.

  health <before.json> <after.json>
      Acceptance from two /health reads bracketing one arm. `spec_accepted` /
      `spec_drafted` are CUMULATIVE and warmup increments them too, so the two
      reads must bracket the arm; summing across arms double-counts, and a
      supervisor restart between them resets the counters (reported as
      ``straddled_restart`` with a null rate rather than a negative one).
      ``accept_len`` is the probe's generated-per-forward = d_generated /
      d_decode_forwards -- the engine's own forward counter, so no depth factor
      is guessed (a d3 arm drafts 3 per forward and the counter already says so).

Bare invocation runs the hermetic self-check (no torch, no card, no file): a
fixture in the exact serve log format, the dense and sparse selections, the
window and restart guards, and the forward-counter divisor in accept_len.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import steady_filter as sf  # noqa: E402

# serve_h20.sh's boot line; #758 appended the arm descriptor for self-certification.
BOOT = re.compile(r"serve_h20: tree (\S+) sha (\w+) boot (\d+) (.*?) at (\S+)")


def read_arm(log: str, lo: int, hi: int | None, want_sparse: bool) -> dict:
    """Steady-set statistics for one arm's window. ``lo``/``hi`` are 1-based lines.

    The window exists because one serve log holds every arm of a window: an arm
    is the line span between its boot line and the next one.
    """
    with open(log, errors="replace") as fh:
        raw = fh.readlines()
    boots: list[dict] = []
    rows: list[dict] = []
    for i, line in enumerate(raw, 1):
        if i < lo or (hi is not None and i > hi):
            continue
        # Inside the window, so this boot line is THIS arm's -- collected after the
        # range check. Before it, a multi-boot log reported every boot in the file
        # and the self-certification named the wrong one.
        b = BOOT.search(line)
        if b:
            boots.append(
                {
                    "line": i,
                    "tree": b.group(1),
                    "sha": b.group(2),
                    "boot": int(b.group(3)),
                    "arm": b.group(4),
                    "at": b.group(5),
                }
            )
        r = sf.parse_line(line)
        if r is not None:
            r["line"] = i
            rows.append(r)

    # The standard set, then the arm's own sparse/dense clause. `is_standard`
    # ALREADY requires `sparse == 1`, so a dense arm cannot be selected by adding
    # a clause on top of it -- that intersection is empty and A5 would report zero
    # ticks from a log full of them. The dense arm therefore drops that one clause
    # and keeps the rest (dec/model/sample/path), and pins sparse == 0 itself.
    def in_arm(r: dict) -> bool:
        if not want_sparse:
            return (
                r["dec"] == 1
                and r["sparse"] == 0
                and r["model"] > 0
                and r["sample"] > 0
                and r["path"] != "graph"
            )
        return sf.is_standard(r)

    steady = [r for r in rows if in_arm(r)]
    tail = [r for r in steady if r["total"] > sf.TAIL_MS]
    body = [r for r in steady if r["total"] <= sf.TAIL_MS]
    return {
        # the arm's own boot line is the self-certification, so it goes in the output
        "boots_in_window": boots,
        "filter": sf.STANDARD_FILTER
        if want_sparse
        else ("dec==1 and sparse==0 and model>0 and sample>0 and path!=graph"),
        "ticks_in_window": len(rows),
        "steady_ticks": len(steady),
        "body_n": len(body),
        "body_p50_ms": sf.median([r["total"] for r in body]),
        "body_p90_ms": sf.pct([r["total"] for r in body], 0.9),
        "body_mean_ms": (round(sum(r["total"] for r in body) / len(body), 1) if body else 0),
        "body_tok_s": (
            round(1000.0 / (sum(r["total"] for r in body) / len(body)), 3) if body else 0.0
        ),
        "model_p50_ms": sf.median([r["model"] for r in body]),
        "model_mean_ms": (round(sum(r["model"] for r in body) / len(body), 1) if body else 0),
        "tail_n": len(tail),
        "tail_p50_ms": sf.median([r["total"] for r in tail]),
        "tail_max_ms": max((r["total"] for r in tail), default=None),
        "excluded_path_graph_n": len([r for r in rows if r["path"] == "graph"]),
        "excluded_undecidable_n": len(
            [r for r in rows if r["path"] is None or r["sparse"] is None]
        ),
        "steady_ids": sorted(r["n"] for r in steady),
    }


def health_delta(before: dict, after: dict) -> dict:
    """Acceptance over one arm from the two /health reads that bracket it.

    ``accept_len`` divides by the engine's OWN decode-forward delta, not by a
    depth the caller supplies: at depth 3 the drafted delta is already 3x the
    forward count, so `d_drafted / depth` was a second way of writing the same
    number and a place to get it wrong. `decode_forwards` is that count.

    Counters are CUMULATIVE WITHIN ONE ENGINE, so the delta is only meaningful
    while both reads come from the same boot. A supervisor restart between them
    builds a new engine whose counters start at zero (`Engine.__init__` resets
    `_decode_forwards` / `_tokens_generated` / `_spec_drafted` / `_spec_accepted`),
    and the subtraction then yields a negative or nonsense delta that reads like a
    real acceptance figure. Detected, not assumed: a negative delta on any
    cumulative counter, or `decode_forwards` going backwards, means the bracket
    straddled a restart.
    """

    def g(d: dict, k: str) -> int:
        return (d.get("stats") or d).get(k, 0)

    da = g(after, "spec_accepted") - g(before, "spec_accepted")
    dd = g(after, "spec_drafted") - g(before, "spec_drafted")
    dg = g(after, "tokens_generated") - g(before, "tokens_generated")
    df = g(after, "decode_forwards") - g(before, "decode_forwards")
    straddled = min(da, dd, dg, df) < 0
    return {
        "d_accepted": da,
        "d_drafted": dd,
        "d_generated": dg,
        "d_decode_forwards": df,
        # ALL THREE ratios carry the straddle guard. Guarding only the two that
        # divide by `dd`/`df` when they are positive left this one bare, and a
        # restart that drove `dd` negative while `df` stayed positive printed
        # `-199.7` -- a number that reads like a measurement, in the same dict as
        # two nulls. The ratios are one claim, so they fail as one.
        "straddled_restart": straddled,
        "drafted_per_forward": round(dd / df, 3) if df > 0 and not straddled else None,
        "accept_rate": round(da / dd, 4) if dd > 0 and not straddled else None,
        # generated-per-FORWARD, the sweep probe's own ratio (n_gen / n_fwd)
        "accept_len": round(dg / df, 4) if df > 0 and not straddled else None,
        "spec_accept_in": g(after, "spec_accept_in") - g(before, "spec_accept_in"),
        "spec_drafted_in": g(after, "spec_drafted_in") - g(before, "spec_drafted_in"),
    }


def self_check() -> int:
    """Fixture in the exact serve-log format; no torch, no card, no repo file."""
    lines = [
        "serve_h20: tree /work/tilerl-s-h20 sha d135a81b boot 0 "
        "depth=1 sparse_k=128 decode_graph=on ctx=131072 slots=8 at 2026-09-20T10:00:00+08:00\n",
        "[step-timing] tick 1 total=900ms dec=0 pre=1 model=900ms\n",
        "[step-timing] tick 2 total=29ms dec=1 pre=0 graph=29ms path=graph sparse=0\n",
        "[step-timing] tick 3 total=29ms dec=1 pre=0 graph=29ms path=graph sparse=0\n",
        "[step-timing] tick 4 total=180ms dec=1 pre=0 model=170ms sample=1ms path=eager sparse=1\n",
        "[step-timing] tick 5 total=186ms dec=1 pre=0 model=172ms sample=1ms path=eager sparse=1\n",
        "[step-timing] tick 6 total=6400ms dec=1 pre=0 model=170ms sample=6200ms "
        "path=eager sparse=1\n",
        # a DENSE eager decode tick: the A5 arm's whole population. The dense
        # selection must find this one, which the old `is_standard and sparse==0`
        # intersection could not (is_standard already pins sparse==1).
        "[step-timing] tick 8 total=205ms dec=1 pre=0 model=200ms sample=1ms path=eager sparse=0\n",
        "[step-timing] tick 7 total=910ms dec=0 pre=1 model=910ms\n",
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.writelines(lines)
        path = fh.name
    try:
        r = read_arm(path, 1, None, want_sparse=True)
        assert len(r["boots_in_window"]) == 1, r["boots_in_window"]
        assert r["boots_in_window"][0]["arm"] == (
            "depth=1 sparse_k=128 decode_graph=on ctx=131072 slots=8"
        ), r["boots_in_window"]
        # the standard set drops the prefill and the two captured ticks
        assert r["steady_ticks"] == 3, r["steady_ticks"]
        assert r["excluded_path_graph_n"] == 2, r["excluded_path_graph_n"]
        # the 6400 is the close tail, reported apart, never in the body
        assert r["tail_n"] == 1 and r["tail_max_ms"] == 6400, r["tail_max_ms"]
        assert r["body_n"] == 2, r["body_n"]
        # even n -> the true median 183.0, not the nearest-rank 180
        assert r["body_p50_ms"] == 183.0, r["body_p50_ms"]
        assert 5.4 <= r["body_tok_s"] <= 5.5, r["body_tok_s"]
        # A DENSE arm must select the dense tick and NOT the sparse ones. Asserting
        # "0 ticks" here is what hid the empty-intersection bug: an arm whose whole
        # point is a number must never be allowed to read empty without complaint.
        d = read_arm(path, 1, None, want_sparse=False)
        assert d["steady_ticks"] == 1 and d["steady_ids"] == [8], d
        assert d["body_p50_ms"] == 205.0, d["body_p50_ms"]
        # the window is a line span: a window holding only the close tail has an
        # empty body, so the headline never comes from a tail tick alone
        assert read_arm(path, 7, 7, want_sparse=True)["body_n"] == 0
        # ... and a boot line OUTSIDE the window is not this arm's self-cert, which
        # a multi-boot log would otherwise report as every boot in the file
        assert read_arm(path, 900, 901, want_sparse=True)["boots_in_window"] == []
    finally:
        os.unlink(path)
    # accept_len divides by the forward counter, so the depth is READ, not told:
    # 300 drafted over 100 forwards is depth 3 on its own evidence.
    b = {
        "stats": {
            "spec_accepted": 100,
            "spec_drafted": 300,
            "tokens_generated": 50,
            "decode_forwards": 10,
        }
    }
    a = {
        "stats": {
            "spec_accepted": 163,
            "spec_drafted": 600,
            "tokens_generated": 150,
            "decode_forwards": 110,
        }
    }
    d = health_delta(b, a)
    assert (d["d_accepted"], d["d_drafted"]) == (63, 300), d
    assert d["drafted_per_forward"] == 3.0, d
    assert d["straddled_restart"] is False, d
    assert abs(d["accept_rate"] - 0.21) < 1e-9, d
    assert abs(d["accept_len"] - 1.0) < 1e-4, d
    # no forwards -> no ratio. None, not 0.0: "we measured nothing" and "we measured
    # 0% acceptance" are different claims and the table must not conflate them.
    z = health_delta(b, b)
    assert z["accept_len"] is None and z["accept_rate"] is None, z
    assert z["straddled_restart"] is False, z
    # a supervisor restart between the reads builds a fresh engine at zero, so the
    # subtraction goes negative. That is NOT a 0% acceptance: it is no measurement,
    # and it must say so rather than print a negative rate.
    restart = {
        "stats": {
            "spec_accepted": 2,
            "spec_drafted": 3,
            "tokens_generated": 4,
            "decode_forwards": 1,
        }
    }
    s = health_delta(a, restart)
    assert s["straddled_restart"] is True, s
    assert s["accept_rate"] is None and s["accept_len"] is None, s
    # The shape the fixture above missed: `df` stays POSITIVE (the new engine ran
    # some forwards) while `dd` goes negative. A guard written only on `df > 0`
    # printed `drafted_per_forward = -199.7` here -- a number that reads like a
    # measurement, sitting in the same dict as two nulls. The straddle nulls ALL
    # THREE ratios, not only those whose divisor went negative. One fixture per
    # shape of the bug, or the guard only covers the shape it was written for.
    later = {
        "stats": {
            "spec_accepted": 2,
            "spec_drafted": 30,
            "tokens_generated": 4,
            "decode_forwards": 9100,
        }
    }
    big = {
        "stats": {
            "spec_accepted": 100,
            "spec_drafted": 20000,
            "tokens_generated": 5000,
            "decode_forwards": 9000,
        }
    }
    sr = health_delta(big, later)
    assert sr["straddled_restart"] is True and sr["d_decode_forwards"] > 0, sr
    assert sr["drafted_per_forward"] is None, sr
    assert sr["accept_rate"] is None and sr["accept_len"] is None, sr
    print("h20_arm_read self-check ok")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    # Bare invocation runs the self-check: the repo-wide test_main_selfchecks gate
    # runs every hermetic script with NO args and requires rc 0, so a missing
    # subcommand must default here rather than to an argparse usage error.
    if not argv or argv[0] == "selfcheck":
        return self_check()
    if argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "log" and len(argv) >= 2:
        want_sparse = "--arm-sparse" not in argv or int(argv[argv.index("--arm-sparse") + 1]) > 0
        lo = int(argv[argv.index("--from-line") + 1]) if "--from-line" in argv else 1
        hi = int(argv[argv.index("--to-line") + 1]) if "--to-line" in argv else None
        print(json.dumps(read_arm(argv[1], lo, hi, want_sparse), indent=1))
        return 0
    if argv[0] == "health" and len(argv) >= 3:
        with open(argv[1]) as f1, open(argv[2]) as f2:
            print(json.dumps(health_delta(json.load(f1), json.load(f2)), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":  # runnable check
    assert main() == 0, "self-check failed"
