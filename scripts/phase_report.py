#!/usr/bin/env python3
"""Split an arm into its two cold-tier phases and report tick stats per phase.

Why phases and not a "plateau": with the SSD cap off and one independent
long-context request at a time, the four-key cold total has NO plateau by
construction -- it grows until the disk fills. The reproducible boundary across
boots is the moment the HOST budget saturates, after which cold only grows on
the SSD side and `ssd_mmap` starts appearing on ticks. That boundary is fixed
configuration, so every arm crosses it at the same occupancy.

    phase A (host filling)  : before the boundary
    phase B (host full)     : from the boundary on   [ssd_mmap may appear]

A running "the four-key total plateaus" test was tried and is WRONG: on the A2
trace the total wobbles ~2% per sample (private rises while shared falls)
against a ~2% fill rate, so a windowed range test reports a plateau while the
tier is still filling. The phase boundary replaces it.

**The boundary is the first steady tick whose `ssd_mmap` is nonzero**, not the
first trace sample whose `shared` reaches the cap. The trace samples every 10 s,
so a cap comparison lands after the true crossing (the sample before it can be a
GiB short with no SSD at all); the log's own first `ssd_mmap` tick is the action
itself. The trace comparison is printed as corroboration and is never the
boundary.

Two coordinates, two files, and one bridge. The trace and the serve log are
different files in different units; the only field in the same coordinate as the
log's tick counter is the trace's `decfwd` (decode forwards), and the identity is
exact: on A2 each of the 30 per-request boundaries satisfies
``cum_steady_ticks == Δdecode_forwards`` (diff 0). So a boundary taken from the
trace maps onto a tick by `decfwd` -- never by sample index, and never by line
number. The first version of this file sliced on the sample index and cut inside
the wrong request.

Read-only. The steady set comes from
`scripts/steady_filter.py`'s ``is_standard`` and the cold trace from
`scripts/cold_trace.py`; neither is re-implemented here, so there is one
definition of "steady" and one of "how a trace line is read" in the tree.

    python3 scripts/phase_report.py --trace cold_trace.txt --log serve.log
    python3 scripts/phase_report.py --trace t.txt --log l.log --cap-gib 64
    python3 scripts/phase_report.py --self-check
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cold_trace import ColdTraceError, known_total  # noqa: E402
from cold_trace import rows as trace_rows  # noqa: E402
from steady_filter import TAIL_MS, is_standard, median, parse_rows, pct  # noqa: E402

#: Default host cold budget the boundary is corroborated against: the V100/H20
#: hybrid serve runs `--kv-cold-bytes 8 GiB`. A parameter, never a constant --
#: the next context length or a re-sized budget would otherwise be read against
#: the wrong cap and fail silently.
DEFAULT_CAP_GIB = 8.0

#: The fraction of the cap at which the host budget counts as saturated for the
#: corroborating trace comparison. The trace is a 10 s sample, so this is a
#: coarse "is it full" test, not the boundary.
CAP_FRACTION = 0.995


def ssd_mmap_by_tick(log_path: str) -> dict[int, int]:
    """`tick number -> ssd_mmap ms`, from the same lines `parse_rows` reads.

    `steady_filter.parse_line` keeps a fixed field list and drops the rest, so
    `ssd_mmap` is not on its rows; this reads that one segment without widening
    that module's schema for a single report. `is_standard` stays the only
    definition of "steady".
    """
    import re
    tick = re.compile(r"\[step-timing\] tick (\d+) ")
    kv = re.compile(r"(\w+)=(-?\d+)ms")
    out: dict[int, int] = {}
    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = tick.search(line)
            if not m:
                continue
            segs = dict(kv.findall(line))
            if "ssd_mmap" in segs:
                out[int(m.group(1))] = int(segs["ssd_mmap"])
    return out


def _tick_rows(rows: list[dict], ssd_mmap: dict[int, int]) -> list[dict]:
    """Steady ticks, carrying the one extra segment this report reads."""
    return [{**r, "ssd_mmap": ssd_mmap.get(r["n"], 0)} for r in rows if is_standard(r)]


def boundary_tick(ticks: list[dict]) -> int | None:
    """First steady tick with `ssd_mmap > 0` -- the phase boundary in tick space."""
    return next((t["n"] for t in ticks if t["ssd_mmap"] > 0), None)


def decfwd_identity(trace: list[dict], ticks: list[dict], slack: int = 2) -> dict:
    """Is the trace→tick bridge sound for this pair of files?

    ``cum_steady_ticks == Δdecode_forwards`` over the trace's span. If this does
    not hold, a phase boundary cannot be placed on a tick and the report must say
    so rather than placing it anyway.
    """
    base = trace[0]["decfwd"]
    last = trace[-1]["decfwd"]
    inferred = base + len(ticks)
    return {
        "base_decfwd": base,
        "trace_last_decfwd": last,
        "steady_ticks": len(ticks),
        "inferred_last": inferred,
        "diff": last - inferred,
        "ok": abs(last - inferred) <= slack,
    }


def phase_boundary_from_trace(trace: list[dict], cap_gib: float) -> dict:
    """The corroborating trace reading: when did `shared` reach the cap?

    Reported, not used as the boundary. Returns the sample and the two samples
    it fell between, which is what makes the 10 s sampling visible.
    """
    hit = next((r for r in trace if (r["shared"] or 0.0) >= CAP_FRACTION * cap_gib), None)
    if hit is None:
        return {"saturated": False}
    before = next((r for r in reversed(trace) if r["ts"] < hit["ts"]), None)
    return {
        "saturated": True,
        "sample_ts": hit["ts"],
        "sample_shared_gib": round(hit["shared"], 3),
        "sample_decfwd": hit["decfwd"],
        "between_finished": (before["finished"] if before else None, hit["finished"]),
    }


def cut_from_decfwd(trace: list[dict], ticks: list[dict], want: float) -> list[dict]:
    """Ticks at or after the steady index the trace's `decfwd` names.

    `--from-decfwd` and `--from-line` are two spellings of "start here"; the
    conversion runs through the `decfwd` identity, which is checked first.
    """
    base = trace[0]["decfwd"]
    k = int(want - base)
    if k < 0:
        raise ValueError(
            f"--from-decfwd {want} precedes the trace's first decfwd {base}; "
            "the boundary is outside this pair of files"
        )
    return [t for t in ticks if t["n"] >= ticks[min(k, len(ticks) - 1)]["n"]]


def report(ticks: list[dict], lo: int | None, hi: int | None, tail_ms: int) -> dict:
    """One phase's tick stats. The long tail is reported apart from the body --
    at these sample counts a quantile cut separates nothing, so the split is the
    absolute `tail_ms` threshold the rest of the tree uses."""
    sub = [t for t in ticks if (lo is None or t["n"] >= lo) and (hi is None or t["n"] < hi)]
    if not sub:
        return {"n": 0}
    body = [t for t in sub if t["total"] <= tail_ms]
    return {
        "n": len(sub),
        "body_n": len(body),
        "total_p50_ms": median([t["total"] for t in body]),
        "total_p90_ms": pct([t["total"] for t in body], 0.9),
        "model_p50_ms": median([t["model"] for t in sub]),
        "total_max_ms": max(t["total"] for t in sub),
        "ssd_mmap_nonzero": len([t for t in sub if t["ssd_mmap"] > 0]),
        "ssd_mmap_p50_ms": median([t["ssd_mmap"] for t in sub if t["ssd_mmap"] > 0]),
        "tok_s": round(1000 / statistics.median([t["total"] for t in body]), 2) if body else None,
    }


def analyse(trace_path: str, log_path: str, cap_gib: float = DEFAULT_CAP_GIB,
            tail_ms: int = TAIL_MS, from_line: int | None = None,
            from_decfwd: float | None = None, log_rows: list[dict] | None = None) -> dict:
    """The whole report as a dict. `log_rows` is a test seam (pre-parsed ticks)."""
    if from_line is not None and from_decfwd is not None:
        raise ValueError("--from-line and --from-decfwd are mutually exclusive")
    trace = trace_rows(trace_path)
    raw = log_rows if log_rows is not None else parse_rows(log_path)
    ticks = _tick_rows(raw, ssd_mmap_by_tick(log_path))
    if not ticks:
        raise ValueError(f"{log_path}: no steady ticks (the standard set matched nothing)")

    ident = decfwd_identity(trace, ticks)
    ssd_first = boundary_tick(ticks)
    trace_hit = phase_boundary_from_trace(trace, cap_gib)

    if from_decfwd is not None and not ident["ok"]:
        # The conversion bridge is `decfwd`, so a failed identity means the
        # requested cut cannot be placed. Placing it anyway is how a boundary
        # lands inside the wrong request.
        raise ValueError(
            f"--from-decfwd cannot be placed: the decfwd identity does not hold for "
            f"this pair (base {ident['base_decfwd']} + {ident['steady_ticks']} steady "
            f"ticks = {ident['inferred_last']}, trace ends at {ident['trace_last_decfwd']}, "
            f"diff {ident['diff']})"
        )
    if from_line is not None:
        ticks = [t for t in ticks if t["n"] >= from_line]
    elif from_decfwd is not None:
        ticks = cut_from_decfwd(trace, ticks, from_decfwd)
        ssd_first = boundary_tick(ticks)

    total, complete = known_total(trace[-1])
    return {
        "trace": trace_path,
        "log": log_path,
        "trace_format": trace[0]["format"],
        "trace_samples": len(trace),
        "cap_gib": cap_gib,
        "tail_ms": tail_ms,
        # The trace's values are ALREADY GiB (A2's `shared_ssd=59.501` is the
        # /health `kv_cold_shared_ssd_bytes` 63888556032 B = 59.501 GiB), so no
        # byte conversion happens here. The four-key total only when every term
        # was recorded: on a legacy trace two keys are absent because the sampler
        # did not record them, and a partial sum printed as "the total" is how
        # 8.0 GiB stands in for 67.5.
        "last_total_gib": (
            round(trace[-1]["total"], 3)
            if complete and trace[-1]["total"] is not None else None
        ),
        "last_total_partial_gib": None if complete else round(total, 3),
        "last_total_complete": complete,
        "unrecorded_keys": trace[0]["unrecorded"],
        "decfwd_identity": ident,
        "boundary_tick": ssd_first,
        "boundary_source": "first steady tick with ssd_mmap > 0",
        "trace_corroboration": trace_hit,
        "phase_a": report(ticks, None, ssd_first, tail_ms),
        "phase_b": report(ticks, ssd_first, None, tail_ms),
    }


def _self_check() -> int:
    """Synthetic trace + log. The controls that matter are the ones that fail.

    The boundary is placed by `decfwd`, so the decisive control is the wrong
    placement: slicing by SAMPLE INDEX (what the first version did) must land
    inside a different request. Both placements are computed here and asserted to
    differ, so the test cannot pass by the two coinciding.
    """
    import tempfile

    # A trace whose host saturates at sample 4 of 6, and a log whose first
    # ssd_mmap tick is tick 40. decfwd makes the two agree by construction: the
    # trace starts at decfwd 10 and ends at 65, and the log carries exactly 55
    # steady ticks, so `cum_steady_ticks == Δdecode_forwards` (10 + 55 == 65).
    trace_lines = [
        "1000 priv=0.000 ssd=0.000 shared=1.000 shared_ssd=0.000 TOTAL=1.000 fin=1 decfwd=10 tok=20",
        "1010 priv=0.000 ssd=0.000 shared=3.000 shared_ssd=0.000 TOTAL=3.000 fin=1 decfwd=10 tok=20",
        "1020 priv=0.000 ssd=0.000 shared=6.000 shared_ssd=0.000 TOTAL=6.000 fin=1 decfwd=25 tok=50",
        "1030 priv=0.000 ssd=0.000 shared=8.000 shared_ssd=0.000 TOTAL=8.000 fin=2 decfwd=40 tok=80",
        "1040 priv=0.000 ssd=0.000 shared=8.000 shared_ssd=1.000 TOTAL=9.000 fin=2 decfwd=40 tok=80",
        "1050 priv=0.000 ssd=0.000 shared=8.000 shared_ssd=2.000 TOTAL=10.000 fin=3 decfwd=65 tok=110",
    ]

    def tick(n, total, ssd_mmap=0, model=None):
        model = total - 16 if model is None else model
        return (f"[step-timing] tick {n} total={total}ms dec=1 pre=0 model={model}ms "
                f"sample=3ms ssd_mmap={ssd_mmap}ms path=eager sparse=1")

    # 39 ticks with no ssd_mmap, then tick 40 onward with it.
    log_lines = [tick(n, 100) for n in range(1, 40)] + [tick(n, 110, ssd_mmap=120) for n in range(40, 56)]

    with tempfile.TemporaryDirectory() as d:
        tp, lp = Path(d) / "t.txt", Path(d) / "l.log"
        tp.write_text("\n".join(trace_lines) + "\n")
        lp.write_text("\n".join(log_lines) + "\n")

        # The parser must carry `ssd_mmap`; steady_filter's row schema does not,
        # so re-read the raw segment line for it. Assert that plumbing works.
        raw = parse_rows(str(lp))
        assert raw and len(raw) == 55, len(raw)
        mmap = ssd_mmap_by_tick(str(lp))
        assert len(mmap) == 55, len(mmap)
        # The fixture writes ssd_mmap=0 for ticks 1..39 and 120 for 40..55, so the
        # split below is what makes "boundary = first NONZERO" load-bearing: a
        # `>= 0` test would put the boundary at tick 1.
        assert set(n for n, v in mmap.items() if v > 0) == set(range(40, 56)), mmap
        ticks = _tick_rows(raw, mmap)
        assert len(ticks) == 55, len(ticks)
        assert boundary_tick(ticks) == 40, boundary_tick(ticks)

        # The decfwd bridge: base 10, so steady tick #k <-> decfwd 10+k.
        trace = trace_rows(str(tp))
        ident = decfwd_identity(trace, ticks)
        assert ident["ok"], ident

        # `--from-decfwd` is a DIFFERENT unit from `--from-line`, and this fixture
        # makes that visible: the trace's first sample is already at decfwd 10, so
        # decfwd 40 is the 30th decode forward after the base -> steady index 30 ->
        # tick 31, not tick 40. Asserting the two spellings coincide would encode
        # the sample-index confusion this whole file exists to avoid.
        k = 40 - int(trace[0]["decfwd"])              # 30
        a = cut_from_decfwd(trace, ticks, 40)
        assert a[0]["n"] == ticks[k]["n"] == 31, (a[0]["n"], ticks[k]["n"])
        assert a[0]["n"] != 40, "decfwd was read as a tick number"

        # THE control: the wrong placement. Slicing by trace SAMPLE index (the
        # first version's bug) puts the boundary at the sample's POSITION, not at
        # the forward it names. Trace sample index 3 names decfwd 40; slicing by
        # that index would cut at tick 4.
        by_sample_index = ticks[3]["n"]
        assert by_sample_index == 4, by_sample_index
        assert by_sample_index != a[0]["n"], (
            "the sample-index placement coincided with the decfwd placement, so "
            "this control proves nothing"
        )
        assert by_sample_index != boundary_tick(ticks)

        # The boundary MOVES LATER as the cap grows: at a 16 GiB cap the trace
        # never saturates, and the corroboration says so instead of inventing one.
        hi_cap = phase_boundary_from_trace(trace, 16.0)
        assert hi_cap["saturated"] is False, hi_cap
        lo_cap = phase_boundary_from_trace(trace, 8.0)
        assert lo_cap["saturated"] is True, lo_cap

        # Mutual exclusion, and the identity gate that makes `--from-decfwd`
        # trustworthy: a pair whose identity fails cannot place a decfwd cut.
        try:
            analyse(str(tp), str(lp), from_line=1, from_decfwd=10)
        except ValueError as exc:
            assert "mutually exclusive" in str(exc), exc
        else:
            raise AssertionError("both cut flags were accepted")

        # An identity that does NOT hold must refuse the decfwd cut rather than
        # placing it. Build a mismatch: the same log against a trace whose decfwd
        # span is far too short.
        short = Path(d) / "short.txt"
        short.write_text(
            "1000 priv=0.0 ssd=0.0 shared=1.0 shared_ssd=0.0 TOTAL=1.0 fin=1 decfwd=10 tok=20\n")
        try:
            analyse(str(short), str(lp), from_decfwd=12)
        except ValueError as exc:
            assert "decfwd identity does not hold" in str(exc), exc
        else:
            raise AssertionError("a decfwd cut was placed on a failed identity")

        # A legacy trace reports a PARTIAL total, never a number that reads as the
        # tier's occupancy. The unrecorded `shared_ssd` is the difference between
        # 8.0 GiB (what the recorded keys sum to) and the 67.5 GiB A1 really held.
        lp2 = Path(d) / "legacy.txt"
        lp2.write_text(
            "1000 cold=0.000 coldssd=0.000 shared=7.999 finished=1 decfwd=10 tok=20\n")
        rep = analyse(str(lp2), str(lp), cap_gib=8.0)
        assert rep["last_total_complete"] is False, rep
        assert rep["last_total_gib"] is None, rep
        # 7.999, computed from the keys the legacy sampler DID record -- and it
        # equals the host cap, which is exactly why it would not be questioned.
        assert rep["last_total_partial_gib"] == 7.999, rep
        assert rep["unrecorded_keys"] == ["shared_ssd", "total"], rep

        # The report itself, end to end, with the phase split by the boundary.
        rep2 = analyse(str(tp), str(lp), cap_gib=8.0)
        assert rep2["boundary_tick"] == 40, rep2
        assert rep2["phase_a"]["n"] == 39 and rep2["phase_b"]["n"] == 16, rep2
        assert rep2["phase_a"]["ssd_mmap_nonzero"] == 0, rep2
        assert rep2["phase_b"]["ssd_mmap_nonzero"] == 16, rep2

        # A trace whose host never saturates must not produce a boundary from the
        # trace comparison -- and the log boundary still stands on its own.
        try:
            cut_from_decfwd(trace, ticks, 5)
        except ValueError as exc:
            assert "precedes" in str(exc), exc
        else:
            raise AssertionError("a decfwd before the trace start was accepted")

        # Empty steady set is an error, not an empty report.
        bares = Path(d) / "bare.log"
        bares.write_text("[step-timing] tick 1 total=30ms dec=0 pre=0 model=0ms sample=0ms path=graph sparse=0\n")
        try:
            analyse(str(tp), str(bares))
        except ValueError as exc:
            assert "no steady ticks" in str(exc), exc
        else:
            raise AssertionError("an empty steady set produced a report")

    print("phase_report: decfwd-placed boundary, cap moves it, partial totals said OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", help="cold_trace file (either format)")
    ap.add_argument("--log", help="the arm's serve log")
    ap.add_argument("--cap-gib", type=float, default=DEFAULT_CAP_GIB,
                    help="the host cold budget this arm ran with (--kv-cold-bytes)")
    ap.add_argument("--tail-ms", type=int, default=TAIL_MS)
    ap.add_argument("--from-line", type=int,
                    help="start at this tick number (mutually exclusive with --from-decfwd)")
    ap.add_argument("--from-decfwd", type=float,
                    help="start at the tick the trace's decfwd names (converted via the "
                         "decfwd identity)")
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check or not (a.trace and a.log):
        rc = _self_check()
        assert rc == 0
        return rc
    try:
        rep = analyse(a.trace, a.log, a.cap_gib, a.tail_ms, a.from_line, a.from_decfwd)
    except (ColdTraceError, ValueError) as exc:
        print(f"phase_report: {exc}", file=sys.stderr)
        return 1
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv or len(sys.argv) == 1:
        rc = _self_check()
        assert rc == 0
        raise SystemExit(rc)
    raise SystemExit(main())
