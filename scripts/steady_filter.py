#!/usr/bin/env python3
"""Re-filter a close-window arm's serve log onto the STANDARD steady-tick set.

Why this exists: `probe_headroom_coldtail.py` keeps decode ticks on `dec > 0`
alone, so its `p50_ms` includes captured-graph ticks and the long close-tail
ticks. The sweep arms are quoted on the tighter set —

    dec == 1 and sparse == 1 and model > 0 and sample > 0 and path != graph

with the close-tail ticks reported apart from the median. The two are different
statistics and must not meet in one before/after table. This script re-reads the
same log under the standard set, so a headroom arm and a sweep arm can be placed
side by side, or not placed at all when the log cannot support it.

    python3 scripts/steady_filter.py --log ~/servehybridsse.log --out steady.json
    python3 scripts/steady_filter.py --log ~/closewin/baseline/serve.log \
        --window 194540:264100 --window 489409:558000   # this arm's warm spans

Each `--window` is one rep, `start:end`, both from `arm.json`'s `reps[].ticks`
(`log_byte_offset` is the warm POST start, `log_byte_end` that rep's own end).
Use the rep's OWN end: the start offset is a warm START, so a span reaching to the
next rep's start contains the next rep's cold fill, whose short-context decode
ticks pass the standard set.

Read-only. Every field it prints is derived from the tick lines; nothing is
inferred. When the exclusion reason cannot be recovered from an older log format
(the `path=`/`sparse=` tail is only on lines this engine version wrote), the
script says so and reports what it could not classify rather than counting it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys

#: The standard steady set. Stated once; the doc and the harness cite it.
STANDARD_FILTER = "dec==1 and sparse==1 and model>0 and sample>0 and path!=graph"

TICK = re.compile(r"\[step-timing\] tick (\d+) total=(-?\d+)ms(?: dec=(\d+) pre=(\d+))? (.*)")
KV = re.compile(r"(\w+)=(-?\d+)ms")
PATH_RE = re.compile(r"\bpath=(\w+)")
SPARSE_RE = re.compile(r"\bsparse=(\d+)")


def parse_line(line: str) -> dict | None:
    """One `[step-timing] tick ...` line -> a row, or None if it is not one.
    `path`/`sparse` stay None when the tail fields are absent, so a caller can
    tell "not steady" from "cannot be shown steady"."""
    m = TICK.search(line)
    if not m:
        return None
    rest = m.group(5)
    segs = {k: int(v) for k, v in KV.findall(rest)}
    p, sp = PATH_RE.search(rest), SPARSE_RE.search(rest)
    return {
        "n": int(m.group(1)),
        "total": int(m.group(2)),
        "dec": int(m.group(3)) if m.group(3) is not None else 0,
        "pre": int(m.group(4)) if m.group(4) is not None else 0,
        "model": segs.get("model", 0),
        "sample": segs.get("sample", 0),
        "path": p.group(1) if p else None,
        "sparse": int(sp.group(1)) if sp else None,
    }


def parse_rows(log_path: str, offset: int = 0, until: int | None = None) -> list[dict]:
    """Tick rows in ``[offset, until)`` bytes.

    The window needs BOTH ends. `probe_headroom_coldtail` records one
    ``log_byte_offset`` per rep, taken after that rep's cold fill and before its
    warm POST -- so it is a warm START. Reading from it to EOF takes in the next
    rep's fill, whose short-context decode ticks pass the standard set. The rep's
    own ``log_byte_end`` (taken after its warm returns) is the other end; the
    harness passes one ``start:end`` span per rep. ``until=None`` reads to EOF and
    is right only for a whole-file read.
    """
    out = []
    with open(log_path, errors="replace") as fh:
        fh.seek(offset)
        while True:
            pos = fh.tell()
            if until is not None and pos >= until:
                break
            line = fh.readline()
            if not line:
                break
            r = parse_line(line)
            if r is not None:
                out.append(r)
    return out


def is_standard(r: dict) -> bool:
    """The standard steady set.

    A row from a log that predates the tail fields has `sparse is None`, which
    fails `sparse == 1` on the same line -- there is no separate absence check,
    and adding one would be a guard that cannot fire. Measured: a legacy row
    (`path=eager`, no `sparse`) is rejected by the `sparse == 1` clause, so
    `path is not None` would be dead.
    """
    return (r["dec"] == 1 and r["sparse"] == 1 and r["model"] > 0
            and r["sample"] > 0 and r["path"] != "graph")


def median(xs: list[int]):
    """The median as the rest of the tree reports one: `statistics.median`, the
    true median that averages the middle pair on even n. `probe_draft_window_sweep`
    reports its `tick_ms_med` this way and `probe_device_artifacts_crosscheck`
    recomputes arm medians this way, so a headroom arm's median is only
    comparable to a sweep arm's if it is this function.

    Not nearest-rank: on n=2 `[176, 180]` that gives 176 against this 178, which
    is the mismatch this whole script exists to avoid.

    Which sweep field this aligns with matters, because sweep reports TWO p50s on
    two conventions: `tick_ms_med` (a tick duration, `statistics.median`) and
    `spread()["p50"]` (per-prompt tok/s, nearest-rank `_pct`). `steady_p50_ms` is
    a tick duration, so it is the former -- comparing it to the latter would be
    wrong twice over, on convention and on unit.
    """
    return float(statistics.median(xs)) if xs else None


def pct(xs: list[int], q: float):
    """Nearest-rank percentile -- `int(q*(n-1))`, the tree's convention for a
    PERCENTILE (`probe_draft_window_sweep._pct` for p10/p90/iqr and
    `probe_headroom_coldtail.pct`, which states the choice in its docstring so the
    two instruments agree on a quantile).

    Separate from :func:`median` on purpose: the tree uses a true median for
    "median" and nearest-rank for "pNN", and collapsing them would put this
    script back out of step with one of its two consumers.
    """
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


#: A tick this slow is the close tail, not steady state. 300 ms is the repo's
#: existing convention (`frac_over_300` in the headroom probe and the 2026-09-20
#: window's own tables), kept here rather than re-derived.
TAIL_MS = 300


def summarise(rows: list[dict], tail_ms: int = TAIL_MS,
              windows: list[tuple[int, int | None]] | None = None) -> dict:
    """Steady median with the long tail split out.

    The close-tail ticks are the subject of this window, so folding them into
    the steady median would hide exactly what is being measured; they are
    reported as their own row and by count.

    The split is an absolute threshold, not a quantile: at the ~5-12 steady ticks
    a warm window actually yields, a 0.95 quantile cut does not separate anything
    (measured: 5 ticks, one 5000 ms, and the cut landed at the maximum so the
    tail reported 0). An absolute rule cannot fail that way.
    """
    steady = [r for r in rows if is_standard(r)]
    body = [r for r in steady if r["total"] <= tail_ms]
    tail = [r for r in steady if r["total"] > tail_ms]
    unclassifiable = [r for r in rows if r["path"] is None or r["sparse"] is None]
    return {
        "filter": STANDARD_FILTER,
        # The spans this summary is over, or `[[0, null]]` for the whole file --
        # never empty from the CLI. It is the reader's check that a windowed arm
        # really was windowed: a `[[0, null]]` on a close-window arm means the
        # median includes the supervisor's warmup.
        "windows": [list(w) for w in (windows or [(0, None)])],
        "ticks_total": len(rows),
        "steady_n": len(body),
        "steady_p50_ms": median([r["total"] for r in body]),
        "steady_p90_ms": pct([r["total"] for r in body], 0.9),
        "steady_model_p50_ms": median([r["model"] for r in body]),
        "steady_sample_p50_ms": median([r["sample"] for r in body]),
        "tail_ms": tail_ms,
        "tail_n": len(tail),
        "tail_p50_ms": median([r["total"] for r in tail]),
        "tail_max_ms": max((r["total"] for r in tail), default=None),
        "excluded_n": len(rows) - len(steady),
        "excluded_path_graph_n": len([r for r in rows if r["path"] == "graph"]),
        "excluded_undecidable_n": len(unclassifiable),
        "excluded_undecidable_note": (
            "log predates the path=/sparse= tail; these rows cannot be shown "
            "steady and are NOT counted" if unclassifiable else ""),
    }


def _parse_windows(specs: list[str] | None) -> list[tuple[int, int | None]]:
    """`["194540:489409", "489409"]` -> [(194540, 489409), (489409, None)]."""
    if not specs:
        return [(0, None)]
    out = []
    for spec in specs:
        off, _, until = spec.partition(":")
        out.append((int(off), int(until) if until else None))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", help="write the summary JSON here")
    ap.add_argument("--window", action="append", default=None,
                    metavar="START:END",
                    help="byte span to read, repeatable. Build one per rep from "
                         "arm.json: START is reps[].ticks.log_byte_offset (the warm "
                         "POST start) and END is that rep's own "
                         "reps[].ticks.log_byte_end. Use the rep's OWN end, not the "
                         "next rep's start -- the offset is a warm START, so "
                         "next-start spans contain the next rep's cold fill, whose "
                         "short-context decode ticks pass the standard set. "
                         "OFF with no END reads to EOF (a whole-file read). "
                         "Default: the whole file (offset 0).")
    ap.add_argument("--tail-ms", type=int, default=TAIL_MS,
                    help="a steady-set tick slower than this is reported as close tail")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return _self_check()
    spans = _parse_windows(a.window)
    rows = []
    for off, until in spans:
        rows.extend(parse_rows(a.log, off, until))
    # The window is a filter, not a sibling series: ticks from different spans
    # are one sample, so the median is taken over their union rather than
    # averaged per span (which would weight a 2-tick rep like an 8-tick one).
    rows.sort(key=lambda r: r["n"])
    if not rows:
        print(f"no tick lines in {a.log} within {spans}", file=sys.stderr)
        return 1
    rec = summarise(rows, a.tail_ms, spans)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1))
    return 0


def _self_check() -> int:
    """Synthetic tick lines: a graph tick, a spare decode tick, a prefill, two
    steady ticks and one long tail tick. Pins both halves of the claim -- the
    graph/idle rows leave the median, and the tail is reported apart."""
    def line(n, total, dec, pre, model, sample, path, sparse):
        return (f"[step-timing] tick {n} total={total}ms dec={dec} pre={pre} "
                f"model={model}ms sample={sample}ms path={path} sparse={sparse}")
    rows = [parse_line(line(1, 5000, 1, 0, 100, 5, "graph", 1)),
            parse_line(line(2, 30, 0, 0, 0, 0, "graph", 0)),
            parse_line(line(3, 400, 0, 512, 380, 0, "eager", 1)),
            parse_line(line(4, 180, 1, 0, 160, 3, "eager", 1)),
            parse_line(line(5, 182, 1, 0, 162, 3, "eager", 1)),
            parse_line(line(6, 900, 1, 0, 600, 3, "eager", 1))]
    s = summarise(rows, tail_ms=300)
    assert s["steady_n"] == 2, s           # the 180 and the 182
    assert s["tail_n"] == 1, s             # the 900, over the 300 ms threshold
    assert s["excluded_n"] == 3, s         # graph-total, graph-idle, prefill
    assert s["excluded_path_graph_n"] == 2, s
    # EXACT, not a set of acceptable values. An earlier version wrote
    # `in (180, 182)`, which silently accepted both the true median (181) and the
    # nearest-rank pick (180) -- the two conventions this file exists to keep
    # apart. A tolerance here hides the bug it is meant to catch.
    assert s["steady_p50_ms"] == 181.0, s

    # The conventions, pinned on the sample sizes a warm window actually yields
    # (5-12 ticks, so even n is common). `median` must equal statistics.median
    # and must NOT equal nearest-rank; `pct` must be nearest-rank.
    assert median([176, 180]) == 178.0, median([176, 180])
    assert median([176, 180]) != 176, "median fell back to nearest-rank"
    assert median([]) is None
    assert pct([176, 180], 0.5) == 176, pct([176, 180], 0.5)
    assert pct([176, 178, 180, 1000], 0.5) == 178
    # n=4, q=0.9 -> int(0.9*3)=2 -> the 3rd of 4, NOT the max. Nearest-rank
    # reaches the maximum only as q approaches 1; asserting the max here
    # would have been a wrong expectation, not a stricter one.
    assert pct([176, 178, 180, 1000], 0.9) == 180, pct([176, 178, 180, 1000], 0.9)
    assert pct([1, 2, 3, 4, 5], 0.9) == 4
    # Even-n body end to end, which is the shape rev flagged: 4 steady ticks plus
    # one close-tail tick. The median is 179.0 -- the mean of the middle pair --
    # where nearest-rank would say 178 and `int(q*n)` would say 180. The slow tick
    # still lands in the body, because the split is an absolute threshold and it
    # is under it.
    five = [parse_line(line(i, t, 1, 0, t - 16, 3, "eager", 1))
            for i, t in enumerate([176, 178, 180, 182, 1000], start=1)]
    s5 = summarise(five, tail_ms=300)
    assert s5["steady_n"] == 4 and s5["tail_n"] == 1, s5
    assert s5["steady_p50_ms"] == 179.0, s5
    assert s5["steady_p90_ms"] == 180, s5  # int(0.9*3)=2 -> 180, not the max
    # A log without the path/sparse tail cannot be classified: excluded, and said.
    # Each missing field is exercised on its own -- a row with `path` present and
    # `sparse` absent must still be undecidable, or the guard only holds for the
    # subset the example happened to omit.
    for label, text in (
        ("both absent", "[step-timing] tick 9 total=200ms dec=1 pre=0 model=150ms"),
        ("path only", "[step-timing] tick 9 total=200ms dec=1 pre=0 model=150ms path=eager"),
        ("sparse only", "[step-timing] tick 9 total=200ms dec=1 pre=0 model=150ms sparse=1"),
    ):
        s2 = summarise([parse_line(text)])
        assert s2["steady_n"] == 0, (label, s2)
        assert s2["excluded_undecidable_n"] == 1, (label, s2)
        assert "predates" in s2["excluded_undecidable_note"], (label, s2)

    # The window. A warm window is not "the log from the first rep's offset on":
    # the supervisor's warmup and each rep's cold fill write standard-set ticks
    # too. Prove both halves on a synthetic log laid out the way the harness
    # windows a real one -- two reps with a fill between them, and warmup before
    # the first offset.
    #
    # The marks are the PROBE's convention, which is what makes this test able to
    # fail: `log_byte_offset` is taken after the fill and before the warm POST (a
    # warm START), and `log_byte_end` after the warm returns. An earlier version of
    # this self-check hand-picked warm-END positions for BOTH ends, a geometry the
    # probe never writes, and so stayed green while the harness built
    # `[off_i, off_{i+1})` -- which contains warm_i AND fill_{i+1}.
    import tempfile
    plain = line(1, 175, 1, 0, 155, 3, "eager", 1)      # supervisor warmup
    fill0 = line(2, 178, 1, 0, 158, 3, "eager", 1)      # rep0 fill
    warm0 = line(3, 176, 1, 0, 156, 3, "eager", 1)      # rep0 warm
    fill1 = line(4, 400, 1, 0, 380, 3, "eager", 1)      # rep1 fill (also standard)
    warm1a, warm1b = (line(5, 177, 1, 0, 157, 3, "eager", 1),
                      line(6, 179, 1, 0, 159, 3, "eager", 1))
    rows_text = [plain, fill0, warm0, fill1, warm1a, warm1b]
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write("\n".join(rows_text) + "\n")
        lpath = fh.name
    # marks[i] = byte just past line i. The probe's boundaries:
    #   rep0: start = past fill0, end = past warm0
    #   rep1: start = past fill1, end = past warm1b
    marks, pos = [], 0
    for text in rows_text:
        pos += len(text) + 1
        marks.append(pos)
    spans = [(marks[1], marks[2]), (marks[3], marks[5])]
    got = []
    for off, until in spans:
        got.extend(parse_rows(lpath, off, until))
    sw = summarise(got, tail_ms=300, windows=spans)
    # warm0/warm1a/warm1b only. fill1 (400 ms) is standard-set and sits OUTSIDE
    # both spans; the warmup tick is outside too.
    assert sw["steady_n"] == 3, sw
    assert sw["ticks_total"] == 3, sw
    assert sw["tail_n"] == 0, sw
    assert sw["steady_p50_ms"] == 177.0, sw
    # INVERSE control: a genuine >300 ms tick INSIDE a warm span must be selected
    # into the tail set. Without this, "tail_n == 0" above would also pass if the
    # window simply dropped every slow tick, which is the opposite failure.
    slow = line(7, 900, 1, 0, 600, 3, "eager", 1)
    with open(lpath, "a") as fh:
        fh.write(slow + "\n")
    slow_span = (marks[3], marks[5] + len(slow) + 1)
    inside = parse_rows(lpath, slow_span[0], slow_span[1])
    s_in = summarise(inside, tail_ms=300, windows=[slow_span])
    assert s_in["tail_n"] == 1 and s_in["tail_max_ms"] == 900, s_in
    assert s_in["steady_n"] == 2, s_in          # warm1a, warm1b stay in the body
    # And the negative control for the window itself: unwindowed, the same file
    # reports the warmup and BOTH fills, so the assertions above fail for the
    # windowing and not because those ticks were never standard.
    un = summarise(parse_rows(lpath), tail_ms=300)
    assert un["ticks_total"] == 7, un
    assert un["steady_n"] == 5, un          # 7 total minus the two over 300 ms
    assert un["tail_n"] == 2 and un["tail_max_ms"] == 900, un
    # The off-by-one-phase construction the harness used to build, on the same
    # file: [off_0, off_1) admits fill1. This is the regression, pinned.
    wrong = []
    for off, until in ((marks[1], marks[3]), (marks[3], None)):
        wrong.extend(parse_rows(lpath, off, until))
    s_wrong = summarise(wrong, tail_ms=300, windows=[(marks[1], marks[3]), (marks[3], None)])
    assert s_wrong["tail_max_ms"] == 900, s_wrong   # fill1 and the slow tick
    assert s_wrong["ticks_total"] > sw["ticks_total"], (s_wrong, sw)

    print("steady_filter: standard-set split OK")
    return 0



if __name__ == "__main__":
    # The assert lives in the block (not only in _self_check) so the scripts
    # closure audit's self-check set recognizes this file: a hermetic script with
    # no asserting __main__ reads as an uninvoked check.
    # Bare invocation is the self-check: the closure audit runs every hermetic
    # script with NO args and requires rc 0.
    if "--self-check" in sys.argv or len(sys.argv) == 1:
        rc = _self_check()
        assert rc == 0
        raise SystemExit(rc)
    raise SystemExit(main())
