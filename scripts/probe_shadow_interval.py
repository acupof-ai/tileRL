#!/usr/bin/env python3
"""PROBE-ONLY #805: explain an anomalous interval_off p50 in shadow v1.

The probe records graph-tick start-to-start gaps in OFF segments, but the
[step-timing] log (TILERL_STEP_TIMING_SLOW_MS=0 prints every step) has every
step's wall total. This parser reconstructs, offline:

  - every step in order: tick n, total ms, dec/pre, sparse, path;
  - graph decode ticks (dec=1 sparse=1 path=graph), enumerated so the probe's
    deterministic 50-tick off/on segments fall out as ordinal//50 (segments
    are pure graph-tick counts; non-graph steps sit between them);
  - for each graph tick in an OFF segment, its start-to-start gap proxy =
    total of the previous graph step plus every intervening step's total
    (the engine steps back-to-back, loop overhead is negligible);
  - that gap split by WHAT sits between them: plain (nothing), spans an eager
    sparse refresh tick (path != graph), or some other intervening step.

Why: an ordinary graph->graph gap is ~46 ms and 1-in-7/8 spans a ~249 ms eager
refresh, so a p50 near 46 cannot move to ~62 from refresh gaps alone (they are
the upper ~1/8). A p50 of 61.77 means >half the gaps were elevated, which has
to show as either elevated plain gaps or an extra class of intervening steps.

Usage: probe_shadow_interval.py shadow_v1_*.dev.err [...]
Pure stdlib; prints one block per file.
"""

import re
import sys

# [step-timing] tick 12 total=47ms dec=1 pre=0 ... path=graph sparse=1 ...
LINE = re.compile(
    r"\[step-timing\] tick (\d+) total=(\d+)ms dec=(\d+) pre=(\d+).*?"
    r"path=(\S+) sparse=(\d)")

SEGMENT = 50


def parse(path):
    steps = []
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if m:
                n, total, dec, pre, fpath, sparse = m.groups()
                steps.append({"n": int(n), "ms": int(total),
                              "dec": int(dec), "pre": int(pre),
                              "path": fpath, "sparse": int(sparse)})
    return steps


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    i = min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1))))
    return s[i]


def analyze(path):
    steps = parse(path)
    # graph decode step indices in the full step stream
    graph_idx = [i for i, s in enumerate(steps)
                 if s["dec"] and s["sparse"] and s["path"] == "graph"]
    plain, refresh, other = [], [], []
    per_seg = {}
    for k in range(1, len(graph_idx)):
        i0, i1 = graph_idx[k - 1], graph_idx[k]
        ordinal = k  # current graph tick's 1-based ordinal among graph ticks
        seg = (ordinal - 1) // SEGMENT  # 1..50->0 off, 51..100->1 on, ...
        if seg % 2:
            continue  # ON segment
        gap = sum(steps[j]["ms"] for j in range(i0, i1))
        between = steps[i0 + 1:i1]
        eager = [s for s in between if s["sparse"] and s["path"] != "graph"]
        tag = "plain"
        if eager:
            tag = "refresh"
            refresh.append(gap)
        elif between:
            tag = "other"
            other.append((gap, [(s["n"], s["ms"], s["path"], s["sparse"],
                                 s["dec"], s["pre"]) for s in between]))
        else:
            plain.append(gap)
        per_seg.setdefault(seg, []).append((gap, tag))
    allg = sorted(plain + refresh + [g for g, _ in other])
    print(f"== {path}")
    print(f"   steps={len(steps)} graph_decode={len(graph_idx)}")
    for seg, vals in sorted(per_seg.items()):
        gs = [g for g, _ in vals]
        print(f"   off-seg {seg}: n={len(gs)} p50={pct(gs,50)} "
              f"p90={pct(gs,90)} min={min(gs)} max={max(gs)}")
    print(f"   OFF gaps total n={len(allg)} p50={pct(allg,50)} "
          f"p90={pct(allg,90)}")
    print(f"   plain   n={len(plain)} p50={pct(plain,50)} "
          f"p90={pct(plain,90)} max={max(plain) if plain else '-'}")
    print(f"   refresh n={len(refresh)} p50={pct(refresh,50)} "
          f"min={min(refresh) if refresh else '-'} "
          f"max={max(refresh) if refresh else '-'}")
    if other:
        gaps = sorted(g for g, _ in other)
        print(f"   OTHER   n={len(other)} gap p50={pct(gaps,50)} "
              f"max={max(gaps)}; examples:")
        for g, b in other[:5]:
            print(f"     gap={g} between={b}")
    if not other:
        print("   OTHER   n=0")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
