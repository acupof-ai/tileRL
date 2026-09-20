#!/usr/bin/env python3
"""Cross-check the V100 close/cap device window's prose claims against the
vendored artifacts. READ-ONLY: it never runs a server and never produces a
measurement; it recomputes the numbers reviewers kept catching by hand —
hand-picked medians, decimal/binary unit muddles, percentages with no counter,
and headline figures with no attachment — and exits non-zero if the markdown
and the artifacts disagree.

Inputs (all committed alongside the entries):
- wins/bg-publish-device-2026-09-20/bg-close-ticks.tsv  per-tick close segments
- wins/bg-publish-device-2026-09-20/bg{0,1,2}.json       arm aggregates
- wins/bg-publish-device-2026-09-20/bg{1,2}-{follower,cancel}.json
- wins/bg-publish-device-2026-09-20/bg{1,2}-sizes.txt
- wins/close-batch-cap-device-2026-09-19/cb2-sizes.txt

Pure functions are unit-tested in _self_check with synthetic rows; the
file-driven run is the CI gate. Default (no args) and --self-check both run the
hermetic self-check so the repo-wide script closure gate stays green.
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG_DIR = os.path.join(ROOT, "docs", "experience", "wins", "bg-publish-device-2026-09-20")
CB_DIR = os.path.join(ROOT, "docs", "experience", "wins", "close-batch-cap-device-2026-09-19")
WINS_MD = os.path.join(ROOT, "docs", "experience", "wins", "2026-09-20-background-close-publish.md")

#: GiB is 2**30 everywhere. A decimal-GB figure beside a 2**30 artifact is the
#: exact unit class of error this gate exists to catch.
GIB = 2**30
#: Fraction tolerance between a prose percentage and the counter-derived one.
PCT_TOL = 0.05  # percent points (the table rounds 34.50 to 34.5)


class CrosscheckFailure(AssertionError):
    """A prose/artifact mismatch the gate must block on."""


# --------------------------------------------------------------------------- #
# pure cores (no fs)
# --------------------------------------------------------------------------- #
def parse_close_ticks(rows: list[str]) -> dict[str, list[dict[str, int]]]:
    """Parse the TSV body (header + comment lines already stripped). Returns
    arm -> list of per-tick int dicts keyed by the TSV column names."""
    hdr = rows[0].rstrip("\n").split("\t")
    out: dict[str, list[dict[str, int]]] = {}
    for line in rows[1:]:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        c = line.split("\t")
        assert len(c) == len(hdr), f"bad TSV width: {line!r}"
        arm = c[0]
        rec = {hdr[i]: int(c[i]) for i in range(2, len(c))}
        rec["tick"] = int(c[1])
        out.setdefault(arm, []).append(rec)
    return out


def arm_median(rows: list[dict[str, int]], col: str) -> float:
    return float(st.median(r[col] for r in rows))


def arm_range(rows: list[dict[str, int]], col: str) -> tuple[int, int]:
    vals = [r[col] for r in rows]
    return min(vals), max(vals)


def degraded_pct(degraded: int, queued: int) -> float:
    """Inline-fallback share: degraded / (queued + degraded), the two mutually
    exclusive terminal counters. Pure ratio; the doc rounds to one decimal."""
    tot = queued + degraded
    return round(100.0 * degraded / tot, 2) if tot else 0.0


def busy_host_split(wall_ms: float, dev_ms) -> float | None:
    """close_host = close_wall - close_dev; None for a pending device read."""
    if dev_ms is None:
        return None
    return round(wall_ms - dev_ms, 2)


def size_summary_gib(apparent_bytes: list[int]) -> dict[str, float]:
    """Single-operand (apparent bytes), 2**30 GiB summary of a size timeline.
    reclaim = peak - last; shrank only when last is strictly below peak."""
    nz = [b for b in apparent_bytes if b > 0]
    peak = max(apparent_bytes)
    return {
        "peak_gib": round(peak / GIB, 3),
        "last_gib": round(apparent_bytes[-1] / GIB, 3),
        "reclaim_gib": round((peak - apparent_bytes[-1]) / GIB, 3),
        "shrank": bool(nz) and apparent_bytes[-1] < peak,
    }


def parse_size_line(line: str) -> int | None:
    """`HH:MM:SS apparent=<bytes> physMiB=<m> logical_ssd=<b>` -> apparent bytes.
    Reads ONLY apparent so a series cannot silently mix operands; returns None
    for a malformed line (the caller decides whether that is fatal)."""
    for tok in line.split():
        if tok.startswith("apparent="):
            return int(tok.split("=", 1)[1])
    return None


# --------------------------------------------------------------------------- #
# file-driven checks
# --------------------------------------------------------------------------- #
def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _parse_prose_table(md: str) -> dict[str, str]:
    """Extract the three arm result rows from the device table by arm token,
    returning arm -> raw markdown cell string (bold/parens kept). We assert on
    the numeric LEAD of each cell, which is the figure reviewers quote."""
    rows = {}
    for line in md.splitlines():
        if line.startswith("| bg0 "):
            rows["bg0"] = line
        elif line.startswith("| bg1 "):
            rows["bg1"] = line
        elif line.startswith("| bg2 "):
            rows["bg2"] = line
    return rows


def _lead_int(cell: str) -> int:
    """First integer in a markdown table cell (strips **bold** and
    parenthesised ranges), e.g. '**1036** (987–**6478**)' -> 1036."""
    digits = ""
    for ch in cell:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits)


def check_close_ticks_vs_prose(ticks: dict, md: str) -> list[str]:
    errs = []
    table = _parse_prose_table(md)
    # column index in the markdown result table:
    # 0 arm | 1 release_close | 2 ssd_mmap | 3 cold_transfer | 4 frame | 5 degraded
    # prose column -> TSV column (the TSV uses short names).
    expected = ("release_close", "ssd_mmap", "cold_transfer", "frame_d2h")
    expected_cols = {"bg0": expected, "bg1": expected, "bg2": expected}
    for arm in ("bg0", "bg1", "bg2"):
        if arm not in table:
            errs.append(f"{arm}: missing prose table row")
            continue
        cells = [c.strip() for c in table[arm].split("|")]
        cells = [c for c in cells if c != ""]
        rows = ticks.get(arm, [])
        if not rows:
            errs.append(f"{arm}: no TSV rows")
            continue
        # n is embedded as "(n=6)"; assert it equals the TSV tail-tick count.
        import re

        m = re.search(r"n=(\d+)", cells[0])
        if m and int(m.group(1)) != len(rows):
            errs.append(f"{arm}: prose n={m.group(1)} != TSV rows {len(rows)}")
        for idx, col in zip((1, 2, 3, 4), expected_cols[arm]):
            got = _lead_int(cells[idx])
            want = arm_median(rows, col)
            # .5 medians are written either as integer or with .5; compare with
            # a half-unit tolerance and a direct string fallback.
            cell = cells[idx]
            ok = abs(got - want) <= 0.5
            if not ok and (f"{want:.1f}" in cell or f"{int(want)}" in cell):
                ok = True
            if not ok:
                errs.append(f"{arm} {col}: prose lead {got} != TSV median {want}")
    return errs


def check_degraded_ratio() -> list[str]:
    errs = []
    # bg1 cancel snapshot is the largest cumulative count (6473 degraded total).
    cancel = _load_json(os.path.join(BG_DIR, "bg1-cancel.json"))
    deg = cancel["after_publish"]["kv_cold_bg_degraded"]
    queued = cancel["after_publish"]["kv_cold_bg_queued"]
    pct = degraded_pct(deg, queued)
    if abs(pct - 34.50) > PCT_TOL:
        errs.append(f"bg1 degraded pct counter-derived {pct} != ~34.50 (deg={deg} queued={queued})")
    # bg2 must be exactly zero inline at the big queue.
    for kind in ("follower", "cancel"):
        d = _load_json(os.path.join(BG_DIR, f"bg2-{kind}.json"))
        snaps = [
            d.get(k)
            for k in (
                "after_publish",
                "health_after_publisher",
                "after_cancel",
                "health_after_follower",
            )
            if isinstance(d.get(k), dict)
        ]
        for s in snaps:
            if s.get("kv_cold_bg_degraded") != 0:
                errs.append(
                    f"bg2 {kind}: degraded={s.get('kv_cold_bg_degraded')} but doc claims 0% inline"
                )
    return errs


def check_size_timelines() -> list[str]:
    errs = []

    def read_apparent(path: str) -> list[int]:
        vals = []
        with open(path) as fh:
            for line in fh:
                v = parse_size_line(line)
                if v is None:
                    errs.append(f"{os.path.basename(path)}: unparsable size line {line.strip()!r}")
                else:
                    vals.append(v)
        return vals

    # bg2 timeline (bg publisher, cap off): proves bytes keep landing after the
    # request returns -> the series must be non-decreasing overall (worker
    # appends; no reclaim flag), giving an honest peak.
    bg2 = read_apparent(os.path.join(BG_DIR, "bg2-sizes.txt"))
    if bg2:
        s = size_summary_gib(bg2)
        if s["reclaim_gib"] != 0.0 or s["shrank"]:
            errs.append(
                f"bg2 sizes: non-cap timeline reclaimed {s['reclaim_gib']} "
                "GiB; expected append-only high water"
            )
    # cb2 timeline (#740 cap on): pinned at 8192 MiB physical. The peak apparent
    # must be ~8 GiB and never exceed the cap by more than one stride slack.
    cb2 = read_apparent(os.path.join(CB_DIR, "cb2-sizes.txt"))
    if cb2:
        peak_gib = max(cb2) / GIB
        if peak_gib > 8.6:  # 8 GiB cap + one extent/header slack
            errs.append(f"cb2 apparent peak {peak_gib:.3f} GiB exceeds the 8 GiB cap")
        if peak_gib < 7.9:
            errs.append(f"cb2 apparent peak {peak_gib:.3f} GiB never reached the cap")
    return errs


def check_arm_json_consistency() -> list[str]:
    """Internal invariants of the arm aggregate JSON (independent of prose)."""
    errs = []
    for name in ("bg0", "bg1", "bg2"):
        d = _load_json(os.path.join(BG_DIR, f"{name}.json"))
        for rep in d["reps"]:
            t = rep["ticks"]
            if not (t["p50_ms"] <= t["p90_ms"] <= t["max_ms"]):
                errs.append(f"{name} rep{rep['rep']}: p50/p90/max out of order")
            if not (0.0 <= t["frac_over_300"] <= 1.0):
                errs.append(f"{name} rep{rep['rep']}: frac_over_300 out of [0,1]")
            if t["decode_ticks"] <= 0:
                errs.append(f"{name} rep{rep['rep']}: no decode ticks")
            # effective tok/s is monotonic in accepted count at fixed decode time.
            if rep["spec_drafted_delta"] < rep["spec_accepted_delta"]:
                errs.append(f"{name} rep{rep['rep']}: accepted > drafted")
    return errs


def run_file_checks() -> list[str]:
    with open(WINS_MD) as fh:
        md = fh.read()
    tsv_path = os.path.join(BG_DIR, "bg-close-ticks.tsv")
    with open(tsv_path) as fh:
        raw = [line for line in fh.readlines() if not line.startswith("#")]
    ticks = parse_close_ticks(raw)
    errs = []
    errs += check_close_ticks_vs_prose(ticks, md)
    errs += check_degraded_ratio()
    errs += check_size_timelines()
    errs += check_arm_json_consistency()
    return errs


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
def _self_check() -> int:
    # TSV parse + median/range
    hdr = (
        "arm\ttick\ttotal\tmodel\tfinalize\tsample\tssd_mmap\t"
        "cold_transfer\tbounds_d2h\tdraft_clone\t"
        "frame_d2h\tshare_hold\trelease_close\t"
        "fwd_gpu\tsync_streams"
    )
    body = [
        hdr,
        # release_close 1000/2000 -> med 1500; ssd 10/30 -> 20; frame 2/4 -> 3
        "bg0\t1\t1000\t0\t0\t0\t10\t1\t1\t1\t2\t0\t1000\t0\t0",
        "bg0\t2\t2000\t0\t0\t0\t30\t3\t3\t3\t4\t0\t2000\t0\t0",
        "bg1\t9\t5000\t0\t0\t0\t0\t0\t0\t0\t0\t0\t5000\t0\t0",
    ]
    ticks = parse_close_ticks(body)
    assert arm_median(ticks["bg0"], "ssd_mmap") == 20.0
    assert arm_median(ticks["bg0"], "frame_d2h") == 3.0
    assert arm_median(ticks["bg0"], "release_close") == 1500.0
    assert arm_range(ticks["bg0"], "release_close") == (1000, 2000)

    # degraded ratio: 6473/(12289+6473) = 34.50%
    assert abs(degraded_pct(6473, 12289) - 34.50) < 0.01
    assert degraded_pct(0, 100) == 0.0

    # busy/host split incl pending
    assert busy_host_split(500.0, 200.0) == 300.0
    assert busy_host_split(100.0, None) is None

    # size summary single operand; grow then truncate vs plateau
    grow = [0, 4 * GIB, 8 * GIB, 5 * GIB]
    g = size_summary_gib(grow)
    assert g == {"peak_gib": 8.0, "last_gib": 5.0, "reclaim_gib": 3.0, "shrank": True}, g
    plat = size_summary_gib([8 * GIB, 8 * GIB])
    assert plat["reclaim_gib"] == 0.0 and plat["shrank"] is False

    # size line parser reads apparent only (operand discipline)
    assert parse_size_line("01:02:03 apparent=1073741824 physMiB=0 logical_ssd=0") == GIB
    assert parse_size_line("garbage") is None

    # prose lead extraction
    assert _lead_int("**1036** (987–**6478**)") == 1036
    assert _lead_int("41.5 (36–265)") == 41

    # failure detection: a prose lead that disagrees with the TSV must be raised.
    # Build a full-width table (arm + 4 figures + degraded) with arm bg0 which
    # the TSV has with n=2; claim a wrong release median and wrong n.
    full_hdr = (
        "| arm | release_close_request | ssd_mmap | "
        "pub_cold_transfer | pub_frame_d2h | degraded |\n"
        "|---|---|---|---|---|---|\n"
        "| bg0 (n=9) | 9999 | 20 | 1 | 3 | — |\n"
    )
    errs = check_close_ticks_vs_prose(ticks, full_hdr)
    assert any("prose n=9 != TSV rows 2" in e for e in errs), errs
    assert any("bg0 release_close" in e for e in errs), errs
    # ... and the real committed artifacts must pass end to end.
    assert run_file_checks() == []
    print("probe_device_artifacts_crosscheck self-check ok")
    return 0


def main() -> int:
    if "--self-check" in sys.argv[1:]:
        return _self_check()
    errs = run_file_checks()
    if errs:
        print("DEVICE ARTIFACT CROSSCHECK FAILED:")
        for e in errs:
            print(f"  - {e}")
        return 1
    print("device artifact crosscheck ok: prose matches vendored JSON/TSV/sizes")
    return 0


if __name__ == "__main__":
    # hermetic closure gate: bare invocation runs the self-check and asserts 0.
    if {"--self-check", "selfcheck"} & set(sys.argv[1:]) or len(sys.argv) == 1:
        rc = _self_check()
        assert rc == 0
        # bare run also exercises the file gate, so a broken committed artifact
        # fails the no-arg invocation too (CI runs scripts through pytest).
        file_errs = run_file_checks()
        if file_errs:
            print("DEVICE ARTIFACT CROSSCHECK FAILED:")
            for e in file_errs:
                print(f"  - {e}")
            raise SystemExit(1)
        print("device artifact crosscheck ok: prose matches vendored JSON/TSV/sizes")
        raise SystemExit(0)
    raise SystemExit(main())
