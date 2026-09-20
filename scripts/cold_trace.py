#!/usr/bin/env python3
"""Read a cold-tier occupancy trace into four-key rows, both on-disk formats.

The trace is one line per sample, written by the sampler that runs alongside a
serve arm:

    A1 (legacy)  ts cold= coldssd= shared= finished= decfwd= tok= hits= miss=
    A2 (current) ts priv= ssd= shared= shared_ssd= TOTAL= fin= decfwd= tok=

Three things about these lines decide the design, and each was measured on the
vendored artifacts rather than assumed:

1. **The two formats are NOT the same key set with different names.** Only
   ``shared`` and ``decfwd`` appear in both. ``priv``/``ssd`` are the renamed
   ``cold``/``coldssd`` (recorded in both), ``finished`` was shortened to
   ``fin``, and ``shared_ssd``/``TOTAL`` **do not exist in the legacy format at
   all** — the sampler did not yet record them.

2. **A missing key in the legacy format is UNRECORDED, not zero.** This is the
   whole reason this module does not default to 0. On A1 the serve's own
   `/health` reports ``kv_cold_shared_ssd_bytes = 59.501 GiB``, while its trace
   carries no ``shared_ssd`` key: the tier really held 59.501 GiB and the trace
   simply did not sample it. Summing the trace's four keys under "missing = 0"
   gives **8.0 GiB against a true 67.5 GiB** — and 8.0 is a perfectly plausible
   number, so nothing downstream would question it. Those two fields come back
   as ``None`` and the row is tagged ``legacy``; only a caller doing arithmetic
   decides what an unrecorded term means, and it has to say so.

3. **`decfwd` is the only field in the same coordinate as the serve log's tick
   counter.** The identity is exact, not approximate: on A2 every one of the 30
   per-request boundaries satisfies ``cum_steady_ticks == Δdecode_forwards``
   (diff 0, totals 1028 == 1028). A phase boundary therefore maps onto a tick by
   that field — never by sample index, and never by line number. The first
   version of the analysis sliced on the sample index and cut inside the wrong
   request.

Read-only, pure stdlib. Errors are loud: a line whose format cannot be
identified, or that is missing a shared key, raises with the format name and
the key — an unreadable line and an empty arm must not look the same.

    python3 scripts/cold_trace.py --trace cold_trace_a2.txt
    python3 scripts/cold_trace.py --trace a1.txt --json out.json
    python3 scripts/cold_trace.py --self-check
"""

from __future__ import annotations

import argparse
import json
import sys

#: Keys both formats carry, under their own or an aliased name. A line missing
#: any of these is not a trace line we can place in tick space, whatever its
#: version, so they are REQUIRED.
#:
#: `priv` and `ssd` are on this list, not on the unrecorded one: BOTH formats
#: record them, the legacy one under `cold`/`coldssd`. A parser that only knows
#: the current names and falls back to 0 reads EVERY key of a legacy line as 0 --
#: measured, that yields 7.999 GiB where the tier really held 67.500, and 7.999
#: is exactly the host cap, i.e. it reads as "cold tier just filled, SSD not yet
#: in use". The alias map is a correctness requirement, not a convenience.
REQUIRED = ("shared", "priv", "ssd", "decfwd")

#: The two formats. `aliases` maps the on-disk key to the canonical field.
FORMATS = {
    "current": {
        "keys": {"priv", "ssd", "shared", "shared_ssd", "TOTAL", "fin", "decfwd", "tok"},
        "aliases": {"priv": "priv", "ssd": "ssd", "shared": "shared",
                    "shared_ssd": "shared_ssd", "TOTAL": "total",
                    "fin": "finished", "decfwd": "decfwd", "tok": "tok"},
        #: Fields this format does not sample. Empty: it records all four keys.
        "unrecorded": (),
    },
    "legacy": {
        "keys": {"cold", "coldssd", "shared", "finished", "decfwd", "tok", "hits", "miss"},
        "aliases": {"cold": "priv", "coldssd": "ssd", "shared": "shared",
                    "finished": "finished", "decfwd": "decfwd", "tok": "tok",
                    "hits": "hits", "miss": "miss"},
        #: Sampled by the current format but NOT by this one. They are None, never
        #: 0 -- see the module docstring: A1 really held 59.5 GiB of shared SSD
        #: bytes that its trace never wrote down.
        "unrecorded": ("shared_ssd", "total"),
    },
}

#: Every canonical field a row can carry.
FIELDS = ("ts", "priv", "ssd", "shared", "shared_ssd", "total",
          "finished", "decfwd", "tok", "hits", "miss")


class ColdTraceError(ValueError):
    """A trace line that cannot be read as either format."""


def detect_format(pairs: dict[str, str]) -> str:
    """Which format this line is in, from its key set alone.

    Decided by the keys that DISCRIMINATE: a line with any of the current-only
    keys is current; a line with any of the legacy-only keys is legacy. A line
    with both is ambiguous and raises -- that is a corrupt or hand-edited file,
    not a format.
    """
    cur_only = FORMATS["current"]["keys"] - FORMATS["legacy"]["keys"]
    leg_only = FORMATS["legacy"]["keys"] - FORMATS["current"]["keys"]
    has_cur, has_leg = bool(pairs.keys() & cur_only), bool(pairs.keys() & leg_only)
    if has_cur and has_leg:
        both = sorted(pairs.keys() & cur_only) + sorted(pairs.keys() & leg_only)
        raise ColdTraceError(
            f"ambiguous format: line carries both current-only and legacy-only keys {both}"
        )
    if has_cur:
        return "current"
    if has_leg:
        return "legacy"
    raise ColdTraceError(
        "unidentifiable format: line carries neither "
        f"{sorted(cur_only)} nor {sorted(leg_only)}"
    )


def parse_line(line: str, lineno: int = 0) -> dict | None:
    """One trace line -> a canonical row, or None if it is blank.

    Raises :class:`ColdTraceError` (naming the key) rather than returning an
    empty row: "this file cannot be read" and "this arm was empty" must not share
    an appearance.
    """
    parts = line.split()
    if not parts:
        return None
    pairs: dict[str, str] = {}
    for tok in parts[1:]:
        if "=" not in tok:
            raise ColdTraceError(f"line {lineno}: token {tok!r} is not key=value")
        k, _, v = tok.partition("=")
        if k in pairs:
            raise ColdTraceError(f"line {lineno}: duplicate key {k!r}")
        pairs[k] = v
    if not pairs:
        return None
    fmt = detect_format(pairs)
    spec = FORMATS[fmt]
    row: dict = {"format": fmt, "lineno": lineno}
    try:
        row["ts"] = float(parts[0])
    except ValueError as exc:
        raise ColdTraceError(f"line {lineno}: leading field {parts[0]!r} is not a timestamp") from exc
    for disk_key, value in pairs.items():
        field = spec["aliases"].get(disk_key)
        if field is None:
            raise ColdTraceError(
                f"line {lineno}: key {disk_key!r} is not part of the {fmt} format"
            )
        try:
            row[field] = float(value)
        except ValueError as exc:
            raise ColdTraceError(
                f"line {lineno}: {disk_key}={value!r} is not a number"
            ) from exc
    # Required keys, after aliasing: `shared` and `decfwd` both exist in both
    # formats under their own names, so a line missing one is truncated.
    for key in REQUIRED:
        if key not in row:
            raise ColdTraceError(
                f"line {lineno}: {fmt} line is missing required key {key!r} "
                f"(have {sorted(pairs)})"
            )
    # Unrecorded fields are None, explicitly. Never 0.
    for field in FIELDS:
        if field not in row:
            row[field] = None
    row["unrecorded"] = [f for f in spec["unrecorded"]]
    return row


def rows(path: str) -> list[dict]:
    """Every trace line in ``path`` as a canonical row.

    A format change MID-FILE raises: the sampler appends by time, so a file can
    legitimately hold several arms, but it cannot hold two sampler versions
    spliced together without a hand edit.
    """
    out: list[dict] = []
    seen: str | None = None
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            row = parse_line(line, i)
            if row is None:
                continue
            if seen is None:
                seen = row["format"]
            elif row["format"] != seen:
                raise ColdTraceError(
                    f"line {i}: format changed mid-file ({seen} -> {row['format']}); "
                    "a file holds one sampler version"
                )
            out.append(row)
    if not out:
        raise ColdTraceError(f"{path}: no trace lines (a non-empty file is required)")
    return out


def known_total(row: dict) -> tuple[float, bool]:
    """``(sum of the four keys that ARE recorded, complete?)``.

    The second element is False when a term is unrecorded, so a caller cannot
    print the number without also having the fact that it is partial. This is the
    function that would have returned 8.0 GiB on A1 had it defaulted to 0.
    """
    parts = [row.get(k) for k in ("priv", "ssd", "shared", "shared_ssd")]
    recorded = [p for p in parts if p is not None]
    return sum(recorded), len(recorded) == 4


def _self_check() -> int:
    """Synthetic lines. Three controls, and each must be able to fail.

    A1 and A2 are shaped after the vendored files: A1's legacy key set, A2's
    current one. The values are A1's real last sample where it matters.
    """
    a2 = "1789902815 priv=0.000 ssd=0.000 shared=7.999 shared_ssd=59.501 TOTAL=67.500 fin=36 decfwd=1049 tok=1968"
    a1 = "1789895383 cold=0.000 coldssd=0.000 shared=7.999 finished=39 decfwd=1086 tok=2113 hits=2 miss=39"

    r2 = parse_line(a2, 1)
    assert r2["format"] == "current", r2
    assert r2["total"] == 67.5 and r2["shared_ssd"] == 59.501, r2
    assert r2["priv"] == 0.0 and r2["ssd"] == 0.0, r2
    assert r2["finished"] == 36 and r2["decfwd"] == 1049, r2
    assert known_total(r2) == (67.5, True), known_total(r2)

    r1 = parse_line(a1, 1)
    assert r1["format"] == "legacy", r1
    # THE ALIAS MAP, on a SYNTHETIC row with a value that cannot be 0 by accident.
    # A real A1 line is the wrong fixture here: `cold`/`coldssd` are 0.000 on most
    # of them (that is the normal "private stays resident" state), so an assertion
    # of "non-zero" against a real line would be flaky and one of "present" would
    # pass even if the alias were dropped. Construct the input instead.
    alias_row = parse_line(
        "1789894551 cold=1.663 coldssd=2.5 shared=6.955 finished=23 decfwd=555 tok=1089", 1)
    assert alias_row["priv"] == 1.663, alias_row       # from `cold`
    assert alias_row["ssd"] == 2.5, alias_row          # from `coldssd`
    assert alias_row["shared"] == 6.955, alias_row
    # And the same fields through the current names.
    cur_row = parse_line(
        "1 priv=1.663 ssd=2.5 shared=6.955 shared_ssd=59.501 TOTAL=70.619 fin=23 decfwd=555", 1)
    assert (cur_row["priv"], cur_row["ssd"]) == (1.663, 2.5), cur_row
    # Both name the same field, so a legacy line and a current line with equal
    # values must agree on every shared field.
    assert alias_row["priv"] == cur_row["priv"] and alias_row["ssd"] == cur_row["ssd"]
    assert alias_row["shared"] == cur_row["shared"]
    assert alias_row["decfwd"] == cur_row["decfwd"]

    assert r1["finished"] == 39 and r1["decfwd"] == 1086, r1
    # The discriminator: unrecorded is None, NOT 0.
    assert r1["shared_ssd"] is None, r1
    assert r1["total"] is None, r1
    assert r1["unrecorded"] == ["shared_ssd", "total"], r1
    # THE assertion this module exists for. A1 really held 59.501 GiB of shared
    # SSD bytes (/health, a1_32k_n30.json health_after); under "missing = 0" the
    # four keys sum to 8.0 GiB against a true 67.5. If a future edit defaults the
    # unrecorded keys to 0, this line goes red -- and it is checked against the
    # naive sum, so the assertion cannot pass by accident:
    naive = r1["priv"] + r1["ssd"] + r1["shared"]
    total, complete = known_total(r1)
    assert not complete, "legacy row reported a COMPLETE total"
    assert total == naive == 7.999, (total, naive)
    assert total != 67.5, "the unrecorded term was silently counted"

    # Negative control: a truncated line (required key absent) must RAISE, naming
    # the key. `shared` and `decfwd` are the discriminators -- `TOTAL` is allowed
    # to be absent in the current format, so its absence discriminates nothing.
    for label, text, key in (
        ("missing shared", "1 priv=0.0 ssd=0.0 TOTAL=1.0 fin=1 decfwd=5 tok=1", "shared"),
        ("missing decfwd", "1 priv=0.0 ssd=0.0 shared=1.0 TOTAL=1.0 fin=1 tok=1", "decfwd"),
        ("legacy missing decfwd", "1 cold=0.0 coldssd=0.0 shared=1.0 finished=1 tok=1", "decfwd"),
    ):
        try:
            parse_line(text, 7)
        except ColdTraceError as exc:
            assert key in str(exc), (label, exc)
            assert "7" in str(exc), (label, exc)  # the line number is named
        else:
            raise AssertionError(f"{label}: no error raised")

    # A truncated line that still LOOKS legacy-complete must not be mis-filed as
    # legacy: it has current-only keys, so it is current, so it is missing a
    # required key -> raise. (This is why `TOTAL` alone cannot be the test.)
    try:
        parse_line("1 priv=0.0 ssd=0.0 shared=1.0 TOTAL=1.0 fin=1 tok=1", 3)
    except ColdTraceError as exc:
        assert "decfwd" in str(exc), exc
    else:
        raise AssertionError("a current-format line missing decfwd was accepted")

    # A complete current line must NOT be tagged legacy.
    assert parse_line(a2, 1)["format"] == "current"

    # Ambiguous / unidentifiable lines raise rather than guessing a format.
    for label, text in (
        ("both key sets", "1 priv=0.0 cold=0.0 shared=1.0 decfwd=5"),
        ("neither key set", "1 shared=1.0 decfwd=5"),
    ):
        try:
            parse_line(text, 2)
        except ColdTraceError as exc:
            assert "format" in str(exc), (label, exc)
        else:
            raise AssertionError(f"{label}: no error raised")

    # `rows()`: blank lines are skipped, a mid-file format change raises, and an
    # empty file is an error rather than an empty list.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(a2 + "\n\n" + a1 + "\n")
        mixed = fh.name
    try:
        rows(mixed)
    except ColdTraceError as exc:
        assert "mid-file" in str(exc), exc
    else:
        raise AssertionError("a mid-file format change was accepted")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(a2 + "\n" + a2 + "\n")
        same = fh.name
    got = rows(same)
    assert len(got) == 2 and all(r["format"] == "current" for r in got), got

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        empty = fh.name
    try:
        rows(empty)
    except ColdTraceError as exc:
        assert "no trace lines" in str(exc), exc
    else:
        raise AssertionError("an empty trace returned an empty list silently")

    import os
    for p in (mixed, same, empty):
        os.unlink(p)
    print("cold_trace: two-format parse, unrecorded!=0, loud errors OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", help="the cold_trace file to read")
    ap.add_argument("--json", help="write the parsed rows here")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check or not a.trace:
        rc = _self_check()
        assert rc == 0
        return rc
    try:
        got = rows(a.trace)
    except ColdTraceError as exc:
        print(f"cold_trace: {exc}", file=sys.stderr)
        return 1
    fmt = got[0]["format"]
    total, complete = known_total(got[-1])
    summary = {
        "trace": a.trace,
        "format": fmt,
        "samples": len(got),
        "unrecorded": got[0]["unrecorded"],
        "first": got[0],
        "last": got[-1],
        # Never a bare number: an incomplete total carries the fact that it is
        # partial, so it cannot be read as the tier's occupancy.
        # The trace's values are ALREADY GiB -- the sampler wrote them that way
        # (A2's `shared_ssd=59.501` is the /health `kv_cold_shared_ssd_bytes`
        # 63888556032 B). No byte conversion happens in this module.
        "last_known_total_gib": round(total, 3) if not complete else None,
        "last_total_gib": (round(got[-1]["total"], 3)
                           if got[-1]["total"] is not None else None),
        "last_total_complete": complete,
    }
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"summary": summary, "rows": got}, fh, indent=1)
    print(json.dumps(summary, indent=1, default=str))
    if not complete:
        print(
            f"# {fmt} format: {got[0]['unrecorded']} were never sampled by this "
            f"version, so the four-key total above is PARTIAL "
            f"({summary['last_known_total_gib']} GiB from the recorded keys only). "
            "Do not read it as the tier's occupancy.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv or len(sys.argv) == 1:
        rc = _self_check()
        assert rc == 0
        raise SystemExit(rc)
    raise SystemExit(main())
