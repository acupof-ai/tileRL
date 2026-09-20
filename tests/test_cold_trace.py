"""Cold-trace parsing: two formats, and unrecorded is not zero.

The parser (`scripts/cold_trace.py`) reads the cold-tier occupancy trace the
sampler writes alongside a serve arm. Two sampler versions are on disk and the
legacy one does not record two of the four keys, so the interesting property is
not the arithmetic but what a MISSING key means.

This gate runs the script's hermetic self-check and then controls it: a
`missing = 0` default must make it fail, on the assertion that exists to catch
exactly that.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cold_trace.py"


def test_the_self_check_passes():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-check"],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert proc.returncode == 0, (
        f"self-check failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


def _run_flipped(old: str, new: str, expect_in_stderr: str) -> None:
    """Rewrite one line of the script, run its self-check, require a red with the
    named assertion in the traceback. Shared by the controls below."""
    src = SCRIPT.read_text()
    flipped = src.replace(old, new)
    assert flipped != src, f"the line this control flips was not found: {old!r}"
    with tempfile.TemporaryDirectory() as d:
        alt = Path(d) / SCRIPT.name
        alt.write_text(flipped)
        proc = subprocess.run(
            [sys.executable, str(alt), "--self-check"],
            capture_output=True, text=True, timeout=120, cwd=ROOT,
        )
    assert proc.returncode != 0, (
        f"the flipped script still passed its self-check, so this gate does not "
        f"cover {old!r}"
    )
    assert expect_in_stderr in (proc.stderr or ""), (
        f"the control failed for an unrelated reason; expected {expect_in_stderr!r}"
        f"\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


def test_defaulting_an_unrecorded_key_to_zero_fails_the_gate():
    """The regression this parser exists to prevent.

    On A1 the serve's `/health` reports 59.501 GiB of shared SSD bytes while the
    trace has no such key: the tier held them, the sampler never wrote them down.
    Summing the trace under `missing = 0` gives 8.0 GiB against a true 67.5, and
    8.0 looks entirely reasonable. Flipping the default back to 0 must go red on
    the unrecorded-is-None assertion, not somewhere incidental.
    """
    _run_flipped(
        "    for field in FIELDS:\n        if field not in row:\n            row[field] = None",
        "    for field in FIELDS:\n        if field not in row:\n            row[field] = 0.0",
        'r1["shared_ssd"] is None',
    )


def test_a_missing_required_key_fails_the_gate():
    """The other direction: a truncated line must RAISE, not parse to an empty
    row. `shared`/`priv`/`ssd`/`decfwd` are the required set -- `TOTAL` is
    legitimately absent in the current format, so its absence decides nothing.

    With the required-key check removed the parse SUCCEEDS silently, so the
    self-check's own `no error raised` assertion is what fires. That is the
    intended failure: the script-level check is what turns "the line parsed" into
    "the line should not have parsed".
    """
    _run_flipped(
        '    for key in REQUIRED:\n        if key not in row:',
        '    for key in ():\n        if key not in row:',
        "no error raised",
    )


def test_a_parser_that_forgets_the_legacy_alias_map_fails_the_gate():
    """The measured regression, not a hypothetical one.

    A parser that knows only the CURRENT key names and falls back to 0 for
    anything else never resolves a legacy line's `cold`/`coldssd`, so every key of
    that line reads 0. On A1 that gives 7.999 GiB where the tier really held
    67.500 -- and 7.999 is exactly the host cap, so it reads as "cold tier just
    filled, SSD not yet in use", a perfectly ordinary state. This flips the
    aliases away and the lenient path in, then requires the unrecorded-is-None
    assertion to catch it.
    """
    src = SCRIPT.read_text()
    flipped = src.replace(
        '        field = spec["aliases"].get(disk_key)\n        if field is None:\n'
        '            raise ColdTraceError(\n'
        '                f"line {lineno}: key {disk_key!r} is not part of the {fmt} format"\n'
        '            )',
        '        field = spec["aliases"].get(disk_key)\n        if field is None:\n'
        '            row[disk_key] = None\n            continue',
    ).replace(
        "    for field in FIELDS:\n        if field not in row:\n            row[field] = None",
        "    for field in FIELDS:\n        if field not in row:\n            row[field] = 0.0",
    )
    assert flipped != src, "the lenient-parser control no longer matches the script"
    with tempfile.TemporaryDirectory() as d:
        alt = Path(d) / SCRIPT.name
        alt.write_text(flipped)
        proc = subprocess.run(
            [sys.executable, str(alt), "--self-check"],
            capture_output=True, text=True, timeout=120, cwd=ROOT,
        )
    assert proc.returncode != 0, (
        "a parser that drops the legacy alias map passed the self-check, so the "
        "alias mapping is not actually covered"
    )
    assert 'r1["shared_ssd"] is None' in (proc.stderr or ""), (
        f"the control failed for an unrelated reason\n--- stderr ---\n{proc.stderr[-2000:]}"
    )
