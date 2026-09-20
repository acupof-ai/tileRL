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
        '        if field not in spec["derivable"] and field not in spec["never_sampled"]:',
        '        if False:',
        "missing shared: no error raised",
    )


DATA = ROOT / "tests" / "testdata"


def test_two_real_trace_samples_parse_and_carry_their_headline_facts():
    """The synthetic fixtures cannot see a sampler-version drift, because they
    were written from the same reading as the parser. These two files are
    excerpts of the REAL traces -- verbatim lines, in file order, with the source
    named in the header -- so a renamed or dropped key shows up here.

    What they pin, in the two directions that matter:

    * A1 (`legacy`) has no `shared_ssd` and no `TOTAL`. Its own `/health` reported
      `kv_cold_shared_ssd_bytes` = 59.501 GiB, so the absence is UNRECORDED, not
      zero: the recorded keys sum to 7.999 GiB against a true 67.5, and 7.999 is
      exactly the host cap, so it would not be questioned.
    * A2 (`current`) carries all four and its last line is the same arm's real
      total, 67.5 GiB -- the number A1's trace cannot produce.

    No file is truncated into a "whole log": these are samples, and the assertion
    is about what a sample of this format can and cannot say.
    """
    a1 = DATA / "cold_trace_a1_sample.txt"
    a2 = DATA / "cold_trace_a2_sample.txt"
    assert a1.exists() and a2.exists(), "the real-trace samples are missing"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--trace", str(a1)],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "legacy" in proc.stdout, proc.stdout
    # UNRECORDED, said out loud on stderr, with the partial sum named as partial.
    assert "PARTIAL" in proc.stderr and "7.999" in proc.stderr, proc.stderr
    assert "shared_ssd" in proc.stderr and "total" in proc.stderr, proc.stderr
    # ...and never offered as the tier's occupancy.
    assert '"last_total_gib": null' in proc.stdout, proc.stdout
    assert "67.5" not in proc.stdout, "an unrecorded term was counted"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--trace", str(a2)],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    # The same arm's real total, which the legacy trace cannot produce.
    assert '"format": "current"' in proc.stdout, proc.stdout
    assert '"last_total_gib": 67.5' in proc.stdout, proc.stdout
    assert "PARTIAL" not in proc.stderr, proc.stderr


def test_a_comment_header_line_is_skipped_not_rejected():
    """The samples carry a `#` header naming their source. A reader that rejected
    it would fail on the file it was written for, so comment lines are skipped
    like blanks -- and the samples above are the check that they are."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("cold_trace_under_test", SCRIPT)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_line("# a header, not a sample\n", 1) is None
    assert mod.parse_line("\n", 2) is None
    assert mod.parse_line("   \n", 3) is None
    row = mod.parse_line(
        "1 priv=1.0 ssd=0.0 shared=2.0 shared_ssd=3.0 TOTAL=6.0 fin=1 decfwd=4 tok=8", 4)
    assert row is not None and row["total"] == 6.0, row


def test_unrecorded_is_computed_per_row_not_from_a_format_constant():
    """A format-level `unrecorded` constant contradicts its own row.

    d4f-close's finding: a current-format line missing `shared_ssd` and `TOTAL`
    (still identifiable as current by `fin`/`tok`) returned both fields as
    `None` -- correct -- while `unrecorded` was `[]` from the format constant, so
    the summary printed "[] were never sampled" over a row that said two things
    were missing. The summary had overridden the row's evidence.

    The self-check asserts the row-level list; this control flips it back to the
    constant and requires the self-check to go red.
    """
    _run_flipped(
        '    row["unrecorded"] = sorted(\n'
        '        {f for f in missing if f not in spec["derivable"]}\n'
        '        | {f for f in spec["never_sampled"] if row[f] is None}\n'
        '    )',
        '    row["unrecorded"] = sorted(spec["never_sampled"])',
        'derived["unrecorded"] == []',
    )


def test_skipping_the_total_cross_check_fails_the_gate():
    """`TOTAL` is derivable, so a line without it is recomputed. That positive
    case passes whether or not any check runs -- which is why the negative one
    exists: a written `TOTAL` that contradicts the four keys must raise."""
    _run_flipped(
        '            bad = abs(s - row["total"]) > TOTAL_TOL',
        '            bad = False',
        "a TOTAL that contradicts the four keys was accepted",
    )


def test_a_tolerance_free_total_check_fails_the_gate():
    """The cross-check needs a tolerance, or it fires on sound data.

    The trace writes 3 decimals, so summing five rounded numbers leaves ~1e-14 of
    residue: measured on A2, 25 of 205 rows differ under exact equality and 0 of
    205 under `round(sum, 3)`. A strict comparison would raise on a quarter of a
    sound file -- the same class of defect as the decfwd anchor, a gate that
    fires where it should stay green.
    """
    _run_flipped(
        'TOTAL_TOL = 1e-3',
        'TOTAL_TOL = 0.0',
        "residual 1.421e-14",
    )


def test_dropping_the_negative_residual_check_fails_the_gate():
    """A gap between the recorded sum and a written `TOTAL` is normally the
    unrecorded term itself, so it decides nothing. The one direction that DOES
    decide is the sign: a negative gap would need a negative byte count, and the
    four keys are non-negative on all three real traces. Without this branch a
    total below the recorded sum -- provably corrupt -- is accepted.
    """
    _run_flipped(
        '            bad = (row["total"] - s) < -TOTAL_TOL',
        '            bad = False',
        "a total implying a NEGATIVE byte count was accepted",
    )


def test_a_parser_that_forgets_the_legacy_alias_map_fails_the_gate():
    """The measured regression, not a hypothetical one.

    `cold`/`coldssd` are the legacy names for `priv`/`ssd`. Drop one entry from
    the alias map and a real A1 line becomes unreadable -- the key is not in the
    format, so the parser refuses it rather than reading the tier as empty. That
    refusal is the point: a parser that instead ignored the unknown key and
    defaulted it to 0 gives 7.999 GiB where the tier really held 67.5, and 7.999
    is exactly the host cap, so it reads as "cold tier just filled, SSD not yet in
    use" -- an ordinary state, and a silent one.

    One change, one failure site: the alias entry, and the message that names it.
    """
    _run_flipped('"cold": "priv", ', "", "not part of the legacy format")
