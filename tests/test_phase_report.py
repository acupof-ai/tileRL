"""Cold-tier phase split: the boundary is placed by `decfwd`, and a partial total
is never printed as a total.

`scripts/phase_report.py` divides an arm into "host filling" and "host full" and
reports tick stats per phase. Two properties carry the report and both have a
recorded failure behind them:

* the boundary maps from the trace onto the serve log by `decfwd` -- the sample
  index was used once and cut inside the wrong request;
* a legacy trace's four-key total is PARTIAL (its sampler never recorded two of
  the keys), and the recorded keys sum to a plausible number that is not the
  tier's occupancy.

This gate runs the script's hermetic self-check and controls two of its
assertions by mutation.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "phase_report.py"


def test_the_self_check_passes():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-check"],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert proc.returncode == 0, (
        f"self-check failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


def _env() -> dict:
    """`phase_report` imports its two siblings from `scripts/`, and the flipped
    copies live in a temp dir, so the real scripts dir has to be on the path --
    otherwise the copy dies on ImportError and the control passes for a reason
    that has nothing to do with what it flips."""
    import os

    return {**os.environ, "PYTHONPATH": str(ROOT / "scripts")}


def _run_flipped(old: str, new: str, expect: str) -> None:
    src = SCRIPT.read_text()
    flipped = src.replace(old, new)
    assert flipped != src, f"the line this control flips was not found: {old!r}"
    with tempfile.TemporaryDirectory() as d:
        alt = Path(d) / SCRIPT.name
        alt.write_text(flipped)
        proc = subprocess.run(
            [sys.executable, str(alt), "--self-check"],
            capture_output=True, text=True, timeout=120, cwd=ROOT, env=_env(),
        )
    assert proc.returncode != 0, (
        f"the flipped script still passed, so this gate does not cover {old!r}"
    )
    assert expect in (proc.stderr or ""), (
        f"the control failed for an unrelated reason; expected {expect!r}"
        f"\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


def test_a_zero_threshold_boundary_fails_the_gate():
    """The boundary is the first steady tick with `ssd_mmap > 0`. Relaxing that
    to `>= 0` must be caught: every steady tick carries the segment at 0 before
    the crossing, so the boundary would move to the first tick of the arm."""
    _run_flipped(
        '    return next((t["n"] for t in ticks if t["ssd_mmap"] > 0), None)',
        '    return next((t["n"] for t in ticks if t["ssd_mmap"] >= 0), None)',
        "boundary_tick",
    )


def test_printing_a_partial_total_as_a_total_fails_the_gate():
    """A legacy trace's two unrecorded keys must not be silently filled.

    `known_total` lives in `cold_trace.py`, so this control flips THAT script and
    runs `phase_report` against it -- which is also the point: the completeness
    verdict has one definition and one consumer, and weakening the definition must
    show up in the report. Filling the unrecorded keys makes the partial sum look
    complete; on A1 that is 7.999 GiB against a true 67.5, and 7.999 is exactly
    the host cap, so it reads as an ordinary "just filled" state.
    """
    trace_script = ROOT / "scripts" / "cold_trace.py"
    src = trace_script.read_text()
    flipped = src.replace(
        "    recorded = [p for p in parts if p is not None]\n"
        "    return sum(recorded), len(recorded) == 4",
        "    recorded = [p for p in parts if p is not None]\n"
        "    return sum(recorded), True",
    )
    assert flipped != src, "the known_total control no longer matches cold_trace.py"
    import os

    with tempfile.TemporaryDirectory() as d:
        # phase_report imports cold_trace by name, so the flipped copy must come
        # FIRST on the path -- the real scripts dir behind it supplies the other
        # sibling. Without this the copy dies on ImportError and the control
        # passes for a reason unrelated to what it flips.
        (Path(d) / "cold_trace.py").write_text(flipped)
        shim = Path(d) / "phase_report.py"
        shim.write_text(SCRIPT.read_text())
        env = {**os.environ,
               "PYTHONPATH": os.pathsep.join([d, str(ROOT / "scripts")])}
        proc = subprocess.run(
            [sys.executable, str(shim), "--self-check"],
            capture_output=True, text=True, timeout=120, cwd=ROOT, env=env,
        )
    assert proc.returncode != 0, (
        "a `known_total` that always reports complete still passed, so the "
        "partial-total property is not covered"
    )
    # The self-check asserts `last_total_gib is None` on the legacy fixture, so
    # the weakened `known_total` trips THAT. Match on it rather than on the
    # symptom it produces downstream (`round(None)`), which the report now
    # survives.
    assert "last_total_complete" in (proc.stderr or ""), (
        f"the control failed for an unrelated reason\n--- stderr ---\n{proc.stderr[-2000:]}"
    )
