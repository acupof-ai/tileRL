"""Per-prompt spread in the W-sweep probe (`scripts/probe_draft_window_sweep.py`).

The W=2048 verdict is a between-W comparison, and a single cross-prompt median
cannot say whether the prompts agreed. This gate covers the added statistic: the
per-prompt steady-tick band, the nearest-rank quantiles it is built on, and the
steady-set filter (`dec=1 & sparse=1 & model>0 & sample>0`, path != graph).

The probe itself names a backend, so it is outside the CI hermetic set
(`tests/test_main_selfchecks.py`) and nothing else would run its self-check on a
GPU-less host.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_draft_window_sweep.py"


def test_the_spread_self_check_passes():
    proc = subprocess.run(
        [sys.executable, str(PROBE), "--self-check"],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert proc.returncode == 0, (
        f"self-check failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


def test_the_rendered_table_is_covered_by_the_self_check():
    """The other control: the per-prompt-rows printer is keyed by name, so a
    renamed row field would raise on the card after a multi-hour sweep. Flipping
    the key the printer reads must make the self-check fail on the render
    assertion rather than pass because nothing renders it.
    """
    src = PROBE.read_text()
    flipped = src.replace(
        'f"{w:>5} {row[\'i\']:>3} {row[\'prompt_len\']:>10} "',
        'f"{w:>5} {row[\'i\']:>3} {row[\'no_such_field\']:>10} "',
    )
    assert flipped != src, (
        "the per-prompt row printer was not found, so this control no longer "
        "points at the line it is meant to flip"
    )
    with tempfile.TemporaryDirectory() as d:
        alt = Path(d) / PROBE.name
        alt.write_text(flipped)
        proc = subprocess.run(
            [sys.executable, str(alt), "--self-check"],
            capture_output=True, text=True, timeout=120, cwd=ROOT,
        )
    assert proc.returncode != 0, (
        "a printer reading a field the row dict does not define still passed the "
        "self-check, so the render is not covered"
    )
    assert "no_such_field" in (proc.stderr or ""), (
        "the control failed for an unrelated reason\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )


def test_a_mean_based_prompt_statistic_does_not_pass_the_gate():
    """The control for the spread assertion.

    The self-check's whole point is that the between-prompt band is built from
    per-prompt MEDIANS. Substituting the mean must make it fail, and fail on the
    band assertion (`iqr > 0`) rather than somewhere incidental: the toy data is
    two tight prompts and two with a single 370 ms outlier, which have identical
    means and different medians.
    """
    src = PROBE.read_text()
    flipped = src.replace(
        'per_prompt = [d["tok_s_spread"]["p50"] for d in dec if d["steady_ticks"]]',
        'per_prompt = [d["tok_s"] for d in dec if d["steady_ticks"]]',
    )
    assert flipped != src, (
        "the per-prompt derivation was not found, so this control no longer "
        "points at the line it is meant to flip"
    )
    # A temp FILE, not `python -c`: the probe reads `__file__` to put its own
    # tree on sys.path, which `-c` does not define -- the control would then
    # "fail" on a NameError instead of on the assertion it is testing.
    with tempfile.TemporaryDirectory() as d:
        alt = Path(d) / PROBE.name
        alt.write_text(flipped)
        proc = subprocess.run(
            [sys.executable, str(alt), "--self-check"],
            capture_output=True, text=True, timeout=120, cwd=ROOT,
        )
    assert proc.returncode != 0, (
        "a mean-based per-prompt statistic passed the self-check, so the gate "
        "does not actually distinguish the per-prompt band from a pooled mean"
    )
    assert "same_mean" in (proc.stderr or ""), (
        "the control failed for an unrelated reason; expected the between-prompt "
        f"band assertion\n--- stderr ---\n{proc.stderr[-2000:]}"
    )
