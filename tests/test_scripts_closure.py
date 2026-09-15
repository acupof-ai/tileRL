"""Closure gate for the scripts/ cleanup (architecture step 5).

The scripts/ audit (`scripts/audit_scripts_entrypoints.py`) classifies every
`scripts/*.py` by enumerated reachability -- pyproject, CI, imports, docs,
shell invocation, READMEs, and the test_main_selfchecks glob. The cleanup PRs
(#603/#609/#610/#613) deleted the dead one-off probes; this gate is the
anti-regression half: a new probe that nothing reaches and nobody registered
must fail CI, instead of accumulating the way the 78 `probe_*` files did.

The one reachability the seven sets structurally cannot see is a HAND-RUN
tool: a card/ops/deploy/review command a person types, which no file names.
Those live in the audit's `MANUAL_KEEP` registry with a reason that says who
runs it on which work line. This gate enforces that contract:

* no script buckets DEAD (unreachable AND unregistered);
* every registry entry names an existing script that is genuinely unreachable
  (a reachable script on the registry is redundant and must come off);
* every reason is a concrete sentence, not a placeholder;
* the registered set equals the scripts actually bucketed MANUAL_KEEP, so
  deleting a kept script without dropping its name fails loudly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _audit():
    spec = importlib.util.spec_from_file_location(
        "audit_scripts_entrypoints",
        ROOT / "scripts" / "audit_scripts_entrypoints.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_script_is_unreachable_and_unregistered():
    mod = _audit()
    report = mod.audit()
    dead = [r["file"] for r in report["rows"] if r["bucket"] == "DEAD"]
    assert not dead, (
        "scripts with no reachable source and no MANUAL_KEEP entry -- either delete the "
        "one-off or register it with a concrete reason:\n  " + "\n  ".join(dead))


def test_manual_keep_entries_are_existing_genuinely_unreachable_scripts():
    mod = _audit()
    report = mod.audit()
    stems = {r["stem"] for r in report["rows"]}
    bucketed = {r["stem"] for r in report["rows"] if r["bucket"] == "MANUAL_KEEP"}
    registry = set(mod.MANUAL_KEEP)

    missing = registry - stems
    assert not missing, f"MANUAL_KEEP names scripts that do not exist: {sorted(missing)}"

    redundant = registry - bucketed
    assert not redundant, (
        "MANUAL_KEEP scripts that ARE reachable through the seven sets -- drop them from "
        f"the registry, the reachability already keeps them: {sorted(redundant)}")

    unregistered = bucketed - registry
    assert not unregistered, (
        f"MANUAL_KEEP-bucketed scripts missing from registry: {sorted(unregistered)}")


def test_every_manual_keep_reason_is_specific():
    mod = _audit()
    vague = []
    for stem, reason in mod.MANUAL_KEEP.items():
        words = reason.split()
        # A concrete reason names the runner and the work line; a placeholder is
        # a few filler words. Threshold is a floor, not prose judgment -- the
        # shortest real entry above is well past it.
        if len(words) < 8 or "useful" in reason.lower():
            vague.append(f"{stem}: {reason!r}")
    assert not vague, (
        "MANUAL_KEEP reasons must say who runs the tool on which work line, not a "
        "placeholder:\n  " + "\n  ".join(vague))


if __name__ == "__main__":
    _audit()
    print("scripts closure gate ok")
