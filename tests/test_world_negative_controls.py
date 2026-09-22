"""Hermetic gates for the distributed `*_world*` negative controls (finding 8).

Each `tests/*_world[N].py` (and `gdn_cp_scan.py`) ships a positive path CI runs
with no flags and one-or-more NEGATIVE CONTROLS behind a flag that mutates the
collective (drop the backward fork, average per-shard stats, skip the prefix
scan, ...). The control is a real gate only if removing the guard makes it FAIL
where the positive path passes. Until now the controls ran only when a human
typed the flag — CI invoked every gate with no arguments, so a vacuous control
(one that passes despite the broken collective) was invisible.

This harness runs every control as a subprocess (the gates mp.spawn gloo
worlds, so they cannot be imported in-process) and asserts the gate's own
contract:

* exit code 0 — every gate's `main()` returns 0 exactly when its control
  CORRECTLY FAILED, and 1 with a "PASSED -- vacuous gate" line when it did not;
* stdout names a control and contains no "vacuous gate" verdict.

If a control's guard is weakened so the mutation no longer fails, that gate
exits 1 and this parameter goes red — the "guard removed -> red" proof is the
gate's own exit code, asserted in CI rather than by hand.

A nonzero exit is split into two causes (#794). A rendezvous/transport INIT
failure (the pick-then-bind MASTER_PORT race under xdist) is INFRASTRUCTURE,
not a weak guard: it is reported with an `INFRA_RENDEZVOUS` marker and the
subprocess stdout/stderr is landed to tmp_path, so the raw EADDRINUSE artifact
is what CI shows. Everything else nonzero is the control itself (vacuous or a
real error). The two must never share a verdict, or an infra flake reads as a
control failure.

The paired POSITIVE runs live in the CI "Distributed gates" workflow step; here
we only cover the controls that step never invoked.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TILERL_TARGET", "cpu")

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent

#: Strict rendezvous/transport-INIT failure primitives. Bare "gloo" is absent on
#: purpose: the backend banner prints "Gloo" on every healthy run, so it is not a
#: failure signal (a #794 probe produced 66 false positives matching it).
_INFRA_INIT = re.compile(
    r"address already in use|eaddrinuse|errno ?(?:48|98)|"
    r"tcpstore|rendezvous|init_(?:tcp|process_group)|"
    r"connection refused|connect\(\) failed|"
    r"childfailederror|processraisedexception",
    re.IGNORECASE,
)


def _free_port() -> int:
    """Bind an ephemeral port and release it, returning the number. Two are
    grabbed back-to-back for gates that spawn two worlds (dp_world4). There is
    an inherent close/reuse race, but it is far smaller than the hardcoded
    collisions and each control fails loudly (EADDRINUSE) rather than silently.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

#: (gate file, control flag). One case per documented negative control. A gate
#: with several guards (dp has three, tp two, gdn-cp two) lists each, so a
#: vacuous control for one guard cannot hide behind another passing.
_CONTROLS: list[tuple[str, str]] = [
    ("ce_sharded_world2.py", "--gather"),
    ("cp_model_world2.py", "--gdn-only"),
    ("cp_world2.py", "--slice-bwd"),
    ("cp_world2.py", "--seq-mask"),
    ("dp_world4.py", "--no-dp"),
    ("dp_world4.py", "--scramble"),
    ("dp_world4.py", "--scramble-keys"),
    ("gdn_cp_gradcheck_world2.py", "--no-compose"),
    ("gdn_cp_gradcheck_world2.py", "--decay-a"),
    ("gdn_cp_halo_tape_world2.py", "--no-halo"),
    ("gdn_cp_scan.py", "--no-compose"),
    ("gdn_cp_scan.py", "--decay-a"),
    ("gdn_cp_tape_world2.py", "--no-scan"),
    ("gdn_cp_tape_world2.py", "--decay-a"),
    ("gdn_halo_world2.py", "--no-halo"),
    ("gdn_world2.py", "--no-scan"),
    ("gdn_world2.py", "--decay-a"),
    ("mesh_world4.py", "--ungrouped"),
    ("tp_backend_world2.py", "--no-collective"),
    ("tp_backend_world2.py", "--rank0-shard"),
    ("tp_world2.py", "--no-fork"),
    ("tp_world2.py", "--local-stats"),
    ("tp_world2.py", "--local-clip"),
]


def test_the_control_table_covers_every_distributed_gate():
    """The harness cannot silently lose controls. Every file the CI distributed
    step runs must declare at least one control here, and the table must hold the
    full set — a new gate with no negative control fails this until one is added."""
    import glob

    gate_files = {Path(p).name for p in
                  glob.glob(str(_TESTS / "*_world[0-9].py")) +
                  [str(_TESTS / "gdn_cp_scan.py")]}
    covered = {g for g, _ in _CONTROLS}
    missing = gate_files - covered
    assert not missing, f"distributed gates with no hermetic negative control: {sorted(missing)}"
    # One row per control flag (some gates have 2-3); the count is the count of
    # GUARDS, not gates, so weakening one guard cannot hide behind a gate's other.
    assert len(_CONTROLS) >= len(gate_files), (
        f"{len(_CONTROLS)} controls for {len(gate_files)} gates: at least one per gate")
    assert len(_CONTROLS) == 23, (
        f"expected the 23 documented controls, found {len(_CONTROLS)}; update this "
        f"(and the table) when adding or removing a guard")


@pytest.mark.parametrize("gate,flag", _CONTROLS,
                         ids=[f"{g.removesuffix('.py')}[{f}]" for g, f in _CONTROLS])
def test_world_negative_control_fails_as_designed(gate: str, flag: str, tmp_path):
    """The guard removed by `flag` must make the distributed comparison fail:
    the gate subprocess exits 0 (its own 'control correctly FAILED' verdict) and
    never prints 'vacuous gate'. A control that passes exits 1 here.

    On a nonzero exit the cause is split: a rendezvous/transport INIT failure is
    INFRA_RENDEZVOUS (the #794 port race), never conflated with a weak guard; the
    raw subprocess output is landed under tmp_path either way."""
    env = dict(os.environ, TILERL_TARGET="cpu")
    # Unique rendezvous ports per control: the gates hardcode one MASTER_PORT per
    # world (and two world sizes collide), so under xdist parallel controls hit
    # EADDRINUSE and fail as infrastructure noise instead of vacuous-control
    # failures. Every gate reads MASTER_PORT via setdefault (dp_world4 honors an
    # injected base plus MASTER_PORT_2 for its second, different-size world).
    env["MASTER_PORT"] = str(_free_port())
    env["MASTER_PORT_2"] = str(_free_port())
    env["MASTER_ADDR"] = "127.0.0.1"
    proc = subprocess.run(
        [sys.executable, str(_TESTS / gate), flag],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        # Land the raw artifact so CI preserves the actual failure primitive.
        log = tmp_path / f"{gate.removesuffix('.py')}{flag.replace('--', '-')}.log"
        log.write_text(out)
        infra = _INFRA_INIT.search(out)
        if infra:
            pytest.fail(
                f"INFRA_RENDEZVOUS {gate} {flag}: transport init failed "
                f"({infra.group(0)!r}), not a control verdict — port race, "
                f"not a weak guard. raw log: {log}\n--- output ---\n{out[-3000:]}")
        pytest.fail(
            f"{gate} {flag}: negative control did not fail the comparison as required "
            f"(exit {proc.returncode}); either the guard is vacuous (control passed) or "
            f"the control errored. raw log: {log}.\n--- output ---\n{out[-3000:]}")
    assert "control" in out.lower(), (
        f"{gate} {flag}: exited 0 but ran no identifiable control path.\n{out[-1000:]}")
    assert "vacuous gate" not in out.lower(), (
        f"{gate} {flag}: printed a vacuous-gate verdict despite exit 0.\n{out[-1000:]}")
