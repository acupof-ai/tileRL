"""Circuit-breaker gate for scripts/serve_hybrid_v100.sh's restart fuse.

The supervisor already restarted a crash; #2 adds the bound it did not have:
RESTART_FUSE_MAX crashes inside RESTART_FUSE_WINDOW_S must leave the server DOWN
(exit 2 + marker) instead of crash-looping forever, and crashes spaced further
apart than the window must age out and never trip. Runs the real script with a
stub python that crashes immediately; the readiness guard exits on its first
kill -0, so warmup/liveness never start.

Runs on macOS too, where flock(1) is absent, via the same `_flock_shim` helper
`test_serve_v100_sh.py` uses; the linux row keeps the real binary.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import subprocess
import tempfile

from _flock_shim import flock_path

SRC = pathlib.Path(__file__).parent.parent / "scripts" / "serve_hybrid_v100.sh"

@contextlib.contextmanager
def sandbox(env_extra):
    """PATH in the yielded env carries flock(1), real or shimmed.

    Inside sandbox, not at each call site: a test added here later would
    otherwise run the launcher with no flock and fail for a reason unrelated to
    what it checks.
    """
    with flock_path() as path:
        d = pathlib.Path(tempfile.mkdtemp(prefix="serve_hybrid_fuse."))
        try:
            repo = d / "tilerl-v100-sse"
            (repo / "src").mkdir(parents=True)
            (d / "venv70/bin").mkdir(parents=True)
            (d / "models").mkdir()
            (d / "mmlu-assets").mkdir()
            # Crashes immediately, no matter the serve argv; sleeps make boots distinct.
            # Dumps its own environment first, so a gate can read what the supervisor
            # actually handed the child (the extra write is harmless to the fuse tests,
            # which only count boots and read the log).
            stub = d / "venv70/bin/python"
            stub.write_text('#!/bin/bash\nexport > "$SERVE_ROOT/childenv.txt"\nexit 7\n')
            stub.chmod(0o755)
            env = dict(os.environ)
            env.update(
                {
                    "SERVE_ROOT": str(d),
                    "PATH": path,
                    "MAX_RESTARTS": "10",
                    "RESTART_FUSE_MAX": "2",
                    "RESTART_FUSE_WINDOW_S": "600",
                }
            )
            env.update(env_extra)
            yield d, env
        finally:
            shutil.rmtree(d, ignore_errors=True)


def test_a_crash_burst_trips_the_fuse_and_stays_down():
    with sandbox({}) as (d, env):
        r = subprocess.run(["bash", str(SRC)], capture_output=True, text=True, timeout=120, env=env)
        assert r.returncode == 2, r.stderr[:300]
        log = (d / "servehybridsse.log").read_text()
        assert "FUSE: 2 restarts within 600s" in log
        # Two recorded restarts: the fuse check runs before each boot, so the
        # third boot is refused and only boots 0 and 1 were ever attempted.
        assert len((d / ".servehybridsse.fuse").read_text().split()) == 2
        assert log.count("boot ") == 2


def test_crashes_outside_the_window_age_out_and_never_trip():
    # A 1s window and the launcher's own 5s post-crash sleep make every recorded
    # restart older than the window by the next boot, so the fuse never trips and
    # the run ends on MAX_RESTARTS (exit 1), not on the fuse.
    with sandbox({"RESTART_FUSE_WINDOW_S": "1", "MAX_RESTARTS": "2"}) as (d, env):
        r = subprocess.run(["bash", str(SRC)], capture_output=True, text=True, timeout=120, env=env)
        assert r.returncode == 1, r.stderr[:300]
        log = (d / "servehybridsse.log").read_text()
        assert "FUSE:" not in log
        assert "gave up after 2 restarts" in log


def test_the_child_gets_expandable_segments():
    """The sm70 allocator flag must reach the served process, and an operator's
    own value must win over the launcher's default (it changes allocator behaviour
    globally, so a run that opts out has to be able to).

    `export` quotes the value, so the assertions match the assignment rather than
    the bare pair: the first version omitted the quotes and went red against a
    script that was already correct."""
    with sandbox({"MAX_RESTARTS": "0"}) as (d, env):
        subprocess.run(["bash", str(SRC)], capture_output=True, text=True, timeout=120, env=env)
        child = (d / "childenv.txt").read_text()
        assert 'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' in child, child[:400]

    with sandbox({"MAX_RESTARTS": "0",
                  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}) as (d, env):
        subprocess.run(["bash", str(SRC)], capture_output=True, text=True, timeout=120, env=env)
        child = (d / "childenv.txt").read_text()
        assert 'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"' in child, child[:400]
        assert "expandable_segments:True" not in child


def test_only_the_sm70_launcher_sets_it():
    """Scoped to the sm70 launcher, not a cross-backend default: the sm90 and dense
    launch paths must not inherit it."""
    here = pathlib.Path(__file__).parent.parent / "scripts"
    for name in ("serve_h20.sh", "serve_v100_dense.sh", "serve_v100.sh"):
        text = (here / name).read_text()
        assert "PYTORCH_CUDA_ALLOC_CONF" not in text, f"{name} sets the sm70 allocator flag"


def test_the_guard_poll_period_is_passed_through_and_defaults_to_60():
    """The liveness guard injects a real 4-token chat every poll while a slot is
    free, so a zero-traffic baseline needs LIVENESS_POLL_S raised. The launcher must
    pass a caller's value through and leave the shipped 60 s in place otherwise.

    `sandbox`'s stub dumps the environment the supervisor hands its child; the guard
    is a sibling under the same shell, so that dump is what serve_liveness.py reads.
    """
    for extra, want, why in (
        ({"LIVENESS_POLL_S": "999999"}, '"999999"', "a caller's override"),
        ({}, '"60"', "the shipped default when unset"),
        ({"LIVENESS_POLL_S": ""}, '"60"',
         "an empty value must not reach float('') in the guard"),
    ):
        with sandbox({"MAX_RESTARTS": "0", **extra}) as (d, env):
            env.pop("LIVENESS_POLL_S", None)
            env.update(extra)
            subprocess.run(["bash", str(SRC)], capture_output=True, text=True,
                           timeout=120, env=env)
            child = (d / "childenv.txt").read_text()
            line = f'LIVENESS_POLL_S={want}'
            assert line in child, (why, [ln for ln in child.splitlines()
                                         if "LIVENESS" in ln])
