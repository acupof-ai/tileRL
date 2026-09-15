"""Circuit-breaker gate for scripts/serve_hybrid_v100.sh's restart fuse.

The supervisor already restarted a crash; #2 adds the bound it did not have:
RESTART_FUSE_MAX crashes inside RESTART_FUSE_WINDOW_S must leave the server DOWN
(exit 2 + marker) instead of crash-looping forever, and crashes spaced further
apart than the window must age out and never trip. Runs the real script with a
stub python that crashes immediately; the readiness guard exits on its first
kill -0, so warmup/liveness never start.

Skips where flock(1) is absent (every macOS row, including CI macos); fires on
the linux CI row, the same split as test_serve_v100_sh.py.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import subprocess
import tempfile

import pytest

SRC = pathlib.Path(__file__).parent.parent / "scripts" / "serve_hybrid_v100.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("flock") is None, reason="flock(1) absent; the supervisor refuses to run unlocked"
)


@contextlib.contextmanager
def sandbox(env_extra):
    d = pathlib.Path(tempfile.mkdtemp(prefix="serve_hybrid_fuse."))
    try:
        repo = d / "tilerl-v100-sse"
        (repo / "src").mkdir(parents=True)
        (repo / "venv70/bin").mkdir(parents=True)
        (d / "models").mkdir()
        (d / "mmlu-assets").mkdir()
        # Crashes immediately, no matter the serve argv; sleeps make boots distinct.
        stub = d / "venv70/bin/python"
        stub.write_text("#!/bin/bash\nexit 7\n")
        stub.chmod(0o755)
        env = dict(os.environ)
        env.update(
            {
                "SERVE_ROOT": str(d),
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
