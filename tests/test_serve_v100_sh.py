"""Does serve_v100.sh's restart loop restart, forward TERM, and refuse a second copy?

It runs the real script with a stub `python`, because a check that greps a shell script
asserts what the script says, not what it does. Both defects it caught were behavioural:
TERM stopped the supervisor without stopping the server (23466 MiB held, nothing
watching), and the sandboxes landed in the repo checkout because the pod sets no TMPDIR.

Runs on macOS too, where flock(1) is absent: `_flock_shim` puts a real
fcntl-backed flock on PATH for the subprocesses. On Linux the shim is not built
and the real binary is tested, `-n` argument form included.
"""

import contextlib
import os
import pathlib
import shutil
import subprocess
import tempfile
import time

from _flock_shim import flock_path

SRC = pathlib.Path(__file__).parent.parent / "scripts" / "serve_v100.sh"


@contextlib.contextmanager
def sandbox(exit_code, sleep_s=0, path=None):
    """The real script, pointed at a temp ROOT, with `python` stubbed.

    PATH is set on the environment for the duration, not passed per subprocess:
    the tests below spawn bash directly, and the launcher's own `command -v
    flock` must resolve the same flock the lock-holder test takes.
    """
    saved = os.environ.get("PATH")
    if path is not None:
        os.environ["PATH"] = path
    d = pathlib.Path(tempfile.mkdtemp(prefix="serve_v100_check."))
    try:
        (d / "tilerl-git").mkdir()
        (d / "venv70/bin").mkdir(parents=True)
        boots = d / "boots"
        # sleep_s > 0 stands in for a healthy server: runs until signalled, and
        # records into `terms` if TERM is what stops it. It writes its own pid so a
        # test can signal the SERVER rather than the supervisor.
        (d / "venv70/bin/python").write_text(
            f'#!/bin/bash\necho x >> {boots}\necho $$ > {d}/pid\n'
            f'trap "echo t >> {d}/terms; exit 143" TERM\n'
            f"[ {sleep_s} -gt 0 ] && sleep {sleep_s} & wait $!\nexit {exit_code}\n"
        )
        (d / "venv70/bin/python").chmod(0o755)
        script = d / "s.sh"
        script.write_text(
            SRC.read_text()
            .replace("ROOT=/data00/home/chenkailun.c", f"ROOT={d}")
            .replace("MAX_RESTARTS=10", "MAX_RESTARTS=2")
            .replace("sleep 30", "sleep 0")
            .replace("$(git rev-parse --short HEAD)", "stub")
            .replace("$(git status --porcelain | wc -l)", "0")
        )
        script.chmod(0o755)
        yield d, script, boots
    finally:
        if saved is not None:
            os.environ["PATH"] = saved
        shutil.rmtree(d, ignore_errors=True)


def boots_of(path):
    return len(path.read_text().split()) if path.exists() else 0


def test_a_crash_restarts_and_the_cap_gives_up():
    with flock_path() as path, sandbox(7, path=path) as (_, script, boots):
        rc = subprocess.run(["bash", str(script)], capture_output=True, timeout=120).returncode
        assert (boots_of(boots), rc) == (3, 1), (
            f"want 3 boots then give up, got {boots_of(boots)}/{rc}"
        )


def test_a_clean_exit_is_not_restarted():
    with flock_path() as path, sandbox(0, path=path) as (_, script, boots):
        rc = subprocess.run(["bash", str(script)], capture_output=True, timeout=120).returncode
        assert (boots_of(boots), rc) == (1, 0), (
            f"a clean exit must not restart, got {boots_of(boots)}/{rc}"
        )


def test_killing_only_the_server_restarts_it():
    """The defect this caught: restarting the server to pick up new code left nothing
    listening on 8000, and the supervisor was gone too.

    The loop used to read "stopped on purpose" off the CHILD's exit code, and a server
    killed by anything other than the supervisor exits 143 -- a plain `kill -TERM <server
    pid>`, or the OOM killer. So it walked away from exactly the case it exists for. Only
    a signal aimed at the supervisor means stop, which the trap knows and the exit code
    does not.
    """
    with flock_path() as path, sandbox(0, sleep_s=30, path=path) as (d, script, boots):
        sup = subprocess.Popen(["bash", str(script)])
        try:
            for _ in range(100):
                if boots.exists():
                    break
                time.sleep(0.1)
            time.sleep(0.5)
            # The stub writes its own pid; kill THAT, not the supervisor.
            child = int((d / "pid").read_text().strip())
            subprocess.run(["kill", "-TERM", str(child)], check=True)
            for _ in range(200):
                if boots_of(boots) >= 2:
                    break
                time.sleep(0.1)
            assert boots_of(boots) >= 2, (
                f"the server was killed and never came back ({boots_of(boots)} boot(s)): "
                f"nothing is listening and the supervisor has walked away"
            )
            assert sup.poll() is None, "the supervisor exited when only the server was killed"
        finally:
            sup.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                sup.wait(timeout=30)
            if sup.poll() is None:
                sup.kill()


def test_term_to_the_supervisor_reaches_the_server():
    """The defect this caught: the supervisor exited in 1s and left python holding
    23466 MiB of the card with nothing supervising it."""
    with flock_path() as path, sandbox(0, sleep_s=30, path=path) as (d, script, boots):
        sup = subprocess.Popen(["bash", str(script)])
        try:
            for _ in range(100):
                if boots.exists():
                    break
                time.sleep(0.1)
            time.sleep(0.5)
            sup.terminate()
            rc = sup.wait(timeout=30)
        finally:
            if sup.poll() is None:
                sup.kill()
        assert (d / "terms").exists(), "the server never saw TERM; it would outlive the supervisor"
        assert rc == 143 and boots_of(boots) == 1, (
            f"want one boot and rc 143, got {boots_of(boots)}/{rc}"
        )


def test_a_second_supervisor_is_refused_and_the_lock_is_why():
    with flock_path() as path, sandbox(0, path=path) as (d, script, _):
        # The holder takes the lock through the SAME flock the launcher resolves,
        # so this test still means what it did when it only ran on Linux.
        holder = subprocess.Popen(
            ["bash", "-c", f"exec 9>{d}/.serve70.lock; flock -n 9; sleep 10"]
        )
        try:
            time.sleep(0.5)
            r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60)
            assert r.returncode == 1 and "already running" in r.stderr, r.stderr[:200]
            # Control: strip the guard and the second copy runs, so it is the lock
            # that refuses and not some other early exit.
            unlocked = d / "u.sh"
            unlocked.write_text(script.read_text().replace("flock -n 9 ||", "false &&"))
            rc = subprocess.run(["bash", str(unlocked)], capture_output=True, timeout=60).returncode
            assert rc == 0, f"the lock is what refuses, not something else: {rc}"
        finally:
            holder.kill()


def test_the_sandboxes_do_not_land_in_the_cwd():
    """The other defect this caught: the pod sets no TMPDIR, so `gettempdir()` falls
    back to the CWD and six sandboxes were left in the repo checkout."""
    with flock_path() as path, sandbox(0, path=path) as (d, _, _boots):
        assert d.exists()
    assert not d.exists(), "sandbox() did not remove its directory"
    assert not list(pathlib.Path.cwd().glob("serve_v100_check.*"))
