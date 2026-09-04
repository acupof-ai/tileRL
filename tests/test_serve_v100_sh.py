"""Does serve_v100.sh's restart loop restart, forward TERM, and refuse a second copy?

It runs the real script with a stub `python`, because a check that greps a shell script
asserts what the script says, not what it does. Both defects it caught were behavioural:
TERM stopped the supervisor without stopping the server (23 GiB held, nothing watching),
and the sandboxes landed in the repo checkout because the pod has no TMPDIR.

Run it on the pod -- macOS has no flock(1).
"""
import contextlib
import pathlib
import shutil
import subprocess
import tempfile
import time

SRC = pathlib.Path(__file__).parent / "serve_v100.sh"


@contextlib.contextmanager
def sandbox(exit_code, sleep_s=0):
    """The real script, pointed at a temp ROOT, with `python` stubbed."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="serve_v100_check."))
    try:
        (d / "tilerl-git").mkdir()
        (d / "venv70/bin").mkdir(parents=True)
        boots = d / "boots"
        # sleep_s > 0 stands in for a healthy server: runs until signalled, and
        # records into `terms` if TERM is what stops it.
        (d / "venv70/bin/python").write_text(
            f'#!/bin/bash\necho x >> {boots}\ntrap "echo t >> {d}/terms; exit 143" TERM\n'
            f'[ {sleep_s} -gt 0 ] && sleep {sleep_s} & wait $!\nexit {exit_code}\n')
        (d / "venv70/bin/python").chmod(0o755)
        script = d / "s.sh"
        script.write_text(SRC.read_text()
                          .replace("ROOT=/data00/home/chenkailun.c", f"ROOT={d}")
                          .replace("MAX_RESTARTS=10", "MAX_RESTARTS=2")
                          .replace("sleep 30", "sleep 0")
                          .replace("$(git rev-parse --short HEAD)", "stub")
                          .replace("$(git status --porcelain | wc -l)", "0"))
        script.chmod(0o755)
        yield d, script, boots
    finally:
        shutil.rmtree(d, ignore_errors=True)


def boots_of(path):
    return len(path.read_text().split()) if path.exists() else 0


if __name__ == "__main__":
    if shutil.which("flock") is None:
        raise SystemExit("flock(1) absent (macOS): run this on the pod")

    with sandbox(7) as (_, script, boots):
        rc = subprocess.run(["bash", str(script)], capture_output=True, timeout=120).returncode
        assert (boots_of(boots), rc) == (3, 1), f"crash: want 3 boots then give up, got {boots_of(boots)}/{rc}"

    with sandbox(143) as (_, script, boots):
        rc = subprocess.run(["bash", str(script)], capture_output=True, timeout=120).returncode
        assert (boots_of(boots), rc) == (1, 143), f"TERM must not restart, got {boots_of(boots)}/{rc}"

    # TERM to the supervisor must reach the server.
    with sandbox(0, sleep_s=30) as (d, script, boots):
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
        assert rc == 143 and boots_of(boots) == 1, f"want one boot and rc 143, got {boots_of(boots)}/{rc}"

    # The lock, and the control proving it is the lock that refuses.
    with sandbox(0) as (d, script, _):
        holder = subprocess.Popen(["bash", "-c", f"exec 9>{d}/.serve70.lock; flock -n 9; sleep 10"])
        try:
            time.sleep(0.5)
            r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60)
            assert r.returncode == 1 and "already running" in r.stderr, r.stderr[:200]
            unlocked = d / "u.sh"
            unlocked.write_text(script.read_text().replace("flock -n 9 ||", "false &&"))
            rc = subprocess.run(["bash", str(unlocked)], capture_output=True, timeout=60).returncode
            assert rc == 0, f"the lock is what refuses, not something else: {rc}"
        finally:
            holder.kill()

    left = list(pathlib.Path.cwd().glob("serve_v100_check.*"))
    assert not left, f"the check left sandboxes in the CWD: {left}"
    print("restart, give-up cap, TERM propagation, the lock, and its control: all pass")
