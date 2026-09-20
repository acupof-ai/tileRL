"""Does serve_v100.sh's restart loop restart, forward TERM, and refuse a second copy?

It runs the real script with a stub `python`, because a check that greps a shell script
asserts what the script says, not what it does. Both defects it caught were behavioural:
TERM stopped the supervisor without stopping the server (23466 MiB held, nothing
watching), and the sandboxes landed in the repo checkout because the pod sets no TMPDIR.
The timestamp gate is the exception that is not behavioural: it checks the `date`
FORMAT, because `date -Is` is a GNU-only spelling that BSD `date` rejects outright.

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
        # so this test still means what it did when it only ran on Linux. It
        # writes a marker file only after flock returns, which is what makes the
        # hold observable instead of assumed.
        ready = d / "holder.ready"
        holder = subprocess.Popen(
            [
                "bash",
                "-c",
                f"exec 9>{d}/.serve70.lock; flock -n 9 || exit 9; : > {ready}; sleep 10",
            ]
        )
        try:
            # Poll for the marker, never sleep a fixed interval: under `pytest -n
            # auto` the holder's bash may not have reached flock yet when the
            # second copy starts, and the lock is then legitimately free -- the
            # second supervisor runs and the assertion fails on an unheld lock
            # rather than on a broken one. Measured on this test: a fixed 0.5 s
            # wait failed 4/30 runs under -n auto; polling passes 30/30.
            deadline = time.monotonic() + 30
            while not ready.exists():
                assert holder.poll() is None, (
                    f"the lock holder exited (rc={holder.returncode}) without taking the lock"
                )
                assert time.monotonic() < deadline, "the lock holder never signalled it held the lock"
                time.sleep(0.05)
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


def _date_flavour(binary: str) -> str:
    """'bsd' or 'gnu' -- asked of the BINARY, never inferred from its name or host.

    The first version labelled `date` as "bsd/macOS" and `gdate` as "gnu" and then
    branched on the label, so on Linux (where plain `date` IS GNU) it ran the BSD
    parse command `date -j -f` against GNU date: `date: invalid option -- 'j'`.
    Same defect class as the bug this PR fixes -- a platform-specific branch running
    on the wrong platform -- so it is keyed on a probe, not on an assumption.
    """
    # `-j -f <fmt> <date>` is BSD-only; GNU refuses `-j`.
    probe = subprocess.run([binary, "-j", "-f", "%Y-%m-%d", "2026-01-01", "+%s"],
                           capture_output=True, text=True, timeout=30)
    return "bsd" if probe.returncode == 0 else "gnu"


def test_the_boot_timestamp_is_portable_and_parseable():
    """The boot/exit lines are the only record of when a restart happened, and they
    used `date -Is`, which BSD `date` rejects (`date: invalid argument 's' for -I`)
    -- so on macOS every one of those lines carried an empty timestamp.

    The gate is the timestamp FORMAT, not a grep for the new spelling: run every
    `date` implementation the scripts can meet and require each to emit a value its
    own platform parser accepts. A grep would pass on a string no `date` produces.
    """
    from datetime import datetime

    fmt = "%Y-%m-%dT%H:%M:%S%z"
    # `date` is POSIX-required; `gdate` is the GNU build Homebrew installs beside
    # BSD date on macOS, so that host exercises both flavours.
    seen = set()
    for binary in ("date", "gdate"):
        if shutil.which(binary) is None:
            continue
        flavour = _date_flavour(binary)
        seen.add(flavour)
        out = subprocess.run([binary, "+" + fmt], capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, (binary, flavour, out.stderr[:200])
        stamp = out.stdout.strip()
        # Python is the reader that matters: the artifacts are consumed by code, and
        # `fromisoformat` accepts %z with and without the colon (both verified).
        parsed = datetime.fromisoformat(stamp)
        assert parsed.tzinfo is not None, (binary, stamp)
        # ... and the flavour's OWN parser must accept it, since an operator reading
        # the log reaches for `date -j -f` (bsd) or `date -d` (gnu). Each branch runs
        # only against a binary PROBED to be that flavour.
        if flavour == "bsd":
            cmd = [binary, "-j", "-f", fmt, stamp, "+%s"]
        else:
            cmd = [binary, "-d", stamp, "+%s"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, (binary, flavour, stamp, p.stderr[:200])
    assert seen, "no date implementation was exercised"

    # No launcher may go back to the non-portable form. `%:z` is NOT the fix either:
    # BSD date prints it literally (`...20:37:37:z`, measured).
    here = SRC.parent
    offenders = [p.name for p in sorted(here.glob("serve_*.sh"))
                 if "date -Is" in p.read_text() or "date +%Y-%m-%dT%H:%M:%S%:z" in p.read_text()]
    assert not offenders, f"non-portable timestamp in {offenders}"


def test_the_flavour_probe_separates_the_two_parse_forms():
    """Pin the discriminator itself, so a label-based branch cannot come back.

    The ubuntu row went red because a BSD parse command ran against GNU date. Where
    both flavours exist this asserts each binary answers its own flavour and that the
    two parse forms really are distinct -- the fact the probe relies on.
    """
    fmt = "%Y-%m-%dT%H:%M:%S%z"
    flavours = {}
    for binary in ("date", "gdate"):
        if shutil.which(binary) is None:
            continue
        flavours[_date_flavour(binary)] = binary
    assert flavours, "no date implementation available"
    for flavour, binary in flavours.items():
        stamp = subprocess.run([binary, "+" + fmt], capture_output=True, text=True,
                               timeout=30).stdout.strip()
        own = (["-j", "-f", fmt, stamp, "+%s"] if flavour == "bsd"
               else ["-d", stamp, "+%s"])
        assert subprocess.run([binary, *own], capture_output=True,
                              timeout=30).returncode == 0, (flavour, own)
    # The two forms are not interchangeable: whatever else is on this host must
    # reject the form that is not its own.
    gnu_bin = flavours.get("gnu")
    if gnu_bin is not None:
        r = subprocess.run([gnu_bin, "-j", "-f", fmt, "2026-01-01", "+%s"],
                           capture_output=True, text=True, timeout=30)
        # The message names the invoked binary (`gdate: invalid option -- 'j'`), so
        # match the quoted flag, not the program name.
        assert r.returncode != 0 and "'j'" in r.stderr, r.stderr[:200]


def test_the_old_form_really_was_broken_on_this_platform():
    """Negative control for the gate above: if this platform's `date` accepts `-Is`,
    the test is vacuous here and the portability claim rests on nothing."""
    r = subprocess.run(["date", "-Is"], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        # GNU date accepts it, so on THIS host there is nothing to catch. The
        # macos-14 row is where this control has teeth; do not assert a failure
        # the platform cannot produce (that is how a gate goes red on CI alone).
        return
    assert "invalid argument" in r.stderr or "illegal" in r.stderr, r.stderr[:200]
