"""A portable flock(1) so the launcher-script tests run on macOS too.

`scripts/serve_v100.sh` and `scripts/serve_hybrid_v100.sh` refuse to start
without flock(1), and they are right to: two supervisors on one card fight for
port 8000 and for the GPU. But flock(1) ships on Linux and not on macOS, so both
test files skipped every non-Linux row and their behaviour was only ever
exercised on CI's ubuntu job.

This is not a stub that pretends the lock succeeded. A shim that always exits 0
is worse than the skip: execution runs past the lock check into the restart
loop, and the failure that surfaces (originally a BSD `date -Is` complaint from
that branch -- since fixed to a portable format, so the tell is now just a
launcher that booted rather than refused) names neither the lock nor the
launcher. The lock-holder test in `test_serve_v100_sh.py` exists to catch
exactly that, and it goes red under such a shim.

`fcntl.flock` is the same lock the shell builtin takes, so a process holding it
here really does exclude a launcher that calls flock(1):

* Linux already has flock(1) -- remove the shim from PATH and use it, so the
  linux row keeps testing the real binary, including the `-n` argument form.
* macOS gets this shim, which implements the subset the launchers use: the
  `-n` flag and a numeric fd carrying a shell redirection. The lock is always
  exclusive, which is all a launcher needs; a shared (`-s`) request would need
  LOCK_SH and has no caller in this repo.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import tempfile

#: Faithful subset of flock(1): an exclusive lock on the given fd, honouring -n.
#: Exits 1 on a held lock, like the real binary, so `flock -n 9 ||` still refuses.
_SHIM = """\
#!{python}
import fcntl, sys

flags = 0
for a in sys.argv[1:]:
    if a == "-n":
        flags |= fcntl.LOCK_NB
fd = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 0
try:
    fcntl.flock(fd, fcntl.LOCK_EX | flags)
except OSError:
    sys.exit(1)
"""


class PortableFlock:
    """A PATH with flock(1) available, alive only inside the ``with``.

    The shim lives in a TemporaryDirectory, so the keeper must outlive every
    subprocess that resolves it. Returning the path and letting the caller drop
    the keeper deletes the shim before it is used -- hence the context manager.
    """

    def __init__(self) -> None:
        self._dir: tempfile.TemporaryDirectory | None = None
        self.path = os.environ["PATH"]

    def __enter__(self) -> str:
        if shutil.which("flock") is not None:
            return self.path  # real binary present (Linux): leave PATH alone
        self._dir = tempfile.TemporaryDirectory(prefix="flock_shim.")
        p = pathlib.Path(self._dir.name) / "flock"
        p.write_text(_SHIM.format(python=shutil.which("python3") or "/usr/bin/env python3"))
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.path = f"{self._dir.name}{os.pathsep}{os.environ['PATH']}"
        return self.path

    def __exit__(self, *exc: object) -> None:
        if self._dir is not None:
            self._dir.cleanup()
            self._dir = None


def flock_path():
    """``with flock_path() as path:`` -- `path` carries flock(1), real or shimmed."""
    return PortableFlock()
