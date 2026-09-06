"""A bench row must never carry a blank commit.

Two halves, one defect: `git ... > .synced_commit || true` truncates the file before git
runs, so a git failure leaves it EMPTY; and `_git_commit` used `exists()` rather than the
content, so empty read back as `''` and a bench row got a blank commit instead of
"unknown". Each half is tested against the shipped code, with the old spelling as a red
control so a passing assertion cannot be vacuous.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "bench_harness.py"

_READ = (
    "import sys; sys.path.insert(0,'scripts');"
    "import bench_harness as b; print(b._git_commit())"
)


def _commit_for(stamp: str | None, tmp: Path, harness_src: str | None = None) -> str:
    """What does _git_commit() report in a tree with no .git and this stamp content?"""
    tmp.mkdir(parents=True)
    # `git -C` walks UP: a repo anywhere above tmp would make rev-parse succeed and the
    # stamp path never run. Measured: from a non-repo subdir of a repo it returns that
    # repo's sha. pytest's tmp base has none, and this assert is what says so out loud.
    assert not any((p / ".git").exists() for p in [tmp, *tmp.parents]), f"repo above {tmp}"
    (tmp / "scripts").mkdir()
    (tmp / "scripts" / "bench_harness.py").write_text(harness_src or HARNESS.read_text())
    if stamp is not None:
        (tmp / ".synced_commit").write_text(stamp)
    out = subprocess.run(
        [sys.executable, "-c", _READ], cwd=tmp, capture_output=True, text=True,
        env={"HOME": str(tmp), "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_an_empty_stamp_reads_as_unknown(tmp_path):
    assert _commit_for("", tmp_path / "empty") == "unknown"
    assert _commit_for("   \n", tmp_path / "blank") == "unknown"
    assert _commit_for(None, tmp_path / "absent") == "unknown"
    assert _commit_for("faae3c8\n", tmp_path / "ok") == "faae3c8"


def test_the_exists_spelling_reports_a_blank_commit(tmp_path):
    # red control: the code this fix replaced, so the assertion above is not vacuous
    fixed = 'return (stamp.read_text().strip() if stamp.exists() else "") or "unknown"'
    src = HARNESS.read_text()
    assert src.count(fixed) == 1, "the shipped return line moved; this control is stale"
    old = src.replace(fixed, 'return stamp.read_text().strip() if stamp.exists() else "unknown"')
    assert _commit_for("", tmp_path / "ctl", old) == ""


def _stamp_block() -> str:
    lines = (ROOT / "scripts" / "pod_sync.sh").read_text().splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith("# the pod is not a git repo"))
    j = next(n for n in range(i, len(lines)) if lines[n] == "fi")
    return "\n".join(lines[i : j + 1])


def _stamp_after(block: str, tmp: Path) -> str:
    """Run the stamp block where git must fail, with a previous stamp in place."""
    tmp.mkdir(parents=True)
    assert not any((p / ".git").exists() for p in [tmp, *tmp.parents]), f"repo above {tmp}"
    (tmp / ".synced_commit").write_text("faae3c8\n")
    (tmp / "s.sh").write_text(block)
    subprocess.run(["bash", "s.sh"], cwd=tmp, env={"ROOT": str(tmp), "PATH": "/usr/bin:/bin"},
                   capture_output=True)
    return (tmp / ".synced_commit").read_text()


def test_a_git_failure_leaves_the_previous_stamp(tmp_path):
    assert _stamp_after(_stamp_block(), tmp_path / "fixed").strip() == "faae3c8"


def test_the_redirect_spelling_blanks_the_stamp(tmp_path):
    # red control: the one-liner this fix replaced truncates before git runs
    old = 'git -C "$ROOT" rev-parse --short HEAD > "$ROOT/.synced_commit" 2>/dev/null || true'
    assert _stamp_after(old, tmp_path / "ctl") == ""
