"""The pod sync wipes the remote checkout; runs/ must survive it.

`-delete` implies `-depth`, which disables `-prune`, so the obvious exemption spelling
deletes runs/ while reading as if it protects it. The expression is extracted from the
shipped script rather than copied, so a rewrite there is what this test sees.
"""

import subprocess
from pathlib import Path

import pod_sync_source as src

_wipe = src.wipe_expr


def _run(expr: str, tmp: Path) -> tuple[bool, bool]:
    (tmp / "runs" / "abc").mkdir(parents=True)
    (tmp / "src").mkdir()
    (tmp / "runs" / "abc" / "manifest.json").write_text("{}")
    (tmp / "src" / "x.py").write_text("x")
    subprocess.run(expr, shell=True, cwd=tmp, check=False, capture_output=True)
    return (tmp / "runs" / "abc" / "manifest.json").exists(), (tmp / "src" / "x.py").exists()


def test_the_wipe_keeps_runs_and_removes_everything_else(tmp_path):
    kept, other = _run(_wipe(), tmp_path / "a")
    assert kept, "runs/ was deleted: check the exemption is not -prune, which -delete disables"
    assert not other, "the wipe kept src/: it is no longer wiping the checkout"


def test_the_bare_wipe_deletes_runs(tmp_path):
    # negative control: without the exemption runs/ goes, so the assertion above can fail
    kept, other = _run("find . -mindepth 1 -delete", tmp_path / "b")
    assert not kept and not other


def test_prune_is_not_a_working_exemption(tmp_path):
    """The spelling this test exists to keep out of the script — on either find.

    The two implementations diverge, and only one of them is loud about it. GNU find
    (the pod, ubuntu CI) refuses: rc=1, nothing deleted, and it prints "the -delete
    action automatically turns on -depth, but -prune does nothing when -depth is in
    effect". BSD find (macOS) takes it and deletes runs/ anyway. So the assertion has
    to be "this does not wipe the checkout while keeping runs/", which is true both
    ways, rather than "runs/ is deleted", which is BSD-only.
    """
    kept, other = _run("find . -mindepth 1 -path ./runs -prune -o -delete", tmp_path / "c")
    assert not (kept and not other), "-prune became a working exemption; re-check the script"
