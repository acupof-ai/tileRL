"""The baseline rescue must fail loud, because failing quiet loses a bench row.

`pod_sync.sh` does three things in order: pull the pod's bench-baseline.json into this
tree, wipe the remote checkout, extract this tree's tarball over it. Step 3 overwrites the
pod's copy with the local one, so if step 1 fails and the sync continues, any row the pod
raised is gone -- the wipe is not what loses it, the overwrite is.

Two halves, each with a red control:
  - pull() returns nonzero instead of raising when the launcher is missing
  - the sync line has no `|| true`, so `set -e` aborts before the wipe
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pod_sync_source as src

ROOT = Path(__file__).resolve().parents[1]
BASE = "docs/experience/wins/bench-baseline.json"

_PULL = (
    "import sys, pathlib, unittest.mock as m; sys.path.insert(0, {scripts!r});"
    "import baseline;"
    "h = staticmethod(lambda: pathlib.Path({missing!r}));"
    "m.patch.object(pathlib.Path, 'home', h).start();"
    "print(baseline.pull())"
)


def _pull_with_missing_launcher(src: str | None = None) -> str:
    """Run baseline.pull() with ~/bin/pod pointed at nothing.

    Returns the printed rc, or "raised:<ExceptionName>" — the name matters: a bare
    "it exited nonzero" would let an import error pass the control by the wrong route.
    """
    with tempfile.TemporaryDirectory() as td:
        scripts = Path(td) / "scripts"
        scripts.mkdir()
        (scripts / "baseline.py").write_text(src or (ROOT / "scripts" / "baseline.py").read_text())
        out = subprocess.run(
            [sys.executable, "-c", _PULL.format(scripts=str(scripts), missing=td)],
            capture_output=True, text=True,
        )
        if out.returncode == 0:
            return out.stdout.strip()
        last = [ln for ln in out.stderr.strip().splitlines() if ln and not ln[0].isspace()]
        return f"raised:{last[-1].split(':')[0]}" if last else "raised:?"


def test_pull_returns_nonzero_when_the_launcher_is_missing():
    assert _pull_with_missing_launcher() == "1"


def test_the_unguarded_pull_raises():
    # red control: without the OSError guard this is an uncaught FileNotFoundError, which
    # `|| true` in the caller made indistinguishable from a clean skip
    src = (ROOT / "scripts" / "baseline.py").read_text()
    old = src.replace("    except OSError as e:", "    except ZeroDivisionError as e:")
    assert old != src, "the guard's except line moved; this control is stale"
    assert _pull_with_missing_launcher(old) == "raised:FileNotFoundError"


def _sync_line() -> str:
    return src.line("baseline.py", "SKIP_BASELINE_PULL")


def _reaches_the_wipe(line: str, tmp: Path, skip: str = "0") -> bool:
    """Run the sync line with a pull that fails; did execution continue past it?"""
    tmp.mkdir(parents=True)
    (tmp / "scripts").mkdir()
    (tmp / "scripts" / "baseline.py").write_text("import sys; sys.exit(1)\n")
    (tmp / "s.sh").write_text(f'set -euo pipefail\nROOT="{tmp}"\n{line}\necho REACHED\n')
    out = subprocess.run(["bash", "s.sh"], cwd=tmp, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "SKIP_BASELINE_PULL": skip})
    return "REACHED" in out.stdout


def test_a_failed_pull_aborts_before_the_wipe(tmp_path):
    assert not _reaches_the_wipe(_sync_line(), tmp_path / "a")


def test_skip_baseline_pull_still_short_circuits(tmp_path):
    # the documented deliberate overwrite must not be broken by the abort
    assert _reaches_the_wipe(_sync_line(), tmp_path / "b", skip="1")


def test_the_or_true_spelling_continues(tmp_path):
    # red control: the spelling this fix removes swallows the failure and syncs anyway
    old = _sync_line().replace(">/dev/null", ">/dev/null 2>&1 || true")
    assert _reaches_the_wipe(old, tmp_path / "c")


def test_the_overwrite_is_what_loses_the_row(tmp_path):
    """The three steps in order: a silently-failed pull leaves the pod with the local row.

    This is the consequence the two guards above exist to prevent, asserted on the
    artifact rather than on the exit codes.
    """
    def sim(pull_works: bool, root: Path) -> dict:
        local, pod = root / "local", root / "pod"
        for side in (local, pod):
            (side / BASE).parent.mkdir(parents=True)
        (local / BASE).write_text(json.dumps({"d/a/sm70": {"tok_s": 100.0, "commit": "aaa"}}))
        (pod / BASE).write_text(json.dumps({"d/a/sm70": {"tok_s": 140.0, "commit": "bbb"},
                                            "d/b/sm70": {"tok_s": 55.0, "commit": "bbb"}}))
        if pull_works:
            loc = json.loads((local / BASE).read_text())
            for k, v in json.loads((pod / BASE).read_text()).items():
                if k not in loc or v["tok_s"] > loc[k]["tok_s"]:
                    loc[k] = v
            (local / BASE).write_text(json.dumps(loc))
        subprocess.run(["find", ".", "-mindepth", "1", "-delete"], cwd=pod, capture_output=True)
        shutil.copytree(local, pod, dirs_exist_ok=True)
        return json.loads((pod / BASE).read_text())

    good = sim(True, tmp_path / "ok")
    lost = sim(False, tmp_path / "bad")
    assert good["d/a/sm70"]["tok_s"] == 140.0 and len(good) == 2
    assert lost["d/a/sm70"]["tok_s"] == 100.0 and len(lost) == 1, "the loss path changed"


def test_a_baseline_row_without_tok_s_is_skipped_and_named():
    """`pull` compares rows with `>` on tok_s, so a row lacking it used to raise
    KeyError and kill the sync for every session — before the wipe, so nothing was
    half-synced, but no sync could run at all. One hand-written `secs_per_step`
    row on the pod did exactly that.

    Skipping silently would be the other failure, so the stray is NAMED on stderr
    and never merged. Driven through `baseline.py selfcheck` rather than importing
    the function, because the selfcheck is otherwise reachable only by typing it.
    """
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "baseline.py"), "selfcheck"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "selfcheck ok" in out.stdout, out.stdout

    # And the shipped baseline has no stray of its own: the row this fixed is gone,
    # so a reader of the file cannot reintroduce the shape by copying a neighbour.
    rows = json.loads((ROOT / BASE).read_text())
    strays = {k: sorted(v) for k, v in rows.items() if "tok_s" not in v}
    assert not strays, f"bench-baseline.json rows without tok_s: {strays}"
