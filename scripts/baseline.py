"""Merge the pod's bench-baseline.json into the repo's: per key the higher tok/s wins, unless
the pod row's commit is a proper ancestor of the local row's (the repo corrected it later).
A row deleted locally returns on the next pull while the pod still has it; make the deletion
stick with `SKIP_BASELINE_PULL=1 scripts/pod_sync.sh`.

  python scripts/baseline.py pull|show|selfcheck
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

LOCAL = Path(__file__).resolve().parent.parent / "docs/experience/wins/bench-baseline.json"
REMOTE = "/work/tilerl/docs/experience/wins/bench-baseline.json"


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _tok_s_only(rows: dict) -> tuple[dict, list[tuple[str, list[str]]]]:
    """Split rows into the mergeable ones and the strays, which are NAMED not dropped.

    Every row here is higher-is-better tok/s (`cli.py:762`), and `pull` compares with
    `>`. One hand-written `secs_per_step` row on the pod raised `KeyError: 'tok_s'`
    and killed the sync for every session -- before the wipe, so nothing was
    half-synced, but no sync could run at all. Skipping silently would be the other
    failure: a pod-raised row that never reaches the repo and nobody notices.
    """
    keep, skip = {}, []
    for k, v in rows.items():
        (keep.__setitem__(k, v) if "tok_s" in v else skip.append((k, sorted(v))))
    return keep, skip


def _local_is_newer(remote_commit: str | None, local_commit: str | None) -> bool:
    """Pod row stale = its commit is a proper ancestor of the local row's; unknown -> higher-wins."""
    if not remote_commit or not local_commit or remote_commit == local_commit:
        return False
    if "unknown" in (remote_commit, local_commit):
        return False
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", remote_commit, local_commit],
        cwd=LOCAL.parent, capture_output=True,
    ).returncode == 0


def pull() -> int:
    launcher = Path.home() / "bin/pod"
    try:
        raw = subprocess.run([str(launcher), f"cat {REMOTE}"], capture_output=True, text=True)
    except OSError as e:  # ~/bin/pod is a symlink into another repo: absent when it moves
        print(f"pull: cannot run {launcher}: {e}", file=sys.stderr)
        return 1
    if raw.returncode != 0 or not raw.stdout.strip():
        print("pull: no remote snapshot", raw.stderr.strip()[:200], file=sys.stderr)
        return 1
    remote, local = json.loads(raw.stdout), _load(LOCAL)
    remote, skipped = _tok_s_only(remote)
    for k, keys in skipped:
        print(f"pull: skipped {k}: no tok_s, keys {keys}", file=sys.stderr)
    raised, held = [], []
    for k, v in remote.items():
        cur = local.get(k)
        if cur is not None and _local_is_newer(v.get("commit"), cur.get("commit")):
            if v["tok_s"] > cur["tok_s"]:
                held.append(f"  {v['tok_s']:.1f} -> kept {cur['tok_s']:.1f}  {k}")
            continue
        if cur is None or v["tok_s"] > cur["tok_s"]:
            was = f"{cur['tok_s']:.1f} -> " if cur else "new "
            raised.append(f"  {was}{v['tok_s']:.1f}  {k}")
            local[k] = v
    LOCAL.write_text(json.dumps(local, indent=2, sort_keys=True) + "\n")
    print(f"pulled {len(remote)} rows, {len(raised)} raised:")
    print("\n".join(raised) or "  (none)")
    if held:
        print(f"{len(held)} held (the repo corrected them after that run):")
        print("\n".join(held))
    return 0


def show() -> int:
    for k, v in sorted(_load(LOCAL).items()):
        print(f"  {k:<44} {v['tok_s']:>9.1f}  {v['date']} {v['commit']}")
    return 0


def _two_commits(d: Path) -> tuple[str, str]:
    """A throwaway repo with a known ancestry. The arms below used HEAD and HEAD~1,
    which needs history CI's depth-1 checkout does not have."""
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(d)]
    subprocess.run(git[:-2] + ["init", "-q", str(d)], check=True)
    out = []
    for msg in ("first", "second"):
        subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", msg], check=True)
        out.append(subprocess.check_output(git + ["rev-parse", "--short", "HEAD"],
                                          text=True).strip())
    return out[0], out[1]


def _selfcheck() -> int:
    global LOCAL
    was = LOCAL
    with tempfile.TemporaryDirectory() as d:
        prev, head = _two_commits(Path(d))
        LOCAL = Path(d) / LOCAL.name  # _local_is_newer resolves git from LOCAL.parent
        try:
            _check_ancestry(prev, head)
        finally:
            LOCAL = was
    _check_strays()
    print("selfcheck ok")
    return 0


def _check_ancestry(prev: str, head: str) -> None:
    assert _local_is_newer(prev, head), "a pod row measured one commit back must be held"
    assert not _local_is_newer(head, prev), "a pod row measured later must still raise"
    assert not _local_is_newer(head, head), "same commit is not newer"
    assert not _local_is_newer("unknown", head), "unknown provenance falls back to higher-wins"


def _check_strays() -> None:
    """A stray row is skipped and named, never merged and never fatal."""
    keep, skip = _tok_s_only({
        "suite/shape/sm90": {"commit": "abc", "tok_s": 1.0},
        "train/step/sm90": {"commit": "abc", "secs_per_step": 34.09},
    })
    assert list(keep) == ["suite/shape/sm90"], keep
    assert skip == [("train/step/sm90", ["commit", "secs_per_step"])], skip
    assert _tok_s_only({}) == ({}, []), "empty stays empty"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    sys.exit({"pull": pull, "show": show, "selfcheck": _selfcheck}[cmd]())
