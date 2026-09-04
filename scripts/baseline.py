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
from pathlib import Path

LOCAL = Path(__file__).resolve().parent.parent / "docs/experience/wins/bench-baseline.json"
REMOTE = "/work/tilerl/docs/experience/wins/bench-baseline.json"


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


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
    raw = subprocess.run(
        [str(Path.home() / "bin/pod"), f"cat {REMOTE}"], capture_output=True, text=True
    )
    if raw.returncode != 0 or not raw.stdout.strip():
        print("pull: no remote snapshot", raw.stderr.strip()[:200])
        return 1
    remote, local = json.loads(raw.stdout), _load(LOCAL)
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


def _selfcheck() -> int:
    head, prev = (
        subprocess.check_output(["git", "rev-parse", "--short", r], cwd=LOCAL.parent,
                                text=True).strip()
        for r in ("HEAD", "HEAD~1")
    )
    assert _local_is_newer(prev, head), "a pod row measured one commit back must be held"
    assert not _local_is_newer(head, prev), "a pod row measured later must still raise"
    assert not _local_is_newer(head, head), "same commit is not newer"
    assert not _local_is_newer("unknown", head), "unknown provenance falls back to higher-wins"
    print("selfcheck ok")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    sys.exit({"pull": pull, "show": show, "selfcheck": _selfcheck}[cmd]())
