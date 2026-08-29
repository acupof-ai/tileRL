"""The bench snapshot is the repo's source of truth; the pod only raises rows.

``pod_sync.sh`` wipes the remote checkout on every sync, so a row raised on the
pod is lost unless it comes home first. This pulls the pod's snapshot and
merges it into the committed one — per key the HIGHER tok/s wins, which is the
same rule the harness gate applies in-process.

  python scripts/baseline.py pull       # merge the pod's rows into the repo's
  python scripts/baseline.py show
  python scripts/baseline.py selfcheck  # the hold rule, against real commits

Higher-wins is overridden when the repo CORRECTED a row after the pod measured
it: every row carries its commit, so a pod row whose commit is a proper
ancestor of the local row's is stale and is held, however fast it reads. That
is what makes a deliberate lowering stick — a row reseeded from a quiet host,
or one belonging to a reverted kernel, each crawled back three times on
2026-08-29 before this rule existed.

Merging is still per key, so a row DELETED from the repo comes back on the next
pull as long as the pod's copy still has it — six retired `spec/*` rows
resurrected that way. The pod's checkout is overwritten by every sync, so the
cure for a deletion is one `SKIP_BASELINE_PULL=1 scripts/pod_sync.sh`.
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
    """Was the repo's row committed strictly after the pod measured its own?

    Every row carries the commit it was measured at, so this is answerable
    exactly: the pod's row is stale when its commit is a proper ancestor of the
    local row's. Unknown or unresolvable commits fall back to higher-wins.
    """
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
            # The repo corrected this row AFTER the pod measured it. Higher-wins
            # would undo the correction, which is how a rejected kernel's number
            # and a load-inflated seed each crawled back three times.
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
    """The rule that keeps a correction from being undone, against real commits."""
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
