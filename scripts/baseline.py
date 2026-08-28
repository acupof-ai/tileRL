"""The bench snapshot is the repo's source of truth; the pod only raises rows.

``pod_sync.sh`` wipes the remote checkout on every sync, so a row raised on the
pod is lost unless it comes home first. This pulls the pod's snapshot and
merges it into the committed one — per key the HIGHER tok/s wins, which is the
same rule the harness gate applies in-process.

  python scripts/baseline.py pull     # merge the pod's rows into the repo's
  python scripts/baseline.py show
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


def pull() -> int:
    raw = subprocess.run(
        [str(Path.home() / "bin/pod"), f"cat {REMOTE}"], capture_output=True, text=True
    )
    if raw.returncode != 0 or not raw.stdout.strip():
        print("pull: no remote snapshot", raw.stderr.strip()[:200])
        return 1
    remote, local = json.loads(raw.stdout), _load(LOCAL)
    raised = []
    for k, v in remote.items():
        cur = local.get(k)
        if cur is None or v["tok_s"] > cur["tok_s"]:
            was = f"{cur['tok_s']:.1f} -> " if cur else "new "
            raised.append(f"  {was}{v['tok_s']:.1f}  {k}")
            local[k] = v
    LOCAL.write_text(json.dumps(local, indent=2, sort_keys=True) + "\n")
    print(f"pulled {len(remote)} rows, {len(raised)} raised:")
    print("\n".join(raised) or "  (none)")
    return 0


def show() -> int:
    for k, v in sorted(_load(LOCAL).items()):
        print(f"  {k:<44} {v['tok_s']:>9.1f}  {v['date']} {v['commit']}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    sys.exit({"pull": pull, "show": show}[cmd]())
