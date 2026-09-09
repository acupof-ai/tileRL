#!/usr/bin/env python3
"""Harvest bench rows from the pod store into the local tracked store.

Reverse-direction mirror of pod_run.sh's merge-by-id: pulls
/work/tilerl-bench/measurements.jsonl from the pod and appends rows the
local tracked store lacks by id.  Same id with different content is
refused, not overwritten.  Nothing is committed or pushed — the human
reviews the output and opens a PR.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_STORE = ROOT / "docs" / "experience" / "bench" / "measurements.jsonl"
POD_STORE_PATH = "/work/tilerl-bench/measurements.jsonl"


def fetch_pod_store() -> str:
    """Read the pod bench store via tn exec + crictl.

    Not pod_sync.sh — that overwrites the remote tree on every call.
    """
    remote = (
        'cid=$(crictl ps -q --name sglang-test --state Running 2>/dev/null | head -1); '
        'if [ -z "$cid" ]; then echo "harvest: container not Running" >&2; exit 1; fi; '
        f"crictl exec $cid cat {POD_STORE_PATH}"
    )
    r = subprocess.run(["tn", "exec", remote], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"harvest: fetch failed: {r.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return r.stdout


def harvest(local_path: Path, pod_text: str) -> int:
    """Append pod rows the local store lacks by id.

    Returns the number of rows appended.  Same id with different content
    is printed to stderr and refused — the local row is never overwritten.
    """
    local_by_id: dict[str, str] = {}
    if local_path.exists():
        for line in local_path.read_text().splitlines():
            if line.strip():
                local_by_id[json.loads(line)["id"]] = line

    new_lines: list[str] = []
    conflicts = 0
    pod_rows = 0
    for line in pod_text.splitlines():
        if not line.strip():
            continue
        pod_rows += 1
        row = json.loads(line)
        rid = row["id"]
        if rid in local_by_id:
            if json.loads(local_by_id[rid]) != row:
                print(
                    f"CONFLICT id={rid} metric={row.get('metric')} "
                    f"value={row.get('value')} — same id, different content; refused",
                    file=sys.stderr,
                )
                conflicts += 1
            continue
        print(f"append id={rid} metric={row.get('metric')} value={row.get('value')}")
        new_lines.append(line)
        local_by_id[rid] = line  # guard against duplicate ids within the pod store

    if new_lines:
        with open(local_path, "a") as f:
            f.writelines(line + "\n" for line in new_lines)

    print(f"harvest: read {pod_rows} pod rows, {len(new_lines)} new, {conflicts} conflicts")
    if conflicts:
        print(f"harvest: {conflicts} conflict(s) refused — review before committing", file=sys.stderr)
    return len(new_lines)


def main() -> None:
    harvest(LOCAL_STORE, fetch_pod_store())


if __name__ == "__main__":
    main()
