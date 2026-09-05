"""The run ledger: one ``manifest.json`` per run under ``$TILERL_RUNS``
(default ``./runs``). ``id = hash(inputs)``, so a rerun is a no-op and a changed
input is a new run. Gates are data here and exit codes in the CLI. Stdlib only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def runs_root() -> Path:
    return Path(os.environ.get("TILERL_RUNS", "runs"))


def run_id(inputs: dict) -> str:
    """First 12 hex of sha256 over canonical JSON: key order does not matter."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]


def file_hash(path: str | os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def new_manifest(command: str, inputs: dict, parents: list[str] | None = None) -> dict:
    return {"id": run_id(inputs), "command": command, "inputs": inputs,
            "parents": list(parents or ()), "commit": inputs.get("commit", commit()),
            "started": now(),
            "finished": None, "metrics": {}, "gates": [], "artifacts": {}}


def write_manifest(root: str | os.PathLike, m: dict) -> Path:
    d = Path(root) / m["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    return d


def read_manifest(root: str | os.PathLike, id: str) -> dict | None:
    p = Path(root) / id / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def list_runs(root: str | os.PathLike) -> list[dict]:
    """Newest first."""
    ms = [json.loads(p.read_text()) for p in Path(root).glob("*/manifest.json")]
    return sorted(ms, key=lambda m: m["finished"] or m["started"], reverse=True)


def lineage(root: str | os.PathLike, id: str) -> list[dict]:
    """The run, then its parents, breadth first."""
    out: list[dict] = []
    todo = [id]
    while todo:
        m = read_manifest(root, todo.pop(0))
        if m and all(m["id"] != x["id"] for x in out):
            out.append(m)
            todo += m["parents"]
    return out


def gates_pass(m: dict) -> bool:
    return all(g.get("skipped", False) or g["passed"] for g in m["gates"])


def format_run(m: dict) -> str:
    mt = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                  for k, v in m["metrics"].items() if v is not None)
    verdict = "skip" if m["gates"] and all(g.get("skipped", False) for g in m["gates"]) else "pass" if gates_pass(m) else "FAIL"
    return f"{m['id']}  {m['command']:<6} {m['finished'] or 'running':<25} {verdict:<5} {mt}"


if __name__ == "__main__":  # runnable check
    assert run_id({"a": 1, "b": [2]}) == run_id({"b": [2], "a": 1})
    assert run_id({"a": 1}) != run_id({"a": 2})
    print("ledger: ids OK")
