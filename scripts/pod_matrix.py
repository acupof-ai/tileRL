#!/usr/bin/env python3
"""Pod-side matrix runner: N variants in one pod session, one sync.

Spec (JSON file path, or '-' for stdin):
  {"command": "python3 scripts/_ab_gemv_direct.py 0",
   "variants": [{"name": "prmt_bm16", "env": {"ARMS": "prmt", "BM": "16"}},
                {"name": "shipped", "env": {"ARMS": "gemv"}}],
   "timeout": 1800}

Each variant runs ``command`` as a subprocess with its env overlay on top of
the runner's own environment. Per variant: exit code, wall seconds, and the
last 4 KB of combined output append to ``results.jsonl`` next to the spec;
a summary table prints at the end. On exit a ``matrix.done`` stamp carries
the overall status, so a detached run (setsid + this script, see
scripts/_pod_baseline_launch.sh) survives the launching agent's turn — poll
the stamp, then read the JSONL.

The JIT cache (/work/tilelang_cache) is on-disk, so variants in one session
warm each other: put the slowest-compiling variant first.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

TAIL_BYTES = 4096


def _run_variant(command: str, env: dict, timeout: float) -> dict:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        out = proc.stdout.decode(errors="replace")
        return {"exit": proc.returncode, "seconds": round(time.perf_counter() - t0, 1), "tail": out[-TAIL_BYTES:]}
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else ""
        return {"exit": "timeout", "seconds": round(time.perf_counter() - t0, 1), "tail": out[-TAIL_BYTES:]}


def main() -> int:
    spec_path = Path(sys.argv[1] if len(sys.argv) > 1 else "-")
    spec = json.loads(sys.stdin.read() if spec_path == "-" else spec_path.read_text())
    command = spec["command"]
    variants = spec["variants"]
    timeout = float(spec.get("timeout", 1800))
    out_dir = Path(spec.get("out_dir", "."))
    results_path = out_dir / "results.jsonl"
    done_path = out_dir / "matrix.done"

    rows = []
    for v in variants:
        row = {"name": v["name"], **_run_variant(command, v.get("env", {}), timeout)}
        rows.append(row)
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{row['name']}] exit={row['exit']} {row['seconds']}s", flush=True)

    print("\nname                  exit      seconds")
    print("-" * 44)
    for r in rows:
        print(f"{r['name']:<20}  {str(r['exit']):<6}  {r['seconds']:>8}")

    failed = [r["name"] for r in rows if r["exit"] not in (0,)]
    done_path.write_text("ok\n" if not failed else f"failed: {','.join(failed)}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
