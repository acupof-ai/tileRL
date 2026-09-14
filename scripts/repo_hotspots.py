#!/usr/bin/env python3
"""Repository hotspots: churn x fan-in x LOC for src/tilerl modules.

Usage: python3 scripts/repo_hotspots.py [--since 2026-07-16] [--top 20]

churn   = commits in the window that touched the file (git log)
fan-in  = import sites across src, tests, scripts and packages
LOC     = current line count

A module high on churn and fan-in is where defects and merge conflicts land;
LOC alone measures size, not risk.
"""
from __future__ import annotations

import ast
import collections
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tilerl"


def churn(since: str) -> collections.Counter[str]:
    out = subprocess.run(
        ["git", "log", f"--since={since}", "--name-only", "--format=", "--", "src/tilerl"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    c: collections.Counter[str] = collections.Counter()
    for line in out.splitlines():
        if line.startswith("src/tilerl/"):
            c[pathlib.Path(line).name] += 1
    return c


def fan_in() -> collections.Counter[str]:
    edges: collections.Counter[str] = collections.Counter()
    names = {p.stem for p in SRC.glob("*.py")}
    for root in ("src", "tests", "scripts", "packages"):
        for p in (ROOT / root).rglob("*.py"):
            try:
                tree = ast.parse(p.read_text())
            except (SyntaxError, OSError):
                continue
            in_src = root == "src" and "tilerl_kernels" not in str(p)
            for n in ast.walk(tree):
                found = []
                if isinstance(n, ast.ImportFrom) and n.module:
                    if n.module.startswith("tilerl."):
                        found.append(n.module.split(".")[1])
                    elif n.level and in_src:
                        found.append(n.module)
                for m in found:
                    if m in names:
                        edges[m] += 1
    return edges


def main() -> None:
    since = "2026-07-16"
    top = 20
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--since":
            since = args[i + 1]
        if a == "--top":
            top = int(args[i + 1])
    ch, fi = churn(since), fan_in()
    rows = []
    for f in sorted(SRC.glob("*.py")):
        rows.append((ch[f.name], fi[f.stem], sum(1 for _ in open(f)), f.stem))
    rows.sort(key=lambda r: (r[0] * max(r[1], 1), r[2]), reverse=True)
    print(f"# since {since}  (score = churn x max(fan-in,1))")
    print(f"{'module':18} {'churn':>6} {'fan-in':>7} {'LOC':>6}")
    for c, imp, loc, name in rows[:top]:
        print(f"{name:18} {c:>6} {imp:>7} {loc:>6}")


if __name__ == "__main__":
    main()
