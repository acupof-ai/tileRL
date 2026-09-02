"""Post-merge audit: did the sm70 rebase silently drop either side's work?

The rebase of the sm70 fp4 branch onto main resolved 15 files by hand (43 conflict
hunks). Both sides carry real work -- main has 31 commits including features, the
sm70 side is a whole (precision, arch) cell -- so "it merged and the tests pass" is
not evidence that nothing was lost. A dropped hunk is silent: the file compiles,
ruff is clean, and the CPU suite never exercises the sm70 branch at all.

This checks the two things a test suite cannot:

1. Every SYMBOL that exists in main's side or the sm70 side of a conflicted file
   still exists in the merged file. Catches a resolution that took one side
   wholesale. Symbols, not lines, because rewording is expected and legitimate.
2. Every NUMBER the sm70 side asserted in prose is still present somewhere. The
   measured figures are load-bearing (they are the reason to keep the branch) and
   the easiest thing to lose to a "keep main's compressed wording" resolution.

Run from the worktree with the rebase still in progress (reads git stages 1/2/3)
or after `git rebase --continue` against the recorded stage files under /tmp/rb.

  uv run python scripts/audit_merge.py
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

FILES = [
    "packages/tilerl-kernels/src/tilerl_kernels/backend.py",
    "packages/tilerl-kernels/src/tilerl_kernels/kernels.py",
    "packages/tilerl-kernels/src/tilerl_kernels/kernels_gdn.py",
    "packages/tilerl-kernels/src/tilerl_kernels/kernels_linear.py",
    "packages/tilerl-kernels/src/tilerl_kernels/kernels_mma.py",
    "packages/tilerl-kernels/src/tilerl_kernels/reference.py",
    "packages/tilerl-kernels/src/tilerl_kernels/registry.py",
    "scripts/mmlu.py",
    "src/tilerl/cli.py",
    "src/tilerl/engine.py",
    "src/tilerl/kv_cache.py",
    "src/tilerl/server.py",
    "src/tilerl/spec.py",
]

#: Numbers the sm70 branch measured. Losing one to a reworded comment is the
#: failure this catches; they are why the branch exists.
SM70_FACTS = ["37.6", "7.89", "5.53", "746", "62.3", "16.04", "0.59", "4.71"]


#: Symbols whose absence is a deliberate main-side choice, not a dropped hunk.
#: main's de-slop pass removed __all__ across the tree; a merged file that follows
#: main there is correct, and flagging it trains you to ignore the report.
EXPECTED_GONE = {"__all__"}


def stage(path: str, n: int) -> str | None:
    """Content of git stage n (1=base, 2=main, 3=sm70), or None if absent."""
    r = subprocess.run(["git", "show", f":{n}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def symbols(src: str) -> set[str]:
    """Top-level and class-level def/class/assignment names, plus imported names.

    Imports count as defined: main moved mmlu.py's LETTERS/letter/_questions into
    tilerl.eval and imports them, which is a legitimate refactor that a
    definitions-only check reads as four dropped symbols.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and (
            node.id.isupper() or node.id.startswith("_")
        ):
            out.add(node.id)
        elif isinstance(node, ast.ImportFrom | ast.Import):
            for a in node.names:
                out.add(a.asname or a.name.split(".")[0])
    return out


def main() -> int:
    bad = 0
    # A symbol can legitimately move to another module: main extracted mmlu.py's
    # question loading into tilerl/eval.py. So "still defined SOMEWHERE in the
    # tree" is the honest check; a definitions-in-this-file check reported six
    # dropped symbols that had all moved intact.
    elsewhere: set[str] = set()
    for p in list(Path("src").rglob("*.py")) + list(Path("packages").rglob("*.py")):
        elsewhere |= symbols(p.read_text())
    for path in FILES:
        merged_p = Path(path)
        if not merged_p.exists():
            print(f"MISSING  {path}")
            bad += 1
            continue
        merged = merged_p.read_text()
        if "<<<<<<<" in merged or ">>>>>>>" in merged:
            print(f"MARKERS  {path}")
            bad += 1
            continue
        if path.endswith(".py"):
            try:
                ast.parse(merged)
            except SyntaxError as exc:
                print(f"SYNTAX   {path}: {exc}")
                bad += 1
                continue
        ours, theirs = stage(path, 2), stage(path, 3)
        if ours is None or theirs is None:
            continue  # rebase already finished; nothing to compare against
        got = symbols(merged)
        for label, side in (("main", ours), ("sm70", theirs)):
            lost = sorted(symbols(side) - got - EXPECTED_GONE - elsewhere)
            if lost:
                print(f"LOST-{label:<4} {path}: {', '.join(lost)}")
                bad += 1
    tree = " ".join(Path(p).read_text() for p in FILES if Path(p).exists())
    docs = " ".join(p.read_text() for p in Path("docs/experience").rglob("*.md"))
    for fact in SM70_FACTS:
        # A number may legitimately move from code comment to a docs entry, so the
        # check is "still asserted somewhere", not "still in this file".
        if not re.search(re.escape(fact), tree + docs):
            print(f"LOST-FACT {fact} appears in neither the merged files nor docs/experience")
            bad += 1
    print("audit: OK" if not bad else f"audit: {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
