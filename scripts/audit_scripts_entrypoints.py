"""Phase 1 of the scripts/ audit: classify all 200 files against six entry-point sets.

The evidence standard is ENUMERATION, not search. For each set we build the set of scripts
it reaches, then a file's classification is set membership -- so a file is DEAD only when it
is absent from all six *enumerated* sets, and the zeros are reported. A keyword search would
only prove what I thought to search for.

Two traps this is built to avoid:

1. A module is imported by MODULE NAME, not filename. `import ab_gemv` and
   `from scripts.ab_gemv import x` and `runpy.run_path("scripts/ab_gemv.py")` all reach the
   same file by different spellings, so every set resolves to a stem and compares stems.
2. `docs/experience/` citing a script is PROVENANCE, not DEAD -- deleting it destroys the
   provenance of a number we still quote. That set is collected separately from the other
   doc mentions so the bucket cannot be diluted.

Run: python3 scripts/audit_scripts_entrypoints.py > audit.json
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
STEMS = {p.stem for p in SCRIPTS}


def _read(p: pathlib.Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def _files(globs: list[str], skip_scripts: bool = False) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for g in globs:
        for p in ROOT.glob(g):
            if p.is_file() and ".git/" not in str(p):
                if skip_scripts and p.parent.name == "scripts" and p.suffix == ".py":
                    continue
                out.append(p)
    return out


def _mentions(text: str) -> set[str]:
    """Stems this text refers to, by any spelling that would actually reach the file.

    Deliberately generous: a false LIVE costs a file we keep, a false DEAD costs a file we
    delete. The asymmetry says to over-collect here.
    """
    hits = set()
    for stem in STEMS:
        # `/` must NOT be in the lookbehind: the normal way to cite a script is
        # `scripts/<stem>.py`, and excluding a preceding slash rejected every one of them.
        # Caught by a negative control -- probe_kv_fp8_27b.py read DEAD while an entry cited
        # it by path.
        if re.search(rf"(?<![\w.-]){re.escape(stem)}(?![\w-])", text):
            hits.add(stem)
    return hits


def set1_project_scripts() -> set[str]:
    """pyproject.toml entry points."""
    return _mentions(_read(ROOT / "pyproject.toml"))


def set2_ci() -> set[str]:
    """.github/workflows/*.yml"""
    hits = set()
    for p in _files([".github/workflows/*.yml", ".github/workflows/*.yaml"]):
        hits |= _mentions(_read(p))
    return hits


def set3_imports() -> dict[str, set[str]]:
    """Python imports across src/, tests/, packages/, scripts/ -- by MODULE NAME.

    Returns stem -> the set of FILES that reach it, because a file must not count as its own
    importer: every script's docstring says "Run: scripts/<self>.py", so a self-mention made
    113 of 200 files look imported. The caller drops the self-edge.
    """
    hits: dict[str, set[str]] = {}

    def add(stem: str, src_path: pathlib.Path) -> None:
        if stem in STEMS:
            hits.setdefault(stem, set()).add(str(src_path.relative_to(ROOT)))

    for p in _files(["src/**/*.py", "tests/**/*.py", "packages/**/*.py", "scripts/**/*.py"]):
        src = _read(p)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            for stem in _mentions(src):
                add(stem, p)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    add(a.name.split(".")[-1], p)
                    add(a.name.split(".")[0], p)
            elif isinstance(node, ast.ImportFrom) and node.module:
                add(node.module.split(".")[-1], p)
                add(node.module.split(".")[0], p)
                for a in node.names:
                    add(a.name, p)
        # runpy / exec / subprocess reaching a path: textual, since it is not an import node
        for m in re.finditer(r"scripts/([A-Za-z0-9_]+)\.py", src):
            add(m.group(1), p)
    return hits


def set4_experience() -> set[str]:
    """docs/experience/** -- PROVENANCE. Kept separate so the bucket cannot be diluted."""
    hits = set()
    for p in _files(["docs/experience/**/*.md", "docs/experience/*.md"]):
        hits |= _mentions(_read(p))
    return hits


def set4b_other_docs() -> set[str]:
    """docs/** outside experience/, plus CHANGELOG."""
    hits = set()
    for p in _files(["docs/**/*.md", "docs/*.md", "CHANGELOG.md"]):
        if "experience" in str(p):
            continue
        hits |= _mentions(_read(p))
    return hits


def set5_invocation() -> set[str]:
    """Shell, Makefile, POD-VERIFY.md -- a script invoked by another script."""
    hits = set()
    for p in _files(["scripts/*.sh", "*.sh", "Makefile", "POD-VERIFY.md", "**/*.sh"]):
        hits |= _mentions(_read(p))
    return hits


def set6_readmes() -> set[str]:
    hits = set()
    for p in _files(["README.md", "AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md"]):
        hits |= _mentions(_read(p))
    return hits


def main() -> int:
    # A wrong ROOT globs nothing and every bucket reads 0 -- silently, since an empty
    # enumeration is indistinguishable from "nothing is reachable".
    if len(SCRIPTS) < 50:
        raise SystemExit(f"only {len(SCRIPTS)} scripts under {ROOT}/scripts -- wrong ROOT?")
    imports = set3_imports()
    sets = {
        "1_pyproject": set1_project_scripts(),
        "2_ci": set2_ci(),
        "4_experience": set4_experience(),
        "4b_other_docs": set4b_other_docs(),
        "5_invocation": set5_invocation(),
        "6_readmes": set6_readmes(),
    }

    rows = []
    for p in SCRIPTS:
        stem = p.stem
        # The self-edge is dropped: a file importing or naming itself is not a reference.
        # Every script's docstring carries "Run: scripts/<self>.py", which made 113 of 200
        # look imported before this.
        importers = sorted(imports.get(stem, set()) - {f"scripts/{p.name}"})
        where = {k: (stem in v) for k, v in sets.items()}
        where["3_imports"] = bool(importers)
        prov = where["4_experience"]
        live = any(where[k] for k in ("1_pyproject", "2_ci", "3_imports", "5_invocation",
                                      "6_readmes", "4b_other_docs"))
        rows.append({
            "file": f"scripts/{p.name}",
            "lines": len(_read(p).splitlines()),
            "sets": sorted(k for k, v in where.items() if v),
            "importers": importers[:4],
            "bucket": "LIVE" if live else ("PROVENANCE" if prov else "DEAD"),
        })

    out = {
        "head": "origin/main",
        "total_files": len(SCRIPTS),
        "total_lines": sum(r["lines"] for r in rows),
        "set_sizes": {k: len(v & STEMS) for k, v in sets.items()}
        | {"3_imports": sum(1 for r in rows if "3_imports" in r["sets"])},
        "buckets": {b: sum(1 for r in rows if r["bucket"] == b)
                    for b in ("LIVE", "PROVENANCE", "DEAD")},
        "dead_lines": sum(r["lines"] for r in rows if r["bucket"] == "DEAD"),
        "provenance_lines": sum(r["lines"] for r in rows if r["bucket"] == "PROVENANCE"),
        "rows": rows,
    }
    json.dump(out, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
