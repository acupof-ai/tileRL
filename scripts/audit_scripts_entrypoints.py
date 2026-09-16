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

#: Hand-run tools the seven reachability sets structurally cannot see: no test,
#: doc, CI job, import or glob names them, but a person invokes them by hand on
#: a card, a deploy box or in an ad-hoc review. Deleting one makes the next
#: person rewrite it -- proven for repo_hotspots (#595 shipped it explicitly as
#: "dev-only tooling"). Each line is a review point: the reason must say WHO
#: runs it and ON WHICH WORK LINE, never "useful script". A name here is kept
#: only while the reason holds; the closure gate fails if a name is added
#: without a reason or a script this reaches is deleted without dropping the
#: name. May only shrink by intent.
MANUAL_KEEP: dict[str, str] = {
    "compile_gate_sm70": "ops runs on V100 sm70: JIT-compiles every kernel the "
                         "sm70 fp4 cell dispatches to catch a deleted CUDA extern "
                         "(CUDA externs are Python string constants, invisible to "
                         "grep). Card gate, no CPU twin.",
    "parity_dequant_fp4": "kernel parity probe for the fused frozen-base backward "
                          "(dequant+gemm_nn block16); run by hand on the GPU box "
                          "(scripts header: CUDA_VISIBLE_DEVICES=7). Card only.",
    "quantize_nvfp4": "deploy path: stream-quantizes a bf16 HF checkpoint to "
                      "tilerl NVFP4 one shard at a time to fit the 31GB V100 "
                      "box; output is what load_hf(fp4=True) reads. Run per "
                      "checkpoint by hand.",
    "pod_portcheck": "pre-port gate ops runs on a card: compiles the upstream "
                     "tilelang corpus we copy kernels from against our pinned "
                     "tilelang (scripts/_portcheck_corpus is gitignored, a local "
                     "clone). 15/15 on 0.1.13/sm90.",
    "tp_parity": "tensor-parallel correctness gate, CPU+gloo via "
                 "`torchrun --nproc_per_node=2`; hand-run TP check (no pytest "
                 "wrapper), deliberately not a single-process test.",
    "pod_sync_check": "ops hand check that a non-git V100 pod tree matches the "
                      "push checkout before trusting a run (the pod is fed by "
                      "scp, so it converges to a mix of commits).",
    "repo_hotspots": "ad-hoc review tool: churn x fan-in x LOC for src/tilerl "
                     "modules over a git window. Shipped #595 expressly as "
                     "dev-only tooling, read-only, zero deps.",
    "floor_diff": "P1 eval analysis tool (#356): gross/net flips between two "
                  "same-weights eval arms paired by row index. Hand-run on "
                  "eval jsonl by the eval work line; sibling of probe_math_boxed.",
    "paired_2x2": "P1 eval analysis tool (#356): kept/lost/fixed/untouched + "
                  "at_cap transitions for before/after eval arms. Hand-run on "
                  "eval jsonl; sibling of probe_math_boxed.",
    "gate_margin_report": "P1 observational margin report (#381): how close "
                          "each gate's operand came to its threshold across "
                          "recorded runs, to tell a live gate from a decorative "
                          "one. Hand-run over the measurements store.",
    "probe_draft_window_sweep": "perf1 runs on the V100 sm70 sparse 27B line as "
                                "the #684 device acceptance tool: in-process "
                                "W=0/1k/2k/4k/8k paired arms measuring draft "
                                "CUDA ms, spec acceptance and tok/s, with a "
                                "three-state read_window_stats proof that the "
                                "window really truncated. Manual run after "
                                "deploy, --time-draft; no CPU twin.",
}



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
        # importlib.util.spec_from_file_location(name, path): neither arg is an import
        # node, so without this a test that loads scripts/board.py via spec read DEAD.
        # Path spellings handled: a ".../scripts/x.py" literal, and
        # os.path.join(..., "scripts", "x.py"); other dynamic paths are not resolvable.
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "spec_from_file_location"
            ):
                continue
            if node.args:  # the explicit module name ("board", "benchrec", ...)
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    add(first.value, p)
            if len(node.args) > 1:
                path_arg = node.args[1]
                if isinstance(path_arg, ast.Constant):
                    m = re.search(r"scripts[/\\]([A-Za-z0-9_]+)\.py$", path_arg.value)
                    if m:
                        add(m.group(1), p)
                elif isinstance(path_arg, ast.Call):
                    consts = [
                        a.value
                        for a in path_arg.args
                        if isinstance(a, ast.Constant) and isinstance(a.value, str)
                    ]
                    if "scripts" in consts:
                        for c in consts:
                            if c.endswith(".py"):
                                add(c[:-3], p)
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


def set7_selfcheck() -> set[str]:
    """Scripts tests/test_main_selfchecks.py executes: a hermetic (no torch/
    backend/build_engine) script whose ``__main__`` block carries an assert.

    The architecture keep-rule names this glob explicitly ("... or
    test_main_selfchecks glob reaches"), but the original six sets did not model
    it, so a script reached ONLY by that glob read DEAD. The reachability test is
    content-derived (same filter as the test), not a name list.
    """
    import ast as _ast

    hits = set()
    for p in SCRIPTS:
        src = _read(p)
        if any(k in src for k in ("import torch", "get_backend", "build_engine")):
            continue
        try:
            tree = _ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if (isinstance(node, _ast.If) and "__main__" in _ast.unparse(node.test)
                    and any(isinstance(n, _ast.Assert) for n in _ast.walk(node))):
                hits.add(p.stem)
                break
    return hits


def audit() -> dict:
    """Classify every script. Same data whether printed to JSON or imported by
    the closure gate (tests/test_scripts_closure.py)."""
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
        "7_selfcheck": set7_selfcheck(),
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
        reached = any(where[k] for k in ("1_pyproject", "2_ci", "3_imports",
                                         "5_invocation", "6_readmes",
                                         "4b_other_docs", "7_selfcheck"))
        if reached:
            bucket = "LIVE"
        elif prov:
            bucket = "PROVENANCE"
        elif stem in MANUAL_KEEP:
            bucket = "MANUAL_KEEP"
        else:
            bucket = "DEAD"
        rows.append({
            "file": f"scripts/{p.name}",
            "stem": stem,
            "lines": len(_read(p).splitlines()),
            "sets": sorted(k for k, v in where.items() if v),
            "importers": importers[:4],
            "bucket": bucket,
        })

    return {
        "head": "origin/main",
        "total_files": len(SCRIPTS),
        "total_lines": sum(r["lines"] for r in rows),
        "set_sizes": {k: len(v & STEMS) for k, v in sets.items()}
        | {"3_imports": sum(1 for r in rows if "3_imports" in r["sets"])},
        "buckets": {b: sum(1 for r in rows if r["bucket"] == b)
                    for b in ("LIVE", "PROVENANCE", "MANUAL_KEEP", "DEAD")},
        "dead_lines": sum(r["lines"] for r in rows if r["bucket"] == "DEAD"),
        "provenance_lines": sum(r["lines"] for r in rows if r["bucket"] == "PROVENANCE"),
        "rows": rows,
    }


def main() -> int:
    json.dump(audit(), sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
