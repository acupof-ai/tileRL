"""Does every markdown reference in docs/ resolve to a file that exists?

main carried 35 dead cross-references on 2026-09-05; a docs reorg plus two sessions took
it to 0. Nothing was checking, so they accumulated one at a time -- each written by
someone recalling a filename rather than reading it.

Two of the three that needed a human to resolve were unresolvable by name at all: no such
file had ever existed, and the intended target had to be found by a content signature in
the citing sentence. That is the cost this gate exists to avoid paying again.

Entries reference siblings as `wins/x.md` / `errors/x.md`, relative to docs/experience/
rather than to the citing file, so the resolver tries several bases. A first version of
this scan omitted that one and reported 103 dead links against a real 3 -- the tool was
the defect, which is why the base list is spelled out rather than inferred.

`docs/analysis/` is deliberately NOT in that list. A bare-name reference should resolve
only near where it was written; when a file moves to another directory the citing text has
to be updated, and the point of this gate is to say so. Adding the new directory would
have made the one real dead link on main resolve itself -- weakening the check rather than
fixing the tool.

Existence is asked of `git ls-files`, not of the filesystem, because the filesystem
answers for the developer's machine and CI checks out git. Measured: this gate passed
locally and failed on ubuntu because a file that #73 moved to docs/analysis/ was still
sitting untracked in my docs/experience/ from before the move. Verified the other way too,
in a throwaway clone of origin/main: the git-based version reports the 4 dead links CI
sees, including the one the filesystem version could not.
"""

import os
import re
import subprocess
from pathlib import Path

DOCS = Path(__file__).parent.parent / "docs"
ROOT = DOCS.parent

#: `[text](path.md)`, then the two bare shapes entries use inside backticks.
_LINK = re.compile(r"\[[^\]]*\]\(([^)#]+\.md)[^)]*\)")
_TICK = re.compile(r"`((?:docs/|\.\./)?(?:experience/)?(?:wins|errors)/[\w./-]+\.md)`")
_BARE = re.compile(r"`(20\d\d-\d\d-\d\d-[\w.-]+\.md)`")


def _tracked() -> set[str]:
    """Repo-relative paths git knows about, which is what CI checks out.

    Existence is asked of git, not of the filesystem. A stale untracked copy of a moved
    file makes the filesystem answer yes where CI answers no -- measured: this gate passed
    locally and failed on ubuntu for exactly that reason, because a file #73 moved to
    docs/analysis/ was still sitting in my docs/experience/ from before the move.
    """
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", "docs", "*.md"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


def _dead() -> list[str]:
    tracked = _tracked()
    out = []
    for rel in sorted(p for p in tracked if p.endswith(".md") and p.startswith("docs/")):
        md = ROOT / rel
        if not md.exists():  # tracked but not checked out (sparse checkout)
            continue
        text = md.read_text(errors="ignore")
        for m in (*_LINK.finditer(text), *_TICK.finditer(text), *_BARE.finditer(text)):
            raw = m.group(1)
            bases = (
                Path(rel).parent,
                Path("docs"),
                Path("docs/experience"),
                Path("docs/experience/wins"),
                Path("docs/experience/errors"),
                Path("."),
            )
            # os.path.normpath collapses `..`, which a plain string join does not: a
            # `../errors/x.md` from wins/ must become docs/experience/errors/x.md before it
            # can be compared against git's paths. Without this the scan reported ~40
            # false positives, all of them `../` shaped.
            cands = {os.path.normpath(str(b / raw)) for b in bases}
            if not (cands & tracked):
                line = text[: m.start()].count("\n") + 1
                out.append(f"{rel}:{line} -> {raw}")
    return out


def test_every_docs_reference_resolves():
    dead = _dead()
    assert not dead, "unresolved markdown references:\n  " + "\n  ".join(dead)


def test_the_resolver_reports_a_reference_that_does_not_exist():
    """The base list tries six directories, so it could match almost anything.

    The probe has to be `git add`ed, since existence is now asked of git: an untracked
    file is invisible to the scan, so writing it alone would make this test pass for the
    wrong reason -- the probe would be skipped rather than reported.
    """
    probe = DOCS / "_link_probe.md"
    rel = str(probe.relative_to(ROOT))
    probe.write_text("[x](2026-01-01-no-such-entry.md)\n")
    subprocess.run(["git", "-C", str(ROOT), "add", "--intent-to-add", rel], check=True)
    try:
        assert rel in _tracked(), "the probe was not staged, so the scan cannot see it"
        dead = _dead()
    finally:
        subprocess.run(["git", "-C", str(ROOT), "rm", "--cached", "-q", "--", rel], check=False)
        probe.unlink()
    assert any("_link_probe.md" in d for d in dead), (
        "a reference to a file that does not exist was not reported: the resolver's base "
        f"list matches too broadly. Got {dead}"
    )


if __name__ == "__main__":
    print(f"{len(_dead())} dead references across {len(_tracked())} tracked docs")
