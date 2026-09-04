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
"""

import re
from pathlib import Path

DOCS = Path(__file__).parent.parent / "docs"
ROOT = DOCS.parent
EXP = DOCS / "experience"

#: `[text](path.md)`, then the two bare shapes entries use inside backticks.
_LINK = re.compile(r"\[[^\]]*\]\(([^)#]+\.md)[^)]*\)")
_TICK = re.compile(r"`((?:docs/|\.\./)?(?:experience/)?(?:wins|errors)/[\w./-]+\.md)`")
_BARE = re.compile(r"`(20\d\d-\d\d-\d\d-[\w.-]+\.md)`")


def _dead() -> list[str]:
    out = []
    for md in sorted(DOCS.rglob("*.md")):
        text = md.read_text(errors="ignore")
        for m in (*_LINK.finditer(text), *_TICK.finditer(text), *_BARE.finditer(text)):
            raw = m.group(1)
            bases = (md.parent, ROOT, DOCS, EXP, EXP / "wins", EXP / "errors")
            if not any((b / raw).exists() for b in bases):
                line = text[: m.start()].count("\n") + 1
                out.append(f"{md.relative_to(ROOT)}:{line} -> {raw}")
    return out


def test_every_docs_reference_resolves():
    dead = _dead()
    assert not dead, "unresolved markdown references:\n  " + "\n  ".join(dead)


def test_the_resolver_reports_a_reference_that_does_not_exist():
    """The base list tries six directories, so it could match almost anything. This
    writes one genuinely dead link into the tree and asserts it comes back."""
    probe = DOCS / "_link_probe.md"
    probe.write_text("[x](2026-01-01-no-such-entry.md)\n")
    try:
        dead = _dead()
    finally:
        probe.unlink()
    assert any("_link_probe.md" in d for d in dead), (
        "a reference to a file that does not exist was not reported: the resolver's base "
        f"list matches too broadly. Got {dead}"
    )


if __name__ == "__main__":
    print(f"{len(_dead())} dead references across {len(list(DOCS.rglob('*.md')))} files")
