import re
from collections import Counter
from pathlib import Path

import pytest

#: Entry lines sharing this many opening characters are two versions of one entry, not two
#: entries. Measured on the real file: at 60 the only collision is the known b071797 pair,
#: and the retracted-vs-corrected pair that #169's rebase resurrected also collides here.
_PREFIX = 60

_KNOWN: tuple[str, ...] = ()

_DATE_HEADER = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


def _entries(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("- **")]


def _assert_unique_lines(path):
    # Exact whole lines catch merge=union duplicates; near-duplicates are outside this check.
    counts = Counter(line for line in path.read_text().splitlines() if line.strip())
    duplicates = [line for line, count in counts.items() if count > 1]
    assert not duplicates, f"Duplicated non-blank lines in {path}: {duplicates}"


def _assert_one_version_per_entry(text: str) -> None:
    """Two entry lines with the same long opening are one entry in two versions.

    This is the union driver's actual failure mode and the exact-line check misses it: a
    rebase can bring back a superseded VERSION of a line, which differs in text. #169 was
    about to restore a retracted claim ("I destroyed it", replaced by "it was never
    written") to main this way, and the diff read +4/-0 with nothing obviously wrong.
    """
    seen: dict[str, str] = {}
    bad = []
    for ln in _entries(text):
        key = ln[:_PREFIX]
        if key.startswith(_KNOWN):
            continue
        if key in seen and seen[key] != ln:
            bad.append(key)
        seen[key] = ln
    assert not bad, (
        "CHANGELOG entries share an opening but differ later — two versions of one entry, "
        f"probably a merge=union resurrection. Reconcile them: {bad}"
    )


def _assert_one_header_per_day(text: str) -> None:
    """One `## YYYY-MM-DD` section per day, whitespace-insensitively.

    A byte-identical duplicate header is already caught by the exact-line check.
    The case it cannot see is two headers for one day that differ by whitespace,
    which is what a union merge of two same-day sections can produce: not
    duplicate *lines*, so that check passes, while the file still reads as though
    the day happened twice. Measured: `## 2026-09-20` + `## 2026-09-20 ` (trailing
    space) passes the exact-line check and trips this one.
    """
    days = _DATE_HEADER.findall(text)
    dupes = sorted(d for d, n in Counter(days).items() if n > 1)
    assert not dupes, (
        "CHANGELOG has more than one '## <date>' header for the same day (whitespace "
        f"included) — fold the entries under one: {dupes}"
    )


def test_changelog_has_no_duplicate_lines(tmp_path):
    changelog = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    _assert_unique_lines(changelog)
    text = changelog.read_text()
    copy = tmp_path / "CHANGELOG.md"
    copy.write_text(text + "\n" + next(line for line in text.splitlines() if line.strip()) + "\n")
    with pytest.raises(AssertionError, match="Duplicated non-blank lines"):
        _assert_unique_lines(copy)


def test_no_entry_appears_in_two_versions():
    # The exemption is keyed by the truncation, so a _PREFIX change silently breaks it in
    # one of two ways: longer, and the exemption never matches (the whole file goes red);
    # shorter, and it matches a prefix of itself plus unrelated entries.
    assert all(len(k) == _PREFIX for k in _KNOWN), "_KNOWN keys must be exactly _PREFIX long"
    _assert_one_version_per_entry((Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text())


def test_no_day_appears_twice():
    _assert_one_header_per_day((Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text())


def test_a_whitespace_variant_day_header_is_caught():
    """Red control for the one shape the exact-line check cannot see.

    A byte-identical second header is already caught as a duplicated line (this
    test's first half proves the sibling gate would fire). The variant differs by
    a trailing space, so it is not a duplicate *line* and only the day-count check
    catches it.
    """
    text = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    day = _DATE_HEADER.search(text).group(1)
    # The exact-line check's own guard does NOT fire on this shape.
    variant = text + f"\n## {day} \n\n- **forged: same day, trailing space**\n"
    counts = Counter(ln for ln in variant.splitlines() if ln.strip())
    assert not [ln for ln, n in counts.items() if n > 1], "control must not be a dup line"
    with pytest.raises(AssertionError, match="more than one"):
        _assert_one_header_per_day(variant)
    # And the clean text passes.
    _assert_one_header_per_day(text)


def test_a_resurrected_version_is_caught():
    """Red control: append a truncated copy of a real entry, which is what a resurrection
    looks like — same opening, different tail. The exact-line check passes it."""
    text = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    victim = next(ln for ln in _entries(text) if not ln.startswith(_KNOWN) and len(ln) > 200)
    forged = victim[:150] + " and then it said something else entirely."
    assert forged != victim
    with pytest.raises(AssertionError, match="two versions of one entry"):
        _assert_one_version_per_entry(text + forged + "\n")
