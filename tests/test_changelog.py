from collections import Counter
from pathlib import Path

import pytest

#: Entry lines sharing this many opening characters are two versions of one entry, not two
#: entries. Measured on the real file: at 60 the only collision is the known b071797 pair,
#: and the retracted-vs-corrected pair that #169's rebase resurrected also collides here.
_PREFIX = 60

_KNOWN: tuple[str, ...] = ()


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


def test_a_resurrected_version_is_caught():
    """Red control: append a truncated copy of a real entry, which is what a resurrection
    looks like — same opening, different tail. The exact-line check passes it."""
    text = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    victim = next(ln for ln in _entries(text) if not ln.startswith(_KNOWN) and len(ln) > 200)
    forged = victim[:150] + " and then it said something else entirely."
    assert forged != victim
    with pytest.raises(AssertionError, match="two versions of one entry"):
        _assert_one_version_per_entry(text + forged + "\n")
