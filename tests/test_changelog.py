from collections import Counter
from pathlib import Path

import pytest


def _assert_unique_lines(path):
    counts = Counter(line for line in path.read_text().splitlines() if line.strip())
    duplicates = [line for line, count in counts.items() if count > 1]
    assert not duplicates, f"Duplicated non-blank lines in {path}: {duplicates}"


def test_changelog_has_no_duplicate_lines(tmp_path):
    changelog = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    _assert_unique_lines(changelog)
    text = changelog.read_text()
    copy = tmp_path / "CHANGELOG.md"
    copy.write_text(text + "\n" + next(line for line in text.splitlines() if line.strip()) + "\n")
    with pytest.raises(AssertionError, match="Duplicated non-blank lines"):
        _assert_unique_lines(copy)
