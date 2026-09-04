"""Does support-matrix.md still match the registry it says it mirrors?

Every number in that file's cell table was wrong on 2026-09-05 -- `_REGISTRY` 8 keys
against a real 10, cpu/metal 13 against 15, sm90 "23 (5 overridden, 10 added)" against
41 (9, 26), and no sm70 column at all while sm70 was the served arch. The file says
"registry.py is the source of truth and this file mirrors it", and nothing checked.

This asserts the counts, not the prose: a maker added to a cell fails it, and the fix is
to update the table. Keeping the arithmetic here rather than in the doc is what makes the
doc checkable.
"""

import re
from pathlib import Path

import pytest
from tilerl_kernels import registry

DOC = Path(__file__).parent.parent / "docs" / "support-matrix.md"


def _cell(name: str) -> dict:
    return getattr(registry, f"_{name.upper()}_KERNELS")


def _split(name: str) -> tuple[int, int, int]:
    """(overrides cpu, added, same maker as cpu) for one accelerated cell."""
    cell, cpu = _cell(name), _cell("cpu")
    shared = set(cell) & set(cpu)
    return (
        sum(1 for k in shared if cell[k] is not cpu[k]),
        len(set(cell) - set(cpu)),
        sum(1 for k in shared if cell[k] is cpu[k]),
    )


def _row(cell: str) -> tuple[int, ...]:
    """The doc's `| cell | entries | overrides | added | same |` row, as ints."""
    m = re.search(rf"^\|\s*{cell}\s*\|([^\n]*)$", DOC.read_text(), re.M)
    assert m, f"no table row for {cell!r} in {DOC.name}"
    return tuple(
        int(n)
        for cellstr in m.group(1).split("|")
        for n in re.findall(r"\b(\d+)\b", cellstr.replace(",", ""))[:1]
    )


def test_the_doc_counts_the_registry_keys_it_has():
    text = DOC.read_text()
    keys, sets = len(registry._REGISTRY), len({id(v) for v in registry._REGISTRY.values() if v})
    assert f"holds {keys} keys" in text, (
        f"_REGISTRY has {keys} keys; the doc does not say so. It claimed 8 while the real "
        f"number was 10, and the whole cell table was derived from that."
    )
    assert f"{sets} distinct kernel sets" in text, f"there are {sets} distinct non-empty sets"


@pytest.mark.parametrize("cell", ["metal", "sm90", "sm70"])
def test_the_doc_cell_table_matches_the_registry(cell):
    over, added, same = _split(cell)
    want = (len(_cell(cell)), over, added, same)
    assert _row(cell) == want, (
        f"{cell} is {want} (entries, overrides, added, same-as-cpu) but the doc's row reads "
        f"{_row(cell)}. A name two cells share is only an override when the maker differs."
    )


def test_sm70_is_listed_as_an_executed_target():
    """It is the served arch (27B NVFP4 on a V100) and was missing from every table."""
    text = DOC.read_text()
    assert re.search(r"Four targets have executed", text), "sm70 makes it four, not three"
    for section in ("## bf16", "## fp4"):
        table = text.split(section, 1)[1].split("\n\n", 2)[1]
        assert "sm70" in table, f"{section} has no sm70 column"
