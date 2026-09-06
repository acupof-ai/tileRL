"""Read pieces of `scripts/pod_sync.sh` so the tests exercise the shipped text.

Three test files each pull a line or block out of that script; copying the expressions
instead would let the script drift while every test stayed green. One extractor here, three
callers, so a rename breaks in one place.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "pod_sync.sh"


def line(*needles: str) -> str:
    """The one line containing every needle. Raises if zero or several match, because a
    silently-picked first match is how a test ends up asserting on the wrong line."""
    hits = [ln for ln in _SCRIPT.read_text().splitlines() if all(n in ln for n in needles)]
    assert len(hits) == 1, f"{len(hits)} lines in pod_sync.sh match {needles}: {hits[:3]}"
    return hits[0]


def block(start_prefix: str, end: str) -> str:
    """From the line starting with `start_prefix` through the first line equal to `end`."""
    lines = _SCRIPT.read_text().splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith(start_prefix))
    j = next(n for n in range(i, len(lines)) if lines[n] == end)
    return "\n".join(lines[i : j + 1])


def wipe_expr() -> str:
    """The find expression, unescaped: the script embeds it for a nested `bash -lc`, so
    `\\\\!` on disk arrives as `\\!` here."""
    return re.sub(r"\\+!", "!", line("wipe=")[len("wipe=") :].strip('"'))
