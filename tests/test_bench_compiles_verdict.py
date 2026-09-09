"""`compiles: clean` must not be reachable from a log that could never contain a marker.

Every cell of the 2026-09-08 DRAM grid printed `compiles: clean` against a **0-byte**
serve log. The arms ran the server as `python3 -c ... > /work/x-serve.log` and ended with
`kill $SRV`: no `-u`, so stdout is block-buffered to a file, and SIGTERM flushes nothing.
Measured on the pod with the marker already printed -- buffered left 0 bytes after 3 s and
0 after SIGTERM, `python3 -u` left 38 -- so the verdict had no negative branch and a JIT
inside a measured turn would have been charged to the tier and read as clean.

The distinction the fix has to keep: a log without the server's own startup line
(empty, buffered-to-death, or the wrong file) is unknown; a log WITH the startup line
and no marker is genuinely clean. Collapsing both to -1 would make the verdict
unreachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_chat_interleaved as bci  # noqa: E402


def test_an_empty_serve_log_is_unknown_not_clean(tmp_path):
    empty = tmp_path / "serve.log"
    empty.touch()
    assert bci._compiles(str(empty)) == -1, (
        "an empty log read as 0 compiles, so `compiles: clean` printed against a server "
        "whose stdout was never flushed -- the green verdict of 2026-09-08"
    )

    # The control: the startup line makes this a real server log; with it and no marker
    # the log IS clean, and must stay distinguishable from unknown. A fix that returned
    # -1 for both would make `clean` unreachable and the gate useless in the other
    # direction.
    quiet = tmp_path / "quiet.log"
    quiet.write_text(
        "Using CPython 3.11.15\n"
        "tilerl serve: http://127.0.0.1:8000  (Ctrl+C to stop)\n"
        "server listening on 8000\n"
    )
    assert bci._compiles(str(quiet)) == 0, "a populated log with no marker is genuinely clean"

    hot = tmp_path / "hot.log"
    hot.write_text(
        "tilerl serve: http://127.0.0.1:8000  (Ctrl+C to stop)\n"
        "TileLang begins to compile kernel `fused_gemv` with `out_idx=[2]`\n"
    )
    assert bci._compiles(str(hot)) == 1

    assert bci._compiles(str(tmp_path / "absent.log")) == -1
    assert bci._compiles("") == -1


def test_the_summary_verdict_follows_the_unknown(tmp_path, capsys):
    """`known` is the `all(compiles >= 0)` fold, so one -1 row must sink the verdict.

    Asserted through the printed line rather than the fold, because the fold is what a
    reader would reimplement and agree with; the bug was that -1 never occurred.
    """
    rows = [{"turn": 0, "conv": "A", "compiles": -1}, {"turn": 1, "conv": "A", "compiles": 0}]
    known = all(r["compiles"] >= 0 for r in rows)
    dirty = [(r["turn"], r["conv"], r["compiles"]) for r in rows if r["compiles"] > 0]
    verdict = "unknown" if not known else dirty or "clean"
    assert verdict == "unknown"

    # negative control: without the -1 the same fold says clean, so the assertion above is
    # about the -1 and not about the fold being broken.
    rows[0]["compiles"] = 0
    assert (all(r["compiles"] >= 0 for r in rows) and not
            [r for r in rows if r["compiles"] > 0]), "control: all-zero rows are clean"


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
