"""board.py alt-schema normalization (migration shim for 13 hand-written rows)."""

import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "board", os.path.join(os.path.dirname(__file__), "..", "scripts", "board.py")
)
board = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(board)


def test_norm_alt_schema():
    alt = {
        "ts": 1788846889,
        "from": "27",
        "kind": "finding",
        "title": "P1's four prerequisites are all closed",
        "body": "roadmap.md P1 lists four CPU-gated prerequisites.",
    }
    r = board._norm(alt)
    assert r["who"] == "27"
    assert r["topic"] == "coord"
    assert r["kind"] == "finding"
    assert r["text"] == alt["body"]
    assert r["ts"] == "2026-09-08 05:54Z"  # epoch 1788846889
    assert "artifact" in r


def test_norm_standard_schema_passthrough():
    std = {
        "ts": "2026-09-10 00:00Z",
        "who": "v100",
        "topic": "pod",
        "kind": "note",
        "text": "hello",
        "artifact": "",
    }
    assert board._norm(std) is std


def test_norm_alt_no_body_falls_back_to_title():
    alt = {"ts": 1788846889, "from": "52", "kind": "finding", "title": "just a title"}
    r = board._norm(alt)
    assert r["text"] == "just a title"
