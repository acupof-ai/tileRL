#!/usr/bin/env python3
"""Same-batch eval floor: gross flips and net between two same-weights eval files.

The before/after arms score the eval file's first N rows in order (no shuffle),
so row index is identity across arms and across reruns. Pairs by index and
asserts the gold answers agree, so a wrong-file or wrong-order pairing goes red
instead of producing a plausible flip count.

Usage: floor_diff.py <a.jsonl> <b.jsonl>
       floor_diff.py --self-check
"""

import json
import sys


def load_gsm8k(path: str) -> list[dict]:
    return [r for r in (json.loads(l) for l in open(path)) if r.get("dataset", "gsm8k") == "gsm8k"]


def diff(a: str, b: str) -> None:
    ra, rb = load_gsm8k(a), load_gsm8k(b)
    assert len(ra) == len(rb), f"row count differs: {len(ra)} vs {len(rb)}"
    bad = [i for i, (x, y) in enumerate(zip(ra, rb)) if x["answer"] != y["answer"]]
    assert not bad, f"{len(bad)} rows pair on different golds (first: {bad[:3]})"
    r2w = [i for i, (x, y) in enumerate(zip(ra, rb)) if x["correct"] and not y["correct"]]
    w2r = [i for i, (x, y) in enumerate(zip(ra, rb)) if not x["correct"] and y["correct"]]
    sa, sb = sum(r["correct"] for r in ra), sum(r["correct"] for r in rb)
    print(f"{a} vs {b}")
    print(f"  scores {sa} vs {sb} (net {sb - sa:+d})")
    print(f"  gross flips {len(r2w) + len(w2r)}: right->wrong {len(r2w)}, wrong->right {len(w2r)}")


def _self_check() -> None:
    import tempfile
    from pathlib import Path
    rows = lambda bits: [{"answer": "42", "correct": b, "tokens": 10} for b in bits]
    write = lambda p, bits: Path(p).write_text("\n".join(json.dumps(r) for r in rows(bits)))
    with tempfile.TemporaryDirectory() as d:
        a, b = f"{d}/a.jsonl", f"{d}/b.jsonl"
        write(a, [1, 1, 0, 0, 1])
        write(b, [1, 0, 0, 1, 1])  # flips at 1 (r2w) and 3 (w2r), net 0
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            diff(a, b)
        out = buf.getvalue()
        assert "gross flips 2: right->wrong 1, wrong->right 1" in out, out
        assert "net +0" in out, out
    print("self-check ok")


if __name__ == "__main__":
    if sys.argv[1] == "--self-check":
        _self_check()
    else:
        diff(sys.argv[1], sys.argv[2])
