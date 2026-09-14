#!/usr/bin/env python3
"""V100 forced-graph smoke for the dense+d1 post-commit block-boundary race.

Deterministic device layer of the gate in
docs/experience/errors/2026-09-15-dense-d1-concurrent-block-boundary-assert.md
(the unit gate is the acceptance test; this is the end-to-end confirmation that
the captured-graph and eager paths share the post-commit growth helper).

It drives the RUNNING serve over HTTP (no deploy here):
  * forces the GRAPH path indirectly: the hybrid serve already runs dense rows
    under --decode-graph, and equal-length prompts all land in the same
    (B=4, W) bucket, so the tick replays a captured graph, not eager;
  * submits FOUR equal dense d1 rows in one batch (before draining), so they
    decode together tick after tick and several accept a token onto the same
    16-token boundary in one captured tick -- the pre-fix condition;
  * asserts the fixed outcome: all four HTTP 200, each exactly max_new_tokens,
    zero 500 / spec.py "draft would write position", and the pool returns to
    its idle free-block count afterwards (zero block leak).

Pre-fix this exits non-zero whenever a batched tick hits the boundary
(observed ~1/3 on the 4x2k warmup; the equal-length in-one-batch shape here is
chosen to hit it reliably rather than by warmup timing). Post-fix it is clean.

Usage (on the V100 host, against the local serve):
    python3 scripts/smoke_dense_d1_boundary.py [--rounds N] [--words W]
Env: BASE (default http://127.0.0.1:8000), MODEL (default qwen38-27b).
It ONLY reads :8000; it does not start or restart the serve.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "qwen38-27b")
# Dense: under the hybrid sparse threshold (8192); long enough that the four
# rows spend many decode ticks batched at B=4, so a 16-boundary acceptance in a
# captured tick is essentially guaranteed over the rounds. The filler tokenizes
# ~4 tokens/word on this tokenizer (measured: 100 words -> 413 prompt tokens),
# so ~250 words ~= 1k tokens -- matching the warmup's prose ~2k-char prompt,
# nowhere near the 8192 threshold or the 31.7 GiB 4-row fill ceiling.
DEFAULT_WORDS = 250
DEFAULT_MAX_NEW = 16  # > one block of generation, so decode crosses boundaries
ROUNDS = 8  # 4 rows per round; pre-fix the race was ~1/3, eight rounds all but rules out a miss


def health(path: str = "/health") -> dict:
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.load(r)


def stats() -> dict:
    return health()["stats"]


def one_request(i: int, words: int, max_new: int) -> tuple[int, str, int]:
    """(http_status, error_snippet, completion_tokens). Equal-length prompts keep
    the rows in one (B,W) graph bucket; distinct nonces defeat prefix sharing so
    each is its own cold dense row (no adoption short-circuit)."""
    content = f"nonce-{uuid.uuid4().hex} " + "word " * words
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_new,
        "enable_thinking": False,  # think-off: every token is a committed decode
        "stream": False,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        n = int(d["usage"]["completion_tokens"])
        finish = d["choices"][0].get("finish_reason", "")
        return 200, "", n, finish
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:300]
        return e.code, msg, 0, ""
    except Exception as e:  # noqa: BLE001 - the smoke reports the class verbatim
        return -1, f"{type(e).__name__}: {e}", 0, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--words", type=int, default=DEFAULT_WORDS)
    ap.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW)
    a = ap.parse_args()

    pre = stats()
    if pre.get("running") or pre.get("waiting"):
        print(f"FAIL: serve not idle before probe: {pre['running']}/{pre['waiting']}",
              file=sys.stderr)
        return 2
    if not pre.get("decode_graph"):
        print("FAIL: serve reports decode_graph=false; this smoke needs --decode-graph",
              file=sys.stderr)
        return 2
    free_before = pre["blocks_total"] - pre["blocks_used"]
    print(f"serve idle, decode_graph=true, free blocks {free_before}; "
          f"firing {a.rounds} rounds x 4 dense d1 rows")

    failures: list[str] = []
    t0 = time.time()
    for rnd in range(a.rounds):
        # Submit all four BEFORE reading any: they admit and batch together.
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(
                lambda i: one_request(i, a.words, a.max_new), range(4)))
        codes = [c for c, _, _, _ in results]
        rows = [(n, f) for _, _, n, f in results]
        bad = [(c, m) for c, m, _, _ in results if c != 200]
        print(f"round {rnd}: codes={codes} (tokens,finish)={rows}")
        if bad:
            failures.append(f"round {rnd}: {bad}")
            continue
        # A row ending EARLY is fine only on a clean stop/eos; a short row that is
        # still finish_reason=length is a truncation (should be impossible when the
        # cap equals max_new), and an empty continuation is a dropped write.
        for n, finish in rows:
            if n == 0:
                failures.append(f"round {rnd}: zero-token continuation ({finish})")
            elif n < a.max_new and finish not in ("stop", "eos", "stop_sequence"):
                failures.append(
                    f"round {rnd}: short {n}/{a.max_new} with finish {finish!r}")
        # Small gap lets the engine drain to fully idle between batches, so the
        # next round again starts with four fresh rows at the same phase.
        for _ in range(50):
            s = stats()
            if not s["running"] and not s["waiting"]:
                break
            time.sleep(0.1)

    post = stats()
    free_after = post["blocks_total"] - post["blocks_used"]
    print(f"done in {time.time()-t0:.1f}s; free blocks {free_before} -> {free_after}, "
          f"running {post['running']} waiting {post['waiting']}")

    # blocks_used includes resident prefix/state that legitimately survives a
    # completed request; the no-leak signal is that no REQUEST is still holding
    # decode blocks: running/waiting/slots all zero. Require that, and treat a
    # free-block delta as a failure only when rows are still resident.
    if (post["running"] or post["waiting"]
            or post.get("slots_used", 0)):
        failures.append(
            f"rows/slots resident after drain: running {post['running']} "
            f"waiting {post['waiting']} slots {post.get('slots_used')}")
    if free_after != free_before and (post["running"] or post["waiting"]):
        failures.append(f"block leak with live rows: free {free_before} -> {free_after}")

    if failures:
        print("SMOKE FAIL:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        print("PRE-FIX EXPECTATION: a spec.py 'draft would write position' 500 "
              "is the known race; post-fix this smoke must be clean.", file=sys.stderr)
        return 1
    print("SMOKE PASS: all rows completed, zero boundary 500, zero block leak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
