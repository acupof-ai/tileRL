#!/usr/bin/env python3
"""Send N paired wikitext-train spans to a LIVE serve and record the health bracket.

The third thing, next to the two existing probes: `probe_draft_window_sweep.py`
builds its OWN engine (which would collide with the serve under measurement), and
`probe_headroom_coldtail.py` talks to a live serve but synthesises its 32k prompt
(a uuid lead plus a word stream). Neither carries the V100 corpus convention, so
neither can answer a question whose comparison partner was measured on it.

This sends the corpus convention over the wire: `corpus.py`'s wikitext-103 split
stream, `tiled_spans(..., skip=512)` giving DISJOINT spans — disjoint on purpose,
since shared-half windows are correlated and bias the acceptance median — and one
`/v1/chat/completions` per span, bracketed by two `/health` reads.

    python3 scripts/probe_h20_train_client.py --url http://127.0.0.1:8000 \
        --ctx 32768 --n 30 --gen 64 --split train --out runs/a1_32k.json

Why the health bracket and not the tick log: acceptance is a per-arm DELTA, and
the tick log carries no accept counters. The two reads must bracket the arm
because `spec_accepted` / `spec_drafted` are cumulative and warmup moves them;
summing across arms double-counts. `probe_h20_arm_read.py health` consumes the
pair this writes.

`--skip 512` drops wikitext's header/newline head: those first rows are short
headers and blank lines that tokenize into runs of newlines, trivially
predictable, so a prompt starting there measures the model reading newlines.

One JSON, written after EVERY request, so a crash at request 29 keeps 28:
per-request prompt/completion tokens and wall time, plus the bracketing health
snapshots. Wall time is reported raw and NOT folded into a rate: prefill and
decode share the wall, and a hit in the prefix cache removes the prefill entirely
(measured on H20: 335.7 s cold vs 3.3 s on a cache hit, same prompt), so a wall
median over a batch that mixes the two is a median of two different things.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

#: The same insert the two probes use, so `corpus` resolves from a pod tree whose
#: scripts/ is on the path rather than from the caller's cwd.
sys.path.insert(0, "scripts")


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def health(base: str) -> dict:
    with urllib.request.urlopen(base + "/health", timeout=10) as r:
        return json.loads(r.read())


def body_for(text: str, gen: int) -> dict:
    return {
        "model": "qwen38-27b",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": gen,
        "temperature": 0.0,
    }


def summarise(h0: dict, h1: dict, rows: list[dict]) -> dict:
    """The arm's bracket deltas plus the wall split that keeps prefill out of it.

    A wall median over every request mixes two populations -- a fresh 32k prefill
    and a prefix-cache hit -- which differ by two orders of magnitude on this
    line. They are split by measured finish time, not by a flag, because the
    client cannot see which requests hit.
    """

    def g(d: dict, k: str) -> int:
        return (d.get("stats") or d).get(k, 0)

    ok = [r for r in rows if "error" not in r]
    walls = sorted(r["wall_s"] for r in ok)
    #: Below this a 32k prefill cannot have run (measured: ~50 s), so the request
    #: was served from a resident prefix. A split point, not a claim about speed.
    HIT_S = 10.0
    return {
        "n_ok": len(ok),
        "n_total": len(rows),
        "delta": {
            "accepted": g(h1, "spec_accepted") - g(h0, "spec_accepted"),
            "drafted": g(h1, "spec_drafted") - g(h0, "spec_drafted"),
            "generated": g(h1, "tokens_generated") - g(h0, "tokens_generated"),
            "decode_forwards": g(h1, "decode_forwards") - g(h0, "decode_forwards"),
            "prefix_hits": g(h1, "prefix_hits") - g(h0, "prefix_hits"),
        },
        "wall_hit_s": [w for w in walls if w < HIT_S],
        "wall_prefill_s": [w for w in walls if w >= HIT_S],
        "prompt_tokens": sorted({r["prompt_tokens"] for r in ok}),
    }


def self_check() -> int:
    """Hermetic: no torch, no card, no network. The bracket arithmetic and the
    wall split, on a hand-built pair."""
    h0 = {
        "stats": {
            "spec_accepted": 13,
            "spec_drafted": 15,
            "tokens_generated": 48,
            "decode_forwards": 21,
            "prefix_hits": 0,
        }
    }
    h1 = {
        "stats": {
            "spec_accepted": 925,
            "spec_drafted": 1007,
            "tokens_generated": 1968,
            "decode_forwards": 1013,
            "prefix_hits": 2,
        }
    }
    rows = [
        {"i": 0, "prompt_tokens": 32778, "completion_tokens": 64, "wall_s": 335.703},
        {"i": 1, "prompt_tokens": 32778, "completion_tokens": 64, "wall_s": 3.309},
        {"i": 2, "prompt_tokens": 32778, "completion_tokens": 64, "wall_s": 53.29},
        {"i": 3, "error": "URLError: timed out"},
    ]
    s = summarise(h0, h1, rows)
    assert s["n_ok"] == 3 and s["n_total"] == 4, s
    assert s["delta"]["accepted"] == 912, s
    assert s["delta"]["drafted"] == 992, s
    assert s["delta"]["decode_forwards"] == 1013 - 21, s
    # the two wall populations are separated, and the slow one is not in the fast
    assert s["wall_hit_s"] == [3.309], s["wall_hit_s"]
    assert s["wall_prefill_s"] == [53.29, 335.703], s["wall_prefill_s"]
    # the request body is the shape the serve accepts, with the raw span as content
    b = body_for("hello", 8)
    assert b["messages"] == [{"role": "user", "content": "hello"}], b
    assert b["max_tokens"] == 8 and b["temperature"] == 0.0, b
    print("h20_train_client self-check ok")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    # Bare invocation runs the self-check: the repo-wide test_main_selfchecks gate
    # runs every hermetic script with NO args and requires rc 0.
    if not argv or argv[0] == "selfcheck":
        return self_check()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--split", default="train")
    ap.add_argument("--skip", type=int, default=512)
    ap.add_argument("--source", default="/data00/Qwen3.8-27B-NVFP4")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from tilerl.tokenizer import get_tokenizer

    sys.path.insert(0, "scripts")
    from corpus import tiled_spans, wikitext_ids_stream  # noqa: E402

    tok = get_tokenizer(a.source)
    # Bounded tokenization: the train split is ~540M chars and one full encode
    # materializes ~140M Python ints (observed SIGKILL rc137 on a 31 GiB host).
    # A sweep needs only skip + n*ctx.
    stream = wikitext_ids_stream(tok, a.split, "", a.skip + a.n * a.ctx + 4096)
    spans, n_eff = tiled_spans(stream, a.n, a.ctx, skip=a.skip)
    print(f"# split={a.split} n_eff={n_eff}/{a.n} corpus_tokens={len(stream)}", flush=True)
    if n_eff < a.n:
        print(
            f"# WARN corpus-limited: {n_eff} disjoint {a.ctx}-token spans, not {a.n}",
            file=sys.stderr,
            flush=True,
        )

    h0 = health(a.url)
    rows: list[dict] = []
    for i, ids in enumerate(spans):
        t0 = time.perf_counter()
        try:
            out = post(a.url + "/v1/chat/completions", body_for(tok.decode(ids), a.gen), a.timeout)
            u = out.get("usage", {})
            rows.append(
                {
                    "i": i,
                    "prompt_tokens": u.get("prompt_tokens", 0),
                    "completion_tokens": u.get("completion_tokens", 0),
                    "finish": out["choices"][0].get("finish_reason"),
                    "wall_s": round(time.perf_counter() - t0, 3),
                }
            )
            r = rows[-1]
            print(
                f"# req {i}: prompt={r['prompt_tokens']} gen={r['completion_tokens']} "
                f"wall={r['wall_s']}s",
                flush=True,
            )
        except Exception as exc:  # one bad request must not lose the earlier ones
            rows.append({"i": i, "error": f"{type(exc).__name__}: {exc}"})
            print(f"# req {i}: FAILED {type(exc).__name__}: {exc}", flush=True)
        # Written after every request: a crash at the last prompt still leaves the
        # health_before that makes the partial batch readable.
        with open(a.out, "w") as fh:
            json.dump(
                {"ctx": a.ctx, "split": a.split, "n_eff": n_eff, "health_before": h0, "rows": rows},
                fh,
                indent=1,
            )

    h1 = health(a.url)
    rec = {
        "ctx": a.ctx,
        "split": a.split,
        "n_eff": n_eff,
        "health_before": h0,
        "health_after": h1,
        "rows": rows,
    }
    rec["summary"] = summarise(h0, h1, rows)
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1)
    s = rec["summary"]
    print(
        f"# DONE ok={s['n_ok']}/{s['n_total']} hits={len(s['wall_hit_s'])} "
        f"prefill={len(s['wall_prefill_s'])} out={a.out}",
        flush=True,
    )
    return 0 if s["n_ok"] else 1


if __name__ == "__main__":  # runnable check
    assert main() == 0, "self-check failed"
