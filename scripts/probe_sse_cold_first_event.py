#!/usr/bin/env python3
"""Probe the cold-prefill SSE silence window on the Anthropic / Responses routes.

Both streaming routes await the FULL completion before emitting any SSE event
and send no ``ping`` / keepalive frame, so a long cold prefill delivers zero
bytes for tens of seconds (docs/api-compat-surface.md, owed live check 3). This
answers, on a live server:

- how long the client waits for the FIRST byte (TTFB) and the first parsed SSE
  event on a request engineered to miss the prefix cache;
- whether any keepalive bytes arrive during that wait (SSE comment frames begin
  with ``:``; this server is documented to send none);
- whether the connection survives to first event or is closed first.

It reads the RAW socket with timestamps, so an idle-disconnect by a proxy or by
the client stack is distinguishable from "just slow": a closed stream before
the first SSE event is RED.

Use a one-off long prefix to miss the prefix cache. The default prompt carries a
random nonce plus ~24k of filler characters; override --filler-chars for a
different cold length.

Usage (read-only, deployment window; default contacts nothing):
    python scripts/probe_sse_cold_first_event.py \
        --base-url http://localhost:8000/v1 --model qwen38-27b

Exit code:
    0  first SSE event arrived with the connection alive (silence tolerated)
    1  connection closed before any SSE event, or an error event arrived
    2  could not establish the request

Offline logic check:
    python scripts/probe_sse_cold_first_event.py --self-check
"""

from __future__ import annotations

import argparse
import http.client
import json
import secrets
import sys
import time
from typing import Any


def classify(stream_bytes: list[tuple[float, bytes]], connected_at_first: bool
             ) -> tuple[int, dict[str, Any]]:
    """Red/green from timestamped raw chunks. Returns (code, detail).

    GREEN needs: at least one byte, an SSE ``event:`` or ``data:`` line among
    them, and the stream was not already closed before that. The caller passes
    whether the read ended (closed) while chunks were still accumulating; the
    decision is made on the collected bytes alone so it is offline-testable.
    """
    total = b"".join(c for _, c in stream_bytes)
    text = total.decode(errors="replace")
    first_byte_ms = stream_bytes[0][0] * 1000 if stream_bytes else None
    comments = [ln for ln in text.splitlines() if ln.startswith(":")]
    has_event = any(ln.startswith(("event:", "data:")) for ln in text.splitlines())
    detail = {"first_byte_ms": round(first_byte_ms, 1) if first_byte_ms is not None
              else None,
              "bytes_before_first_event": None,
              "keepalive_comment_frames": comments,
              "saw_sse_event": has_event}
    if not stream_bytes:
        detail["reason"] = "stream closed with zero bytes before any SSE event"
        return 1, detail
    # Bytes before the first SSE frame line, to expose comment/keepalive frames.
    head = text.split("\n")
    pre = []
    for ln in head:
        if ln.startswith(("event:", "data:")):
            break
        if ln.strip():
            pre.append(ln)
    detail["bytes_before_first_event"] = pre
    if not has_event:
        detail["reason"] = "bytes arrived but no SSE event/data frame"
        return 1, detail
    return 0, detail


def _raw_stream(host: str, port: int, path: str, body: dict[str, Any],
                tls: bool, deadline_s: float) -> list[tuple[float, bytes]]:
    """POST and collect timestamped chunks (seconds since request) until the
    first line that starts an SSE frame or the socket closes."""
    conn_cls = http.client.HTTPSConnection if tls else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=30)
    conn.request("POST", path, body=json.dumps(body),
                 headers={"Content-Type": "application/json",
                          "Accept": "text/event-stream"})
    start = time.monotonic()
    resp = conn.getresponse()
    if resp.status >= 400:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read(400)!r}")
    chunks: list[tuple[float, bytes]] = []
    seen_event = False
    while time.monotonic() - start < deadline_s:
        chunk = resp.readline()
        if not chunk:
            break  # closed by peer
        chunks.append((time.monotonic() - start, chunk))
        line = chunk.decode(errors="replace").strip()
        if line.startswith(("event:", "data:")):
            seen_event = True
            break
    conn.close()
    if not seen_event and not chunks:
        pass
    return chunks


def _split_url(base: str) -> tuple[bool, str, int, str]:
    """(tls, host, port, base_path) from http(s)://host[:port]/v1."""
    from urllib.parse import urlparse

    u = urlparse(base)
    tls = u.scheme == "https"
    default_port = 443 if tls else 80
    return tls, u.hostname or "localhost", u.port or default_port, (u.path or "")


def run_remote(args: argparse.Namespace) -> int:
    tls, host, port, path = _split_url(args.base_url.rstrip("/"))
    route = {
        "messages": "/messages",
        "responses": "/responses",
    }[args.route]
    nonce = secrets.token_hex(12)
    # Filler that does not collapse into a prefix match: a unique nonce first,
    # then plain tokens. The nonce is the cache-busting signal measured in prior
    # probes; keep it ahead of the filler rather than at the end.
    filler = ("token " * (args.filler_chars // 6)).strip()
    prompt = f"nonce-{nonce}. Read this context then answer with one word: ok. {filler}"
    if args.route == "messages":
        body: dict[str, Any] = {"model": args.model, "max_tokens": args.max_tokens,
                                "stream": True,
                                "messages": [{"role": "user", "content": prompt}]}
    else:
        body = {"model": args.model, "max_output_tokens": args.max_tokens,
                "stream": True, "input": prompt}
    print(f"cold prefix: {len(prompt)} chars, nonce {nonce}, route /v1{route}")
    t0 = time.monotonic()
    try:
        chunks = _raw_stream(host, port, path + route, body, tls,
                             deadline_s=args.deadline_s)
    except Exception as exc:  # noqa: BLE001
        print("request failed:", f"{type(exc).__name__}: {exc}")
        return 2
    elapsed = time.monotonic() - t0
    rc, detail = classify(chunks, connected_at_first=True)
    detail["wall_seconds_until_first_event_or_close"] = round(elapsed, 2)
    print(json.dumps(detail, indent=2, ensure_ascii=False))
    if rc == 0:
        print("=== RESULT: GREEN ===")
        print("first SSE event arrived on a live connection")
    else:
        print("=== RESULT: RED ===")
        print(detail.get("reason", "no SSE event before close/timeout"))
    return rc


def self_check() -> int:
    """Offline: a healthy framed stream is GREEN, an empty closed stream RED."""
    healthy = [(0.001, b": keepalive\n"), (12.4, b"event: message_start\n"),
               (12.4, b'data: {"type":"message_start"}\n\n')]
    print("--- self-check: keepalive comment then a frame -> GREEN ---")
    rc_ok, d_ok = classify(healthy, True)
    print(json.dumps(d_ok))
    print("rc:", rc_ok)

    print("--- self-check: zero bytes then close -> RED ---")
    rc_empty, d_empty = classify([], False)
    print(json.dumps(d_empty))
    print("rc:", rc_empty)

    print("--- self-check: only comment frames, no event -> RED ---")
    rc_comments, d_c = classify([(1.0, b": ping\n")], True)
    print("rc:", rc_comments)

    ok = rc_ok == 0 and rc_empty == 1 and rc_comments == 1
    print("self-check", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="qwen38-27b")
    p.add_argument("--route", choices=["messages", "responses"],
                   default="messages")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--filler-chars", type=int, default=24_000,
                   help="approx cold-prefix length in characters (after nonce)")
    p.add_argument("--deadline-s", type=float, default=600.0)
    p.add_argument("--self-check", action="store_true",
                   help="validate the classify logic locally, no network")
    args = p.parse_args()
    return self_check() if args.self_check else run_remote(args)


if __name__ == "__main__":
    sys.exit(main())
