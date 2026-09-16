#!/usr/bin/env python3
"""Probe token-accounting consistency across the Anthropic and OpenAI routes.

Two questions from docs/api-compat-surface.md "live verification still owed":

1. Does /v1/messages streaming double-count? The server puts the FULL usage on
   both ``message_start`` (``message.usage``) and the terminal
   ``message_delta`` (``usage``). Anthropic's contract is input tokens on start
   and output tokens on the delta; an aggregator that SUMS usage across events
   counts the repeated class twice. The probe extracts usage from every
   usage-bearing event, prints the naive per-event sum against the terminal
   value, and flags the structural duplicate.
2. Do /v1/messages and /v1/chat/completions report the same counts for the same
   prompt? Same engine, same tokenizer; input/output should match exactly.

Raw HTTP only (urllib) so it runs with no SDK installed. It observes and
reports; it does not run a real Claude Code client (that remains a manual
check).

Usage (read-only, a deployment window; default contacts nothing):
    python scripts/probe_messages_usage_accounting.py \
        --base-url http://localhost:8000/v1 --model qwen38-27b

Exit code:
    0  stream usage is not duplicated AND the two routes agree
    1  one expectation failed (printable RED evidence)
    2  could not talk to the server

Offline logic check:
    python scripts/probe_messages_usage_accounting.py --self-check
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import urllib.request
from typing import Any


def _post_json(url: str, body: dict[str, Any], api_key: str, stream: bool) -> Any:
    """POST JSON. Non-stream returns the parsed dict; stream returns the raw
    response object for the caller to iterate line by line."""
    req = urllib.request.Request(
        url.rstrip("/"),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}",
                 "Accept": "text/event-stream" if stream else "application/json"},
    )
    return urllib.request.urlopen(req, timeout=600)  # noqa: S310 - configured url


def _sse_events(resp: Any) -> list[tuple[str, dict[str, Any]]]:
    """Parse (event_name, data_json) pairs from one SSE response."""
    events: list[tuple[str, dict[str, Any]]] = []
    name = ""
    for raw in resp:
        line = raw.decode(errors="replace").rstrip("\n").rstrip("\r")
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            with contextlib.suppress(json.JSONDecodeError):
                events.append((name, json.loads(line[5:].strip())))
            name = ""
    return events


def stream_usage(url: str, body: dict[str, Any], api_key: str) -> dict[str, Any]:
    """Every usage object carried by a /v1/messages SSE stream, keyed by event."""
    with _post_json(url, {**body, "stream": True}, api_key, stream=True) as resp:
        events = _sse_events(resp)
    carried: dict[str, dict[str, int]] = {}
    for name, data in events:
        usage = None
        if name == "message_start":
            usage = data.get("message", {}).get("usage")
        elif name == "message_delta":
            usage = data.get("usage")
        if usage:
            carried[name] = {k: int(v) for k, v in usage.items() if isinstance(v, int)}
    return carried


def evaluate(nonstream: dict[str, int], chat: dict[str, int],
             stream_by_event: dict[str, dict[str, int]]) -> int:
    """Apply the red/green criteria. Returns code."""
    print("=== /v1/messages non-stream usage ===")
    print(json.dumps(nonstream, sort_keys=True))
    print("=== /v1/chat/completions usage ===")
    print(json.dumps(chat, sort_keys=True))
    print("=== /v1/messages stream usage by event ===")
    print(json.dumps(stream_by_event, sort_keys=True))

    red: list[str] = []

    inp = nonstream.get("input_tokens")
    out = nonstream.get("output_tokens")
    if inp is None or out is None:
        red.append("messages usage missing input_tokens/output_tokens")
    if chat.get("prompt_tokens") != inp:
        red.append(f"input count disagrees across routes: messages.input_tokens="
                   f"{inp} vs chat.prompt_tokens={chat.get('prompt_tokens')}")
    if chat.get("completion_tokens") != out:
        red.append(f"output count disagrees across routes: messages.output_tokens="
                   f"{out} vs chat.completion_tokens={chat.get('completion_tokens')}")

    # The structural double-count: a token class reported with a nonzero value on
    # two or more events. Anthropic's contract reports input once (start) and
    # output once (delta); the server currently repeats the full usage.
    summed = {"input_tokens": 0, "output_tokens": 0}
    repeated: list[str] = []
    for key in summed:
        seen = [name for name, u in stream_by_event.items() if u.get(key, 0) > 0]
        for u in stream_by_event.values():
            summed[key] += u.get(key, 0)
        if len(seen) > 1:
            repeated.append(f"{key} nonzero on {seen}: a sum-across-events "
                            f"aggregator counts {summed[key]} vs terminal "
                            f"{nonstream.get(key)}")
    if repeated:
        red.append("stream usage repeated across events (double-count risk): "
                   + "; ".join(repeated))

    for name, u in stream_by_event.items():
        for field in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            if field in u and not isinstance(u[field], int):
                red.append(f"{name}.{field} is not an integer: {u[field]!r}")

    if red:
        print("=== RESULT: RED ===")
        for r in red:
            print(" -", r)
        return 1
    print("=== RESULT: GREEN ===")
    print("routes agree and stream usage is not repeated across events")
    return 0


def run_remote(args: argparse.Namespace) -> int:
    base = args.base_url.rstrip("/")
    body = {"model": args.model, "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": args.prompt}]}
    try:
        with _post_json(f"{base}/messages", body, args.api_key, stream=False) as r:
            messages_raw = json.loads(r.read().decode())
        with _post_json(f"{base}/chat/completions", body, args.api_key,
                        stream=False) as r:
            chat_raw = json.loads(r.read().decode())
        stream_by_event = stream_usage(f"{base}/messages", body, args.api_key)
    except Exception as exc:  # noqa: BLE001 - report any transport failure
        print("request failed:", f"{type(exc).__name__}: {exc}")
        return 2

    nonstream = {k: v for k, v in (messages_raw.get("usage") or {}).items()
                 if isinstance(v, int)}
    chat = {k: v for k, v in (chat_raw.get("usage") or {}).items()
            if isinstance(v, int)}
    return evaluate(nonstream, chat, stream_by_event)


def self_check() -> int:
    """Exercise evaluate() offline with one contract-shaped and one
    duplicated-usage stream, plus a disagreeing cross-route fixture."""
    good_ns = {"input_tokens": 10, "output_tokens": 5,
               "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    good_chat = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    # Official streaming split: input on start, output on delta.
    good_stream = {"message_start": {"input_tokens": 10, "output_tokens": 0},
                   "message_delta": {"input_tokens": 0, "output_tokens": 5}}
    print("--- self-check: contract-shaped stream, agreeing routes -> GREEN ---")
    rc_good = evaluate(good_ns, good_chat, good_stream)
    print("rc:", rc_good)

    # The server's actual shape: full usage on both events.
    dup_stream = {"message_start": {"input_tokens": 10, "output_tokens": 5},
                  "message_delta": {"input_tokens": 10, "output_tokens": 5}}
    print("--- self-check: duplicated full usage -> RED ---")
    rc_dup = evaluate(good_ns, good_chat, dup_stream)
    print("rc:", rc_dup)

    print("--- self-check: routes disagree -> RED ---")
    rc_mismatch = evaluate(good_ns, {"prompt_tokens": 9, "completion_tokens": 5},
                           good_stream)
    print("rc:", rc_mismatch)

    ok = rc_good == 0 and rc_dup == 1 and rc_mismatch == 1
    print("self-check", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="not-needed")
    p.add_argument("--model", default="qwen38-27b")
    p.add_argument("--prompt", default="Say the single word: ok")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--self-check", action="store_true",
                   help="validate the red/green logic locally, no network")
    args = p.parse_args()
    return self_check() if args.self_check else run_remote(args)


if __name__ == "__main__":
    sys.exit(main())
