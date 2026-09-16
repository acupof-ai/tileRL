#!/usr/bin/env python3
"""Probe whether the OpenAI Responses reasoning shape this server emits is
readable by the official openai-python Responses client.

Context — docs/api-compat-surface.md "live verification still owed", item 1:
src/tilerl/responses.py emits each reasoning item as

    {"type": "reasoning", "summary": [],
     "content": [{"type": "reasoning_text", "text": "..."}]}

That `content`/`reasoning_text` form was an earlier beta; current GA models the
item on `summary: [{"type": "summary_text", "text": "..."}]`. A current SDK that
only reads `summary` could therefore expose no reasoning. This script answers
that on a live server, plus whether `reasoning_tokens` is reported (static audit
found it hardcoded 0).

It does NOT validate the server's right answer — it reports what the SDK and the
raw JSON expose so a red result can open a server-fix issue with evidence.

Usage (run in a deployment window; default does NOT contact anything):
    python scripts/probe_responses_reasoning_shape.py \
        --base-url http://localhost:8000/v1 --model qwen38-27b \
        --prompt "Think for a sentence, then say hello." --max-output-tokens 128

Exit code:
    0  reasoning text readable from the GA summary field AND reasoning_tokens > 0
    1  probe ran but an expectation failed (printable RED evidence)
    2  could not talk to the server / no reasoning item to inspect

Dry-run / self-check without a server:
    python scripts/probe_responses_reasoning_shape.py --self-check
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _extract_reasoning_text_ga(item: dict[str, Any]) -> str:
    """Read reasoning the way a current GA client is expected to: summary[]."""
    parts: list[str] = []
    for s in item.get("summary") or []:
        if isinstance(s, dict) and s.get("type") == "summary_text":
            t = s.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def _extract_reasoning_text_legacy(item: dict[str, Any]) -> str:
    """The deprecated beta form the server currently emits: content[].text."""
    parts: list[str] = []
    for c in item.get("content") or []:
        if isinstance(c, dict) and c.get("type") == "reasoning_text":
            t = c.get("text")
            if isinstance(t, str):
                parts.append(t)
        # Some early beta builds put plain text on the content entry.
        elif isinstance(c, dict) and isinstance(c.get("text"), str):
            parts.append(c["text"])
    return "".join(parts)


def evaluate(raw: dict[str, Any]) -> int:
    """Apply the red/green criteria to one raw /v1/responses body. Returns code."""
    output = raw.get("output") or []
    reasoning_items = [o for o in output if isinstance(o, dict) and o.get("type") == "reasoning"]
    message_items = [o for o in output if isinstance(o, dict) and o.get("type") == "message"]

    print("=== status ===")
    print("status:", raw.get("status"))

    if not reasoning_items:
        print("NO reasoning item in output. The prompt may not have produced reasoning "
              "(thinking disabled?) or the server omitted it. Cannot judge shape.")
        print("output item types:", [o.get("type") for o in output if isinstance(o, dict)])
        return 2

    item = reasoning_items[0]
    print("=== raw first reasoning item ===")
    print(json.dumps(item, ensure_ascii=False, indent=2))

    ga_text = _extract_reasoning_text_ga(item)
    legacy_text = _extract_reasoning_text_legacy(item)
    print("=== readable text by shape ===")
    print(f"GA summary[summary_text].text  len={len(ga_text)}: {ga_text[:120]!r}")
    print(f"legacy content[reasoning_text] len={len(legacy_text)}: {legacy_text[:120]!r}")

    # SDK cross-check: let openai-python itself validate/parse the raw body.
    sdk_summary = "<sdk not checked>"
    try:
        from openai.types.responses import Response  # type: ignore

        parsed = Response.model_validate(raw)
        sdk_items = [o for o in parsed.output if o.type == "reasoning"]
        if sdk_items:
            ri = sdk_items[0]
            # The SDK model field that holds GA summaries; absent/empty on the
            # legacy shape demonstrates the readability gap concretely.
            summaries = getattr(ri, "summary", None)
            if summaries:
                sdk_summary = " | ".join(
                    getattr(x, "text", "") for x in summaries if getattr(x, "type", "") == "summary_text"
                )
            else:
                sdk_summary = ""
        print("=== openai-python parse ===")
        print("Response.model_validate: OK")
        print(f"SDK reasoning.summary text len={len(sdk_summary)}: {sdk_summary[:120]!r}")
    except Exception as exc:  # noqa: BLE001 - report any SDK rejection
        print("=== openai-python parse ===")
        print("Response.model_validate FAILED:", f"{type(exc).__name__}: {exc}")

    usage = raw.get("usage") or {}
    out_details = usage.get("output_tokens_details") or {}
    reasoning_tokens = out_details.get("reasoning_tokens")
    print("=== usage ===")
    print("output_tokens_details:", out_details)
    print("reasoning_tokens:", reasoning_tokens)

    answer_text = ""
    for m in message_items:
        for c in m.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "output_text":
                answer_text += c.get("text", "")
    print("answer text len:", len(answer_text))

    red: list[str] = []
    if not ga_text:
        red.append("reasoning NOT readable from GA summary[summary_text] "
                   "(a current SDK sees no reasoning; legacy content/reasoning_text present="
                   f"{bool(legacy_text)})")
    if not reasoning_tokens or reasoning_tokens <= 0:
        red.append("reasoning_tokens is 0/missing while a reasoning item exists")

    if red:
        print("=== RESULT: RED ===")
        for r in red:
            print(" -", r)
        return 1
    print("=== RESULT: GREEN ===")
    print("reasoning readable from GA summary and reasoning_tokens > 0")
    return 0


def run_remote(args: argparse.Namespace) -> int:
    # Try the SDK first: its internal model_validate is itself a compatibility
    # probe (a missing required usage field raises before reasoning is reached).
    # If the SDK rejects the envelope, fall back to raw HTTP so the reasoning
    # shape can still be judged and the envelope failure reported alongside.
    try:
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        resp = client.responses.create(
            model=args.model,
            input=args.prompt,
            max_output_tokens=args.max_output_tokens,
        )
        print("sdk responses.create: OK")
        return evaluate(resp.model_dump())
    except Exception as exc:  # noqa: BLE001
        print("sdk responses.create FAILED:", f"{type(exc).__name__}: {exc}")
        print("falling back to raw HTTP to inspect the reasoning shape anyway...")
        import urllib.request

        url = args.base_url.rstrip("/") + "/responses"
        body = json.dumps({
            "model": args.model,
            "input": args.prompt,
            "max_output_tokens": args.max_output_tokens,
        }).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json",
                                     "Authorization": f"Bearer {args.api_key}"})
        with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 - configured base url
            raw = json.loads(r.read().decode())
        print("raw HTTP fetch: OK")
        evaluate(raw)
        # An SDK envelope rejection is itself a red for compatibility even if the
        # raw reasoning shape happened to parse.
        return 1


def self_check() -> int:
    """Exercise the evaluator against one GA-shaped and one legacy-shaped body,
    no network. Used to prove the red/green logic before the real run."""
    base = {
        "id": "resp_1", "object": "response", "created_at": 1.0, "model": "m",
        "status": "completed", "incomplete_details": None, "error": None,
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "instructions": None, "metadata": {}, "temperature": None, "top_p": None,
    }
    ga = {**base, "output": [
        {"id": "rs_1", "type": "reasoning", "status": "completed",
         "summary": [{"type": "summary_text", "text": "GA thinking"}]},
        {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "hi", "annotations": []}]}],
        "usage": {"input_tokens": 1, "output_tokens": 5, "total_tokens": 6,
                  "input_tokens_details": {"cached_tokens": 0,
                                           "cache_read_input_tokens": 0,
                                           "cache_write_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 4}}}
    legacy = {**base, "output": [
        {"id": "rs_1", "type": "reasoning", "status": "completed", "summary": [],
         "content": [{"type": "reasoning_text", "text": "LEGACY thinking"}]},
        {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "hi", "annotations": []}]}],
        "usage": {"input_tokens": 1, "output_tokens": 5, "total_tokens": 6,
                  "input_tokens_details": {"cached_tokens": 0,
                                           "cache_read_input_tokens": 0,
                                           "cache_write_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}}}
    print("--- self-check: GA body should be GREEN ---")
    rc_ga = evaluate(ga)
    print("rc:", rc_ga)
    print("--- self-check: current-server legacy body should be RED ---")
    rc_legacy = evaluate(legacy)
    print("rc:", rc_legacy)
    ok = rc_ga == 0 and rc_legacy == 1
    print("self-check", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="not-needed")
    p.add_argument("--model", default="qwen38-27b")
    p.add_argument("--prompt", default="Think briefly about what 2+2 is, then answer.")
    p.add_argument("--max-output-tokens", type=int, default=128)
    p.add_argument("--self-check", action="store_true",
                   help="validate the red/green logic locally, no network")
    args = p.parse_args()
    return self_check() if args.self_check else run_remote(args)


if __name__ == "__main__":
    sys.exit(main())
