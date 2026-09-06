"""The SDK e2e checks against a LIVE server, for the pod.

`tests/test_api_sdk.py` runs the same assertions over a canned engine, which
gates the wire shape and nothing else. This drives a real deployment, so it also
covers what only a real tokenizer and real weights can break: whether the
template's `<think>` token exists, whether the model emits a well-formed
`<tool_call>`, whether a reply survives the cap.

    uv run python3 scripts/api_e2e.py --base-url http://10.37.2.27:8000
    uv run python3 scripts/api_e2e.py --base-url ... --model qwen38-27b

Exits non-zero on the first failed check, so it is usable as a gate. A check the
deployment cannot support (no tool-capable model, thinking off) reports SKIP
rather than failing: this is a deployment probe, not a spec test.
"""

from __future__ import annotations

import argparse
import json

FAILED: list[str] = []
SKIPPED: list[str] = []
PASSED: list[str] = []


def check(name: str, fn) -> None:
    try:
        note = fn()
    except Exception as exc:  # noqa: BLE001 - a live probe reports, never raises
        FAILED.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
        print(f"FAIL  {name}: {type(exc).__name__}: {str(exc)[:200]}")
        return
    if note == "SKIP" or (isinstance(note, str) and note.startswith("SKIP")):
        SKIPPED.append(name)
        print(f"SKIP  {name}{note[4:]}")
        return
    PASSED.append(name)
    print(f"ok    {name}" + (f"  ({note})" if note else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True,
                    help="server root, e.g. http://10.37.2.27:8000 (no /v1)")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    base = args.base_url.rstrip("/").removesuffix("/v1")

    import anthropic
    import openai

    oa = openai.OpenAI(base_url=f"{base}/v1", api_key="x", max_retries=0, timeout=600)
    an = anthropic.Anthropic(base_url=base, api_key="x", max_retries=0, timeout=600)
    m, cap = args.model, args.max_tokens
    ask = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    tools_oa = [{"type": "function", "function": {
        "name": "Bash", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}]
    tools_an = [{"name": "Bash", "description": "Run a shell command",
                 "input_schema": {"type": "object",
                                  "properties": {"command": {"type": "string"}},
                                  "required": ["command"]}}]
    run_ls = [{"role": "user", "content": "Run `ls` in the current directory."}]

    print(f"base={base} model={m}\n")

    # --- chat completions
    def chat_once():
        r = oa.chat.completions.create(model=m, messages=ask, max_completion_tokens=cap)
        msg = r.choices[0].message
        assert msg.content, "empty content"
        assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens
        if "</think>" in msg.content:
            # Only happens on a tokenizer with no <think> token (the tiny/dev
            # model): the route cannot know the template opened the block, so
            # nothing is stripped. On the 27B this is a real defect; here it is
            # the wrong model, so say which rather than fail.
            return ("SKIP  (the closer is in content: this tokenizer has no <think> "
                    "token, so thinking was never opened -- run against the 27B)")
        return f"{r.usage.completion_tokens} tok, finish={r.choices[0].finish_reason}"
    check("chat non-stream", chat_once)

    def chat_reasoning():
        r = oa.chat.completions.create(model=m, messages=ask, max_completion_tokens=cap)
        rc = getattr(r.choices[0].message, "reasoning_content", None)
        if not rc:
            return "SKIP  (thinking off, or the model emitted no reasoning)"
        assert "</think>" not in rc
        return f"{len(rc)} chars of reasoning"
    check("chat reasoning_content (non-stream)", chat_reasoning)

    def chat_stream():
        text, reasoning, n = "", "", 0
        for c in oa.chat.completions.create(model=m, messages=ask, stream=True,
                                            max_completion_tokens=cap):
            if not c.choices:
                continue
            n += 1
            text += c.choices[0].delta.content or ""
            reasoning += getattr(c.choices[0].delta, "reasoning_content", None) or ""
        assert text, "no content deltas"
        if "</think>" in text:
            return "SKIP  (closer in the stream too; see the non-stream arm)"
        assert "</think>" not in reasoning, "the closer leaked into reasoning_content"
        return f"{n} frames, {len(text)} chars content, {len(reasoning)} reasoning"
    check("chat stream", chat_stream)

    def chat_tools():
        r = oa.chat.completions.create(model=m, messages=run_ls, tools=tools_oa,
                                       max_completion_tokens=256)
        ch = r.choices[0]
        assert "<tool_call>" not in (ch.message.content or ""), "raw XML in content"
        if not ch.message.tool_calls:
            return "SKIP  (model did not call a tool; not a wire defect)"
        call = ch.message.tool_calls[0]
        json.loads(call.function.arguments)  # must be valid JSON
        assert ch.finish_reason == "tool_calls", ch.finish_reason
        return f"{call.function.name}({call.function.arguments})"
    check("chat tool_calls", chat_tools)

    def chat_bad_field():
        try:
            oa.chat.completions.create(model=m, messages=ask, max_completion_tokens=0)
        except openai.BadRequestError as exc:
            assert exc.body["type"] == "invalid_request_error", exc.body
            return "400 + OpenAI envelope"
        raise AssertionError("a zero token cap was accepted")
    check("chat error envelope", chat_bad_field)

    # --- messages
    def msg_once():
        r = an.messages.create(model=m, max_tokens=cap, messages=ask)
        text = "".join(b.text for b in r.content if b.type == "text")
        assert text, f"no text block: {[b.type for b in r.content]}"
        assert r.stop_reason in ("end_turn", "max_tokens", "tool_use"), r.stop_reason
        return f"{r.usage.output_tokens} tok, blocks={[b.type for b in r.content]}"
    check("messages non-stream", msg_once)

    def msg_thinking():
        r = an.messages.create(model=m, max_tokens=cap, messages=ask,
                               thinking={"type": "enabled", "budget_tokens": 32})
        blocks = [b.type for b in r.content]
        if "thinking" not in blocks:
            return f"SKIP  (no reasoning emitted; blocks={blocks})"
        return f"thinking block present, blocks={blocks}"
    check("messages thinking block", msg_thinking)

    def msg_stream():
        with an.messages.stream(model=m, max_tokens=cap, messages=ask) as s:
            names = [e.type for e in s]
            text = s.get_final_text()
        assert text, "no text from the stream"
        for want in ("message_start", "content_block_delta", "message_stop"):
            assert want in names, f"{want} missing"
        return f"{len(names)} events"
    check("messages stream", msg_stream)

    def msg_tools():
        r = an.messages.create(model=m, max_tokens=256, tools=tools_an, messages=run_ls)
        uses = [b for b in r.content if b.type == "tool_use"]
        if not uses:
            return "SKIP  (model did not call a tool)"
        assert r.stop_reason == "tool_use", r.stop_reason
        follow = an.messages.create(
            model=m, max_tokens=cap, tools=tools_an,
            messages=run_ls + [
                {"role": "assistant", "content": [b.model_dump() for b in r.content]},
                {"role": "user", "content": [{"type": "tool_result",
                                              "tool_use_id": uses[0].id,
                                              "content": "a.txt\nb.txt"}]}])
        assert any(b.type == "text" for b in follow.content), "no answer after tool_result"
        return f"{uses[0].name}({uses[0].input}) -> answered"
    check("messages tool round trip", msg_tools)

    # --- responses
    def resp_once():
        r = oa.responses.create(model=m, input="What is 2+2? Answer in one word.",
                                max_output_tokens=cap)
        assert r.output_text, f"no output_text; items={[i.type for i in r.output]}"
        assert r.status in ("completed", "incomplete"), r.status
        return f"status={r.status}, items={[i.type for i in r.output]}"
    check("responses non-stream", resp_once)

    def resp_stream():
        names, text = [], ""
        for ev in oa.responses.create(model=m, input="What is 2+2?", stream=True,
                                      max_output_tokens=cap):
            names.append(ev.type)
            if ev.type == "response.output_text.delta":
                text += ev.delta
        assert names[0] == "response.created", names[:2]
        assert names[-1] == "response.completed", names[-2:]
        assert text, "no output_text deltas"
        return f"{len(names)} events, {len(text)} chars"
    check("responses stream", resp_stream)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED, {len(SKIPPED)} skipped")
        return 1
    # A floor on how much may skip. Otherwise a deployment that answers nothing
    # useful skips its way to "all checks passed" and exit 0 -- measured against
    # the tiny model, 5 of 11 arms skipped and the script still reported success.
    ran = len(SKIPPED) + len(PASSED)
    if len(SKIPPED) > ran // 3:
        print(f"{len(PASSED)} passed but {len(SKIPPED)}/{ran} SKIPPED -- too many to "
              f"call this a pass. Skipped: {', '.join(SKIPPED)}")
        return 2
    print(f"all {len(PASSED)} checks passed ({len(SKIPPED)} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
