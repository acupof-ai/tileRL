"""Anthropic Messages API in front of the tileRL engine.

Claude Code speaks this, not OpenAI chat completions: it POSTs
``/v1/messages`` with ``system`` as a block list, ``tools`` as JSON Schema, and
message content as blocks. Measured against Claude Code 2.1.258 on 2026-09-02
(docs/lessons/messages_shim_stage1.md) -- the field list here is what a real
request carries, not what the public docs describe.

Two things this module owes the RL loop, and they are the reason it exists at
all rather than being a translation layer someone writes twice:

* **The token ids are recorded, not reconstructed.** BPE is not
  concatenation-invariant, so re-encoding a transcript's *text* does not
  reliably reproduce the ids that were sampled. GRPO trained on a mismatched
  sequence is a silently wrong gradient. Every request appends one JSONL row
  carrying prompt ids, completion ids, per-token logprobs and stop_reason, and
  the id is returned in the ``x-tilerl-request-id`` header so a transcript can
  be joined by identity instead of by text.
* **It is the single reader of the engine's logprobs.** ``Engine.logprobs``
  pops; a second reader would get a KeyError (engine.py), which is the whole
  point -- an empty score list reaching the trainer is the failure that cannot
  be traced back here.

# ponytail: tools are rendered into the prompt as text and tool_use is parsed
# back out of the completion. A tokenizer with real tool-call special tokens is
# the upgrade path; tiny has no such ids, and inventing them would make the
# tiny run diverge from what the 27B will do.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .tokenizer import Tokenizer, render_chat

__all__ = ["mount_messages", "MessagesRequest", "record_path"]

#: One JSONL row per request. Overridable so a test does not write the real one.
_RECORD_ENV = "TILERL_MESSAGES_RECORD"

#: The episode tag, set per rollout via ANTHROPIC_CUSTOM_HEADERS.
_ROLLOUT_HEADER = "x-tilerl-rollout"


def record_path() -> str:
    return os.environ.get(_RECORD_ENV, "runs/messages_requests.jsonl")


class MessagesRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int = Field(default=512, ge=1)
    #: str in the docs, a list of {type,text,cache_control} blocks in practice
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    #: {"type": "adaptive"} from Claude Code; only its presence is used
    thinking: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    context_management: Any | None = None


def _blocks_to_text(content: Any) -> str:
    """Flatten Anthropic content to the text a ChatML turn carries.

    tool_result is folded into the user turn and tool_use into the assistant
    turn, in the same shape :func:`_render_tools` advertises, so a transcript
    replay and a live request render identically.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out: list[str] = []
    for b in content if isinstance(content, list) else []:
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "text":
            out.append(b.get("text", ""))
        elif kind == "tool_use":
            out.append(json.dumps({"tool": b.get("name"), "input": b.get("input") or {}},
                                  ensure_ascii=False, sort_keys=True))
        elif kind == "tool_result":
            body = b.get("content")
            out.append(f"<tool_result>{_blocks_to_text(body)}</tool_result>")
    return "\n".join(x for x in out if x)


def _render_tools(tools: list[dict[str, Any]] | None) -> str:
    """The tool contract, as text the model can answer in.

    One line per tool plus the exact JSON shape a call must take. Claude Code
    sends 28 tools by default and their schemas are most of the prompt, so only
    name + description + the top-level property names go in; a full JSON Schema
    dump was 6,121 characters of the 21,676-byte minimal request.
    """
    if not tools:
        return ""
    lines = ["You may call a tool. To do so, reply with ONLY this JSON and nothing else:",
             '{"tool": "<name>", "input": {...}}', "", "Available tools:"]
    for t in tools:
        props = ((t.get("input_schema") or {}).get("properties") or {})
        args = ", ".join(sorted(props)) or "no arguments"
        lines.append(f"- {t.get('name')}({args}): {(t.get('description') or '').splitlines()[0][:160]}")
    return "\n".join(lines)


_TOOL_CALL_RE = re.compile(r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"input"\s*:\s*(\{.*\})\s*\}', re.S)


def _parse_tool_use(text: str) -> tuple[str, dict[str, Any]] | None:
    """A tool call in the completion, or None.

    Deliberately tolerant of leading prose: a small model rarely emits the bare
    object the prompt asks for, and refusing anything with a preamble would
    make the tiny run fail for a reason the 27B will not have.
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        return m.group(1), json.loads(m.group(2))
    except json.JSONDecodeError:
        return None


def mount_messages(app: FastAPI, engine: Any, tokenizer: Tokenizer, model_name: str) -> FastAPI:
    """Add POST /v1/messages to an existing app, sharing its engine."""
    from .engine import SamplingParams

    def _render(req: MessagesRequest) -> str:
        turns: list[tuple[str, str]] = []
        sys_text = _blocks_to_text(req.system)
        tools_text = _render_tools(req.tools)
        if sys_text or tools_text:
            turns.append(("system", "\n\n".join(x for x in (sys_text, tools_text) if x)))
        for m in req.messages:
            turns.append((str(m.get("role", "user")), _blocks_to_text(m.get("content"))))
        return render_chat(turns)

    def _record(row: dict[str, Any]) -> None:
        path = record_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _run(req: MessagesRequest, rollout: str | None = None) -> tuple[dict[str, Any], int]:
        input_ids = tokenizer.encode(_render(req))
        if not input_ids:
            raise ValueError("empty prompt after tokenization")
        params = SamplingParams(
            temperature=req.temperature if req.temperature is not None else 0.0,
            top_p=req.top_p if req.top_p is not None else 1.0,
            max_new_tokens=req.max_tokens,
            seed=secrets.randbits(31),
            stop_token_ids=tuple(getattr(tokenizer, "stop_token_ids", ())),
            logprobs=True,  # always: the record is the point of this route
        )
        rid = engine.submit(input_ids, params)
        deadline = time.monotonic() + 600.0
        out: list[int] | None = None
        while time.monotonic() < deadline:
            out = engine.take(rid)
            if out is not None:
                break
            time.sleep(0.02)
        if out is None:
            raise TimeoutError(f"request {rid} did not finish within 600s")
        scores = engine.logprobs(rid)  # single reader; a second one raises
        text = tokenizer.decode(out)
        call = _parse_tool_use(text)
        if call is not None:
            name, tool_input = call
            content = [{"type": "tool_use", "id": f"toolu_{rid}", "name": name, "input": tool_input}]
            stop_reason = "tool_use"
        else:
            content = [{"type": "text", "text": text}]
            stop_reason = "max_tokens" if len(out) >= params.max_new_tokens else "end_turn"
        _record({
            "request_id": rid,
            "rollout": rollout,
            "model": req.model or model_name,
            "prompt_ids": [int(t) for t in input_ids],
            "completion_ids": [int(t) for t in out],
            "logprobs": scores,
            "stop_reason": stop_reason,
        })
        return {
            "id": f"msg_{rid}",
            "type": "message",
            "role": "assistant",
            "model": req.model or model_name,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": len(input_ids), "output_tokens": len(out)},
        }, rid

    @app.post("/v1/messages")
    async def messages(req: MessagesRequest, request: Request):  # noqa: D401 - route
        # One rollout is many requests (Claude Code turns), and GRPO's advantage
        # is per EPISODE, not per turn -- so every row needs the episode's tag.
        # ANTHROPIC_CUSTOM_HEADERS carries it: measured 2026-09-02, the CLI
        # passes it through verbatim, which metadata.user_id would not survive.
        rollout = request.headers.get(_ROLLOUT_HEADER)
        try:
            body, rid = _run(req, rollout)
        except ValueError as exc:
            return JSONResponse(status_code=400,
                                content={"type": "error",
                                         "error": {"type": "invalid_request_error",
                                                   "message": str(exc)}})
        except (TimeoutError, RuntimeError) as exc:
            return JSONResponse(status_code=503,
                                content={"type": "error",
                                         "error": {"type": "overloaded_error",
                                                   "message": str(exc)}})
        headers = {"x-tilerl-request-id": str(rid)}
        if not req.stream:
            return JSONResponse(content=body, headers=headers)

        def sse():
            # One whole-message delta, not per-token: the record carries the
            # token sequence, so streaming granularity is a UI concern here.
            # ponytail: per-token deltas when a human watches this live.
            blk = body["content"][0]
            start = {"type": "message_start",
                     "message": {**body, "content": [], "usage": body["usage"]}}
            yield f"event: message_start\ndata: {json.dumps(start)}\n\n"
            yield ("event: content_block_start\ndata: "
                   + json.dumps({"type": "content_block_start", "index": 0,
                                 "content_block": {**blk, "text": "", "input": {}}
                                 if blk["type"] == "tool_use"
                                 else {"type": "text", "text": ""}}) + "\n\n")
            if blk["type"] == "text":
                yield ("event: content_block_delta\ndata: "
                       + json.dumps({"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": blk["text"]}})
                       + "\n\n")
            else:
                yield ("event: content_block_delta\ndata: "
                       + json.dumps({"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "input_json_delta",
                                               "partial_json": json.dumps(blk["input"])}})
                       + "\n\n")
            yield ("event: content_block_stop\ndata: "
                   + json.dumps({"type": "content_block_stop", "index": 0}) + "\n\n")
            yield ("event: message_delta\ndata: "
                   + json.dumps({"type": "message_delta",
                                 "delta": {"stop_reason": body["stop_reason"],
                                           "stop_sequence": None},
                                 "usage": body["usage"]}) + "\n\n")
            yield ("event: message_stop\ndata: "
                   + json.dumps({"type": "message_stop"}) + "\n\n")

        return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)

    return app


if __name__ == "__main__":  # pragma: no cover - self-check
    assert _parse_tool_use('{"tool": "Bash", "input": {"command": "ls"}}') == (
        "Bash", {"command": "ls"})
    assert _parse_tool_use('sure, here you go:\n{"tool":"Read","input":{"file_path":"a"}}') == (
        "Read", {"file_path": "a"})
    assert _parse_tool_use("no call here") is None
    assert _parse_tool_use('{"tool": "Bash", "input": {broken}}') is None
    assert _blocks_to_text([{"type": "text", "text": "hi"},
                            {"type": "tool_result", "content": "out"}]) == "hi\n<tool_result>out</tool_result>"
    assert "Bash(command)" in _render_tools(
        [{"name": "Bash", "description": "Run it\nmore", "input_schema": {"properties": {"command": {}}}}])
    print("messages self-check ok")
