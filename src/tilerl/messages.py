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
# back out of the completion, in the checkpoint's own <tool_call> XML (read off
# chat_template.jinja, 2026-09-02). Special tool tokens would be the upgrade,
# but the template itself is textual, so there is nothing to upgrade to.
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

from .tokenizer import SAMPLING, Tokenizer, render_chat

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

    tool_use renders as the checkpoint's own ``<tool_call>`` XML and tool_result
    as ``<tool_response>``, so a replayed transcript is byte-identical to what
    the model was trained on. Reasoning is stripped: the template re-inserts
    ``<think>`` only for turns after the last real user query, so feeding old
    reasoning back would be off-distribution.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return _strip_think(content)
    out: list[str] = []
    for b in content if isinstance(content, list) else []:
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "text":
            out.append(_strip_think(b.get("text", "")))
        elif kind == "tool_use":
            out.append(render_tool_call(b.get("name") or "", b.get("input") or {}))
        elif kind == "tool_result":
            out.append(f"<tool_response>\n{_blocks_to_text(b.get('content'))}\n</tool_response>")
        elif kind in ("image", "document"):
            # Dropping these silently would send the model a turn that is
            # missing its subject; the 27B is text-only, so say so.
            raise ValueError(f"{kind} blocks are not supported by this model")
    return "\n".join(x for x in out if x)


#: Verbatim from the checkpoint's chat_template.jinja (read on the pod
#: 2026-09-02). Copied rather than paraphrased: this text is what the model was
#: trained to answer in, so a reworded version is a different distribution.
_TOOL_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is no "
    "function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n</IMPORTANT>"
)

#: The template's own wording per reasoning_effort; "medium" renders nothing.
_EFFORT_INSTRUCTIONS = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, "
             "validate key assumptions, consider plausible alternatives, and prioritize "
             "correctness, consistency, and clarity in the final answer.",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
           "directly to the conclusion without unnecessary elaboration.",
}

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def _strip_think(text: str) -> str:
    """Drop a reasoning block from historical assistant text."""
    return _THINK_RE.sub("", text)


def render_tool_call(name: str, args: dict[str, Any]) -> str:
    """One ``<tool_call>`` block, in the template's own shape.

    Non-string values are JSON, strings are raw -- exactly what the template's
    ``args_value | tojson`` branch does, so a replayed assistant turn matches
    what apply_chat_template would have produced.
    """
    lines = [f"<tool_call>\n<function={name}>"]
    for k, v in args.items():
        val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        lines.append(f"<parameter={k}>\n{val}\n</parameter>")
    lines.append("</function>\n</tool_call>")
    return "\n".join(lines)


def _render_tools(tools: list[dict[str, Any]] | None, effort: str | None = None) -> str:
    """The system turn's tool section, as the checkpoint's template builds it.

    Whole tool defs as JSON, one per line, inside ``<tools>`` -- not a trimmed
    summary. The 28 schemas Claude Code sends are most of the prompt, but the
    model was trained on the full defs and inventing a shorter form would be a
    format it has never seen.
    """
    head = _EFFORT_INSTRUCTIONS.get((effort or "").lower(), "")
    if not tools:
        return head
    body = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
    for t in tools:
        body += "\n" + json.dumps(t, ensure_ascii=False)
    body += "\n</tools>" + _TOOL_INSTRUCTIONS
    return (head + "\n\n" + body) if head else body


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(value: str, schema: dict[str, Any] | None) -> Any:
    """A parameter's text as the schema's type.

    The model writes every value as text inside ``<parameter>``; strings stay
    raw and everything else is JSON. Falling back to the raw string on a parse
    failure is deliberate -- a malformed number should reach the tool as what
    the model actually said, not vanish.
    """
    if (schema or {}).get("type") == "string":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_tool_calls(text: str,
                      tools: list[dict[str, Any]] | None = None
                      ) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """``(leading prose, [(name, args), ...])`` from a completion.

    Every ``<tool_call>`` block, not just the first: the template joins parallel
    calls with a newline and Claude Code issues them, expecting one tool_use
    block back per call. Reasoning may precede the calls and never follows
    them, so the text before the first call is a real content block -- the API
    keeps it, and dropping it loses the model's own explanation.
    """
    schemas = {t.get("name"): (t.get("input_schema") or {}).get("properties") or {}
               for t in tools or []}
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        props = schemas.get(m.group(1)) or {}
        calls.append((m.group(1),
                      {k: _coerce(v, props.get(k)) for k, v in _PARAM_RE.findall(m.group(2))}))
    first = _TOOL_CALL_RE.search(text)
    return (text[:first.start()] if first else text).strip(), calls


def _thinking(req: MessagesRequest) -> bool:
    """Whether the prompt opens a reasoning block.

    Claude Code sends ``thinking: {"type": "adaptive"}``; anything but an
    explicit ``disabled`` leaves it on, which matches the template's default.
    """
    kind = (req.thinking or {}).get("type")
    return kind != "disabled"


def _effort(req: MessagesRequest) -> str | None:
    """The template's reasoning_effort. It knows xhigh/medium/low only, and
    aliases high; Claude Code's "max" has to be aliased too or it silently
    renders as medium."""
    e = ((req.output_config or {}).get("effort") or "").lower()
    return "xhigh" if e in ("high", "max") else (e or None)


def mount_messages(app: FastAPI, engine: Any, tokenizer: Tokenizer, model_name: str) -> FastAPI:
    """Add POST /v1/messages to an existing app, sharing its engine."""
    from .engine import SamplingParams

    # The engine refuses prompt+max_new_tokens over its budget; the real API
    # accepts any max_tokens and stops at the context edge, so the completion
    # is clamped to what is left rather than the request rejected.
    engine_limit = getattr(getattr(engine, "limits", None), "max_total_tokens", 0)

    def _render(req: MessagesRequest) -> str:
        # The template merges leading system/developer turns and puts the
        # reasoning-effort sentence FIRST, ahead of the tool block.
        turns: list[tuple[str, str]] = []
        sys_text = _blocks_to_text(req.system)
        tools_text = _render_tools(req.tools, _effort(req))
        if sys_text or tools_text:
            turns.append(("system", "\n\n".join(x for x in (tools_text, sys_text) if x)))
        for m in req.messages:
            turns.append((str(m.get("role", "user")), _blocks_to_text(m.get("content"))))
        # Thinking is switched in the PROMPT, not by forcing tokens at decode
        # time: "assistant\n<think>\n" when on, a closed empty block when off.
        return render_chat(turns, thinking=_thinking(req))

    def _record(row: dict[str, Any]) -> None:
        path = record_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _run(req: MessagesRequest, rollout: str | None = None) -> tuple[dict[str, Any], int]:
        input_ids = tokenizer.encode(_render(req))
        if not input_ids:
            raise ValueError("empty prompt after tokenization")
        thinking = _thinking(req)
        # The model card's values per thinking mode, since Claude Code sends
        # neither temperature nor top_p -- so this table IS the effective
        # default, not a fallback. An explicit request value still wins.
        # ponytail: the card also wants presence_penalty 1.5 non-thinking; the
        # engine has no such knob, add it when a run shows repetition.
        budget = max(1, engine_limit - len(input_ids)) if engine_limit else req.max_tokens
        params = SamplingParams(
            **SAMPLING[thinking]
            | {k: v for k in ("temperature", "top_p")
               if (v := getattr(req, k, None)) is not None},
            # The real API accepts any max_tokens and stops at the context
            # edge; refusing the whole request would 400 every Claude Code
            # turn, which always asks for 32000.
            max_new_tokens=min(req.max_tokens, budget),
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
        text = _strip_think(tokenizer.decode(out))
        prose, calls = _parse_tool_calls(text, req.tools)
        content: list[dict[str, Any]] = []
        if prose or not calls:
            content.append({"type": "text", "text": prose})
        # One block per call, distinct ids: Claude Code runs them in parallel
        # and returns one tool_result per id.
        content += [{"type": "tool_use", "id": f"toolu_{rid}_{i}", "name": n, "input": a}
                    for i, (n, a) in enumerate(calls)]
        if calls:
            stop_reason = "tool_use"
        else:
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
            # stop_sequences is accepted and ignored, so the API's
            # "stop_sequence" stop_reason never occurs here.
            "stop_sequence": None,
            "usage": {"input_tokens": len(input_ids), "output_tokens": len(out),
                      # Claude Code reads these for context accounting; we
                      # cache nothing, and 0 is the shape it expects.
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
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
            # One whole-message delta per block, not per-token: the record
            # carries the token sequence, so streaming granularity is a UI
            # concern here.
            # ponytail: per-token deltas when a human watches this live.
            def ev(name: str, payload: dict[str, Any]) -> str:
                return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

            yield ev("message_start", {"type": "message_start",
                                       "message": {**body, "content": []}})
            for i, blk in enumerate(body["content"]):
                tool = blk["type"] == "tool_use"
                opening = ({k: v for k, v in blk.items() if k != "input"} | {"input": {}}
                           if tool else {"type": "text", "text": ""})
                yield ev("content_block_start", {"type": "content_block_start",
                                                 "index": i, "content_block": opening})
                delta = ({"type": "input_json_delta",
                          "partial_json": json.dumps(blk["input"])} if tool
                         else {"type": "text_delta", "text": blk["text"]})
                yield ev("content_block_delta",
                         {"type": "content_block_delta", "index": i, "delta": delta})
                yield ev("content_block_stop", {"type": "content_block_stop", "index": i})
            yield ev("message_delta", {"type": "message_delta",
                                       "delta": {"stop_reason": body["stop_reason"],
                                                 "stop_sequence": None},
                                       "usage": body["usage"]})
            yield ev("message_stop", {"type": "message_stop"})

        return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)

    return app


if __name__ == "__main__":  # pragma: no cover - self-check
    call = render_tool_call("Bash", {"command": "ls", "timeout": 30})
    assert call == ("<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n"
                    "<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>"), call
    # Round trip: what we render for a replayed turn is what we parse from a
    # completion, so a transcript and a live call cannot diverge.
    schema = [{"name": "Bash", "input_schema": {"properties": {
        "command": {"type": "string"}, "timeout": {"type": "integer"}}}}]
    assert _parse_tool_calls(call, schema) == ("", [("Bash", {"command": "ls", "timeout": 30})])
    # Reasoning may PRECEDE a call and never follows it (the template says so),
    # and the API keeps that prose as its own text block.
    assert _parse_tool_calls("I should list them.\n" + call, schema) == (
        "I should list them.", [("Bash", {"command": "ls", "timeout": 30})])
    # Parallel calls: the template joins them, Claude Code expects them all.
    two = call + "\n" + render_tool_call("Bash", {"command": "pwd"})
    assert [n for n, _ in _parse_tool_calls(two, schema)[1]] == ["Bash", "Bash"]
    assert _parse_tool_calls("no call here")[1] == []
    # A string-typed parameter stays raw even when it looks like JSON.
    s = render_tool_call("Read", {"file_path": "123"})
    assert _parse_tool_calls(s, [{"name": "Read", "input_schema": {
        "properties": {"file_path": {"type": "string"}}}}])[1] == [("Read", {"file_path": "123"})]
    # ...and with no schema at all it parses as a number, which is why the
    # schema is passed in rather than guessed from the text.
    assert _parse_tool_calls(s)[1] == [("Read", {"file_path": 123})]
    assert _blocks_to_text([{"type": "text", "text": "hi"},
                            {"type": "tool_result", "content": "out"}]) == (
        "hi\n<tool_response>\nout\n</tool_response>")
    assert _strip_think("<think>\nplanning\n</think>\n\nthe answer") == "the answer"
    tools = _render_tools([{"name": "Bash", "description": "Run it",
                            "input_schema": {"properties": {"command": {}}}}], "low")
    assert tools.startswith("Reasoning effort is set to low.")
    assert '<tools>\n{"name": "Bash"' in tools and "</tools>" in tools
    assert "<IMPORTANT>" in tools
    print("messages self-check ok")
