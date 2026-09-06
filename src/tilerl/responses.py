"""OpenAI Responses API in front of the tileRL engine.

The third surface. Same engine, same tokenizer, same prompt renderer and the
same ``split_think`` as ``/v1/chat/completions`` and ``/v1/messages`` -- what
differs is only the wire shape: a flat ``output`` list of typed items rather
than a ``choices`` array, and reasoning as its own ``reasoning`` item rather
than a sibling field.

Every field below was read off the SDK's own pydantic models
(``openai.types.responses``, openai 3.8.0) rather than the prose docs:
``Response`` declares ``parallel_tool_calls``, ``tool_choice`` and ``tools``
required, and each stream event a ``sequence_number``, none of which the doc
examples show.

The SDK does NOT enforce that -- measured: ``Response.model_validate`` rejects a
body missing ``parallel_tool_calls``, but the client builds replies with
``construct``, so the field arrives as ``None`` and nothing raises. So omitting a
required field is invisible to the SDK and visible to a client that reads it,
which is the reverse of the usual failure and the reason the tests assert these
fields by name rather than trusting a parse to catch them.

# ponytail: `store` is accepted and ignored, so there is no `previous_response_id`
# and no retrieval by id -- stateless only. Conversation state lives in the
# client's `input` list, which is what an agent loop sends anyway.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .messages import _parse_tool_calls
from .prompt import cut_at_stop, refuse_unsupported, render_prompt, sampling, split_think
from .tokenizer import Tokenizer


class ResponsesRequest(BaseModel):
    model: str | None = None
    #: A bare string or the typed item list; both are documented, and an agent
    #: loop sends the list because that is how it replays its own history.
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool | None = None
    #: Responses puts name/parameters at the top level, not under "function".
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
    reasoning: dict[str, Any] | None = None
    store: bool | None = None
    #: Declared only so they can be REFUSED. pydantic drops an undeclared field
    #: silently, so without these lines `previous_response_id` never reaches the
    #: handler and cannot be rejected -- measured: model_extra is None.
    previous_response_id: str | None = None
    include: list[str] | None = None
    truncation: str | None = None
    #: Not in the Responses schema at all, but honoured on the same terms as the
    #: chat route's: a client that sends one gets it applied, not ignored.
    stop: str | list[str] | None = None
    metadata: dict[str, Any] | None = None
    #: The same vLLM-style override the chat route takes, for the same reason:
    #: whether the template opens <think> is otherwise inferred from the
    #: tokenizer, and a caller on a tokenizer without the token cannot ask.
    chat_template_kwargs: dict | None = None


def _to_messages(inp: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``input`` union as chat turns.

    A function_call_output item is a tool result, and it carries no role at all
    -- rendering it as a user turn is what lets an agent loop's second request
    replay its own history through the same template the first one used.
    """
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]
    out = []
    for item in inp:
        kind = item.get("type", "message")
        if kind == "function_call_output":
            out.append({"role": "user",
                        "content": [{"type": "tool_result",
                                     "content": item.get("output", "")}]})
        elif kind == "function_call":
            out.append({"role": "assistant",
                        "content": _render_call(item.get("name", ""),
                                                item.get("arguments", "{}"))})
        else:
            out.append({"role": item.get("role", "user"), "content": item.get("content")})
    return out


def _render_call(name: str, arguments: str) -> str:
    from .prompt import render_tool_call
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except json.JSONDecodeError:
        args = {}
    return render_tool_call(name, args)


#: Tool types that are the provider's to run, not ours. Declaring one means the
#: client expects the SERVER to perform the search or execution.
_HOSTED = ("file_search", "web_search", "web_search_preview", "computer",
           "computer_use_preview", "code_interpreter", "image_generation",
           "local_shell", "mcp", "custom", "apply_patch", "shell")


def _hosted_tools(tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The hosted tool types present in a request, as refusal kwargs."""
    kinds = {t.get("type") for t in tools or []} & set(_HOSTED)
    return {f"tools[type={k}]": True for k in sorted(kinds)}


def _unsupported_choice(choice: Any) -> bool:
    """`tool_choice` beyond auto/none. We render tools into the prompt and cannot
    force or forbid a call, so anything stronger than a hint is unimplementable."""
    if choice is None:
        return None
    name = choice if isinstance(choice, str) else (choice or {}).get("type")
    return name not in ("auto", "none", None)


def _flatten_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Responses' ``{type, name, parameters}`` as the flat shape the template
    renders and ``_parse_tool_calls`` reads schemas from -- the same vocabulary
    the other two routes use, so a call means one thing across all three."""
    if not tools:
        return None
    return [{"name": t.get("name"), "description": t.get("description", ""),
             "input_schema": t.get("parameters") or t.get("input_schema") or {}}
            for t in tools]


def mount_responses(app: FastAPI, engine: Any, tokenizer: Tokenizer,
                    model_name: str) -> FastAPI:
    """Add POST /v1/responses to an existing app, sharing its engine."""

    def _thinking(req: ResponsesRequest) -> bool | None:
        # An explicit override wins; "none" effort switches it off; otherwise the
        # template's own default, which needs a tokenizer that HAS the tag.
        explicit = (req.chat_template_kwargs or {}).get("enable_thinking")
        if explicit is not None:
            return bool(explicit)
        if (req.reasoning or {}).get("effort") == "none":
            return False
        return len(tokenizer.encode("<think>")) == 1 or None

    def _run(req: ResponsesRequest) -> dict[str, Any]:
        refuse_unsupported(
            previous_response_id=req.previous_response_id,
            include=req.include,
            # "disabled" is our behaviour already, so only "auto" is a lie.
            truncation=req.truncation not in (None, "disabled"),
            store=req.store,
            tool_choice=_unsupported_choice(req.tool_choice),
            **_hosted_tools(req.tools))
        thinking = _thinking(req)
        tools = _flatten_tools(req.tools)
        prompt = render_prompt(_to_messages(req.input), req.instructions, tools,
                               thinking, (req.reasoning or {}).get("effort"))
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            raise ValueError("empty prompt after tokenization")
        params = sampling(tokenizer, thinking, req.max_output_tokens or 512,
                          temperature=req.temperature, top_p=req.top_p, stop=req.stop)
        rid = engine.submit(input_ids, params)
        deadline = time.monotonic() + 1800.0
        out = None
        while time.monotonic() < deadline:
            out = engine.take(rid)
            if out is not None:
                break
            time.sleep(0.02)
        if out is None:
            raise TimeoutError(f"request {rid} did not finish within 1800.0s")
        reasoning, text = split_think(tokenizer.decode(out), bool(thinking))
        stopped = engine.stop_text(rid)
        text, calls = _parse_tool_calls(cut_at_stop(text, stopped), tools)
        return _body(rid, req, model_name, reasoning, text, calls,
                     len(input_ids), len(out), params.max_new_tokens, stopped=stopped)

    def _body(rid: int, req: ResponsesRequest, model: str, reasoning: str, text: str,
              calls: list, n_in: int, n_out: int, max_new: int,
              output: list | None = None, stopped: str | None = None) -> dict[str, Any]:
        """``Response``, with the fields the SDK's model requires.

        ``parallel_tool_calls`` and ``tool_choice`` are declared required and
        appear in no doc example; the SDK will not complain if they are missing
        (it uses ``construct``), so they are asserted in the tests instead.
        ``status`` is "incomplete" with a reason when the cap cut it, which is how
        a client tells a finished answer from a truncated one.
        """
        # A stop sequence is a COMPLETE response: the cap is not what ended it, so
        # a client must not see incomplete/max_output_tokens for a delimiter it asked for.
        cut = n_out >= max_new and not calls and not stopped
        return {
            "id": f"resp_{rid}",
            "object": "response",
            "created_at": float(int(time.time())),
            "model": req.model or model,
            "status": "incomplete" if cut else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if cut else None,
            "error": None,
            "output": _output_items(rid, reasoning, text, calls) if output is None else output,
            "parallel_tool_calls": bool(req.parallel_tool_calls),
            "tool_choice": req.tool_choice or "auto",
            "tools": req.tools or [],
            "instructions": req.instructions,
            "metadata": req.metadata or {},
            "temperature": req.temperature,
            "top_p": req.top_p,
            "usage": {"input_tokens": n_in, "output_tokens": n_out,
                      "total_tokens": n_in + n_out,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        }

    def _output_items(rid: int, reasoning: str, text: str, calls: list) -> list[dict]:
        """Reasoning first, then the message, then one item per call.

        Order is the contract: `output_text` concatenates the message items, and a
        reasoning item placed after the message would read as a second answer.
        """
        items: list[dict[str, Any]] = []
        if reasoning:
            items.append({"id": f"rs_{rid}", "type": "reasoning",
                          # summary is required and must be a list; we do not
                          # summarise, so the text goes in `content` and summary
                          # stays empty rather than being faked.
                          "summary": [],
                          "content": [{"type": "reasoning_text", "text": reasoning}],
                          "status": "completed"})
        if text or not calls:
            items.append({"id": f"msg_{rid}", "type": "message", "role": "assistant",
                          "status": "completed",
                          "content": [{"type": "output_text", "text": text,
                                       "annotations": []}]})
        items += [{"id": f"fc_{rid}_{i}", "type": "function_call",
                   "call_id": f"call_{rid}_{i}", "name": n,
                   "arguments": json.dumps(a, ensure_ascii=False),
                   "status": "completed"}
                  for i, (n, a) in enumerate(calls)]
        return items

    @app.post("/v1/responses")
    async def responses(req: ResponsesRequest):
        try:
            body = _run(req)
        except ValueError as exc:
            return JSONResponse(status_code=400,
                                content={"error": {"message": str(exc),
                                                   "type": "invalid_request_error"}})
        except (TimeoutError, RuntimeError) as exc:
            return JSONResponse(status_code=503,
                                content={"error": {"message": str(exc),
                                                   "type": "api_error"}})
        if not req.stream:
            return JSONResponse(content=body)

        def sse():
            # Every event carries a sequence_number the SDK's model requires, so
            # it is a running counter rather than an afterthought.
            n = 0

            def ev(payload: dict[str, Any]) -> str:
                nonlocal n
                payload["sequence_number"] = n
                n += 1
                return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"

            empty = dict(body, output=[], status="in_progress",
                         incomplete_details=None)
            yield ev({"type": "response.created", "response": empty})
            yield ev({"type": "response.in_progress", "response": empty})
            for i, item in enumerate(body["output"]):
                yield ev({"type": "response.output_item.added", "output_index": i,
                          "item": dict(item, status="in_progress",
                                       **({"content": []} if "content" in item else {}))})
                if item["type"] == "message":
                    part = {"type": "output_text", "text": "", "annotations": []}
                    yield ev({"type": "response.content_part.added", "item_id": item["id"],
                              "output_index": i, "content_index": 0, "part": part})
                    txt = item["content"][0]["text"]
                    yield ev({"type": "response.output_text.delta", "item_id": item["id"],
                              "output_index": i, "content_index": 0, "delta": txt,
                              "logprobs": []})
                    yield ev({"type": "response.output_text.done", "item_id": item["id"],
                              "output_index": i, "content_index": 0, "text": txt,
                              "logprobs": []})
                    yield ev({"type": "response.content_part.done", "item_id": item["id"],
                              "output_index": i, "content_index": 0,
                              "part": item["content"][0]})
                elif item["type"] == "function_call":
                    yield ev({"type": "response.function_call_arguments.delta",
                              "item_id": item["id"], "output_index": i,
                              "delta": item["arguments"]})
                    yield ev({"type": "response.function_call_arguments.done",
                              "item_id": item["id"], "output_index": i,
                              "arguments": item["arguments"]})
                yield ev({"type": "response.output_item.done", "output_index": i,
                          "item": item})
            yield ev({"type": "response.completed", "response": body})

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app
