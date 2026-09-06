"""End-to-end through the official SDKs, not hand-written JSON.

The gate #155 asked for: fixtures produced by our server, parsed by the client
that actually matters. Hand-rolled asserts on a dict cannot catch a field the
SDK's own model rejects, a stream event named wrongly, or a shape that differs
between the streaming and non-streaming path -- all three are real defects this
file found on main.

One uvicorn per module over ``_ScriptedEngine``, so there are no weights and no
GPU: the replies are canned, and what is under test is the HTTP surface.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest
from test_server import _ByteTokenizer, _ScriptedEngine

from tilerl.server import create_app

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")
uvicorn = pytest.importorskip("uvicorn")

#: The 27B template opens <think> in the prompt, so a real reply carries only the
#: closer -- every canned reply here has that shape (errors/2026-09-06-the-reply-
#: carried-the-reasoning-and-a-bare-closer.md).
REASON = "weighing it up"
REPLY = "The answer is 4."
PLAIN = f"{REASON}\n</think>\n\n{REPLY}"

TOOL_CALL = ("</think>\n\nI will run it.\n<tool_call>\n<function=Bash>\n"
             "<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>")


class _PromptKeyedEngine(_ScriptedEngine):
    """``_ScriptedEngine`` serves replies in SUBMIT order, which makes every
    assertion depend on how many requests the arms before it happened to make --
    a reordering silently hands a tool arm a plain reply. Key on the prompt
    instead, so each arm's reply is a function of what it asked for.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer, [])
        self._tok = tokenizer
        #: every prompt this engine was handed, so a test can assert what the
        #: route RENDERED and not only what it parsed back
        self.prompts: list[str] = []

    def submit(self, input_ids, params=None) -> int:
        prompt = self._tok.decode(list(input_ids))
        self.prompts.append(prompt)
        # A tool round trip replays the original "run ls" turn, so keying on the
        # request alone answers the FOLLOW-UP with another tool call and the
        # assertion reads as a route defect. The tool response in the prompt is
        # what distinguishes turn 2.
        wants_call = "run ls" in prompt and "<tool_response>" not in prompt
        self._replies = [TOOL_CALL if wants_call else PLAIN]
        return super().submit(input_ids, params)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def engine_and_url(tmp_path_factory):
    import os

    # /v1/messages appends a JSONL row per request; keep it out of the repo.
    os.environ["TILERL_MESSAGES_RECORD"] = str(tmp_path_factory.mktemp("rec") / "r.jsonl")
    port = _free_port()
    tok = _ByteTokenizer()
    eng = _PromptKeyedEngine(tok)
    app = create_app(eng, tok, model_name="tilerl")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(400):
        if server.started:
            break
        time.sleep(0.05)
    else:
        pytest.fail("uvicorn did not start")
    yield eng, f"http://127.0.0.1:{port}"
    server.should_exit = True


#: The chat route infers `enable_thinking` from whether the tokenizer HAS a
#: <think> token; ByteTokenizer spells it as 8 raw bytes, so the default here is
#: "bare turn" and no reasoning block is opened. The 27B's tokenizer has the
#: token, so asking explicitly is what makes this harness match production --
#: without it every reasoning assertion below passes or fails for the wrong reason.
THINKING_ON = {"chat_template_kwargs": {"enable_thinking": True}}


@pytest.fixture
def base_url(engine_and_url):
    return engine_and_url[1]


@pytest.fixture
def engine(engine_and_url):
    return engine_and_url[0]


@pytest.fixture
def oa(base_url):
    return openai.OpenAI(base_url=f"{base_url}/v1", api_key="x", max_retries=0)


@pytest.fixture
def an(base_url):
    return anthropic.Anthropic(base_url=base_url, api_key="x", max_retries=0)


# --- OpenAI chat completions ------------------------------------------------


def test_chat_non_stream_reconstructs_the_reply(oa):
    m = oa.chat.completions.create(model="tilerl", extra_body=THINKING_ON,
                                   messages=[{"role": "user", "content": "hi"}])
    assert m.choices[0].message.content == REPLY
    assert m.choices[0].finish_reason == "stop"
    assert m.usage.total_tokens == m.usage.prompt_tokens + m.usage.completion_tokens


def test_chat_stream_reconstructs_the_same_text(oa):
    """The SDK's parser accepts every frame, and the deltas rebuild the reply.

    Same assertion as the non-stream arm on purpose: the two paths building
    different strings is the defect class here, not a hypothetical.
    """
    got = "".join(c.choices[0].delta.content or ""
                  for c in oa.chat.completions.create(
                      model="tilerl", messages=[{"role": "user", "content": "hi"}],
                      stream=True, extra_body=THINKING_ON) if c.choices)
    assert got == REPLY


def test_chat_reasoning_is_the_same_field_on_both_paths(oa):
    """#159 put the reasoning in `reasoning_content` on the STREAM only.

    A client that switches `stream` gets a different shape for the same request:
    streaming yields the reasoning, non-streaming drops it on the floor. The
    vLLM extension is defined for both.
    """
    kw = {"model": "tilerl", "messages": [{"role": "user", "content": "hi"}],
          "extra_body": THINKING_ON}
    streamed = "".join(
        getattr(c.choices[0].delta, "reasoning_content", None) or ""
        for c in oa.chat.completions.create(**kw, stream=True) if c.choices)
    once = oa.chat.completions.create(**kw).choices[0].message
    # rstrip: the reasoning ends at the newline before </think>, and whether that
    # newline is inside the block is not a field-shape claim. The point is that
    # both paths carry the SAME reasoning, in the same field.
    assert streamed.rstrip("\n") == REASON, "stream lost the reasoning"
    assert (getattr(once, "reasoning_content", None) or "").rstrip("\n") == REASON, \
        "non-stream dropped the reasoning the stream returns"


def test_chat_stream_usage_is_opt_in_and_final(oa):
    chunks = list(oa.chat.completions.create(
        model="tilerl", messages=[{"role": "user", "content": "hi"}], stream=True,
        stream_options={"include_usage": True}, extra_body=THINKING_ON))
    assert chunks[-1].usage is not None and not chunks[-1].choices
    assert chunks[-1].usage.completion_tokens > 0


def test_chat_tools_come_back_as_tool_calls(oa):
    """The XML the template emits must reach the client as a structured call.

    On main `tools` is not even a field on the request model, so the raw
    `<tool_call>` markup lands in `message.content` and `tool_calls` is None --
    every OpenAI-speaking agent breaks. `messages.py` already parses this shape.
    """
    m = oa.chat.completions.create(
        model="tilerl", messages=[{"role": "user", "content": "run ls"}],
        extra_body=THINKING_ON,
        tools=[{"type": "function", "function": {
            "name": "Bash", "description": "run a command",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}}}}}])
    choice = m.choices[0]
    assert "<tool_call>" not in (choice.message.content or ""), \
        "raw tool XML leaked into content"
    assert choice.message.tool_calls, "tools were dropped: no tool_calls returned"
    call = choice.message.tool_calls[0]
    assert call.function.name == "Bash"
    assert call.function.arguments == '{"command": "ls"}'
    assert choice.finish_reason == "tool_calls"


def test_chat_tools_reach_the_prompt(oa, engine):
    """The schemas must be RENDERED, not only parsed back out.

    Deleting the render call leaves every other tool assertion green, because a
    canned engine emits the same call whether or not the prompt defined the tool
    -- so parsing the reply cannot tell you the model was told what Bash is.
    """
    before = len(engine.prompts)
    oa.chat.completions.create(
        model="tilerl", messages=[{"role": "user", "content": "run ls"}],
        tools=[{"type": "function", "function": {
            "name": "Bash", "description": "run a command",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}}}}}])
    prompt = engine.prompts[before]
    assert "<tools>" in prompt and '"name": "Bash"' in prompt
    assert "run a command" in prompt


def test_chat_rejects_a_bad_field_with_openais_error_envelope(oa):
    """A pydantic failure must not escape as FastAPI's `detail` list.

    The class is the load-bearing assertion: FastAPI's default 422 makes the SDK
    raise UnprocessableEntityError, so a client catching BadRequestError does not
    catch it at all. The route's own 400/503 paths already emit the right
    envelope -- validation ran before them.

    `.body` is the *unwrapped* `error` object, not the whole document: the SDK
    reads `error` off the response itself, which is exactly why the wrapper has
    to be there. The offending field is named in the message so a caller can
    tell which one was rejected.
    """
    with pytest.raises(openai.BadRequestError) as exc:
        oa.chat.completions.create(model="tilerl", max_completion_tokens=0,
                                   messages=[{"role": "user", "content": "hi"}])
    assert exc.value.status_code == 400
    assert exc.value.body["type"] == "invalid_request_error"
    assert "max_completion_tokens" in exc.value.body["message"]


def test_models_list(oa):
    assert [m.id for m in oa.models.list().data] == ["tilerl"]


# --- Anthropic messages ----------------------------------------------------


def _text_of(message):
    """The text block, selected by type. Indexing content[0] assumes the reply
    starts with text, which stops being true the moment a thinking block is
    prepended -- and then reads as a route defect."""
    return "".join(b.text for b in message.content if b.type == "text")


def test_messages_non_stream(an):
    m = an.messages.create(model="tilerl", max_tokens=64,
                           messages=[{"role": "user", "content": "hi"}])
    assert _text_of(m) == REPLY
    assert m.stop_reason == "end_turn"
    assert m.usage.output_tokens > 0


def test_messages_stream_events_and_text(an):
    with an.messages.stream(model="tilerl", max_tokens=64,
                            messages=[{"role": "user", "content": "hi"}]) as s:
        names = [e.type for e in s]
        text = s.get_final_text()
    assert text == REPLY
    # The SDK tolerates a missing ping, so assert the ordering that matters.
    for want in ("message_start", "content_block_start", "content_block_delta",
                 "content_block_stop", "message_delta", "message_stop"):
        assert want in names, f"{want} missing from {names}"
    assert names.index("message_start") == 0
    assert names[-1] == "message_stop"


def test_messages_thinking_is_a_thinking_block(an):
    """Anthropic's native shape for reasoning is a `thinking` content block.

    On main the reasoning is stripped and thrown away, so a client that asked
    for thinking gets text only and cannot show or replay it.
    """
    m = an.messages.create(
        model="tilerl", max_tokens=64,
        thinking={"type": "enabled", "budget_tokens": 32},
        messages=[{"role": "user", "content": "2+2?"}])
    kinds = [b.type for b in m.content]
    assert "thinking" in kinds, f"reasoning was dropped, got {kinds}"
    thinking = next(b for b in m.content if b.type == "thinking")
    assert thinking.thinking.rstrip("\n") == REASON  # see the chat arm on the newline
    assert [b.text for b in m.content if b.type == "text"] == [REPLY]


def test_messages_tool_use_round_trip(an):
    m = an.messages.create(
        model="tilerl", max_tokens=64,
        tools=[{"name": "Bash", "description": "run a command",
                "input_schema": {"type": "object",
                                 "properties": {"command": {"type": "string"}}}}],
        messages=[{"role": "user", "content": "run ls"}])
    assert m.stop_reason == "tool_use"
    use = next(b for b in m.content if b.type == "tool_use")
    assert use.name == "Bash" and use.input == {"command": "ls"}
    # The follow-up turn: a tool_result the SDK serialises must be accepted.
    follow = an.messages.create(
        model="tilerl", max_tokens=64,
        messages=[{"role": "user", "content": "run ls"},
                  {"role": "assistant", "content": [b.model_dump() for b in m.content]},
                  {"role": "user", "content": [{"type": "tool_result",
                                                "tool_use_id": use.id,
                                                "content": "a.txt"}]}])
    assert _text_of(follow) == REPLY


def test_messages_rejects_a_bad_field_with_anthropics_error_envelope(an):
    with pytest.raises(anthropic.BadRequestError) as exc:
        an.messages.create(model="tilerl", max_tokens=0,
                           messages=[{"role": "user", "content": "hi"}])
    assert exc.value.body["error"]["message"]
    assert exc.value.body["type"] == "error"


# --- OpenAI Responses ------------------------------------------------------


def test_responses_non_stream(oa):
    r = oa.responses.create(model="tilerl", input="hi", extra_body=THINKING_ON)
    assert r.output_text == REPLY
    assert r.status == "completed"
    assert [i.type for i in r.output] == ["reasoning", "message"]
    assert r.usage.total_tokens == r.usage.input_tokens + r.usage.output_tokens


def test_responses_carries_the_fields_the_model_declares_required(oa, base_url):
    """`Response` declares parallel_tool_calls / tool_choice / tools required.

    Asserted by name because the SDK will NOT catch their absence: measured,
    `Response.model_validate` rejects a body without parallel_tool_calls, but the
    client builds replies with `construct`, so a missing required field arrives as
    None and every other assertion still passes. Read off the raw JSON, since the
    parsed object cannot distinguish "absent" from "null".
    """
    import httpx

    raw = httpx.post(f"{base_url}/v1/responses", timeout=30,
                     json={"model": "tilerl", "input": "hi"}).json()
    for field in ("parallel_tool_calls", "tool_choice", "tools", "object",
                  "created_at", "usage"):
        assert field in raw, f"{field} missing from the response body"
    assert raw["object"] == "response"


def test_responses_reasoning_is_its_own_item(oa):
    r = oa.responses.create(model="tilerl", input="hi", extra_body=THINKING_ON)
    item = next(i for i in r.output if i.type == "reasoning")
    assert [c.text.rstrip("\n") for c in item.content] == [REASON]
    # output_text is the message items only, so the reasoning must not leak in.
    assert REASON not in r.output_text


def test_responses_stream_events_and_order(oa):
    names, text = [], ""
    for ev in oa.responses.create(model="tilerl", input="hi", stream=True, extra_body=THINKING_ON):
        names.append(ev.type)
        if ev.type == "response.output_text.delta":
            text += ev.delta
    assert text == REPLY
    assert names[0] == "response.created"
    assert names[-1] == "response.completed"
    for want in ("response.in_progress", "response.output_item.added",
                 "response.content_part.added", "response.output_text.done",
                 "response.content_part.done", "response.output_item.done"):
        assert want in names, f"{want} missing from {names}"


def test_responses_tool_call_and_replay(oa, engine):
    """A call out, then its output back in -- the shape an agent loop sends.

    The second request replays the function_call and function_call_output items;
    they must reach the same template the first turn used, or a loop's history
    silently degrades to a bare prompt.
    """
    tools = [{"type": "function", "name": "Bash", "description": "run a command",
              "parameters": {"type": "object",
                             "properties": {"command": {"type": "string"}}}}]
    first = oa.responses.create(model="tilerl", input="run ls", tools=tools,
                               extra_body=THINKING_ON)
    call = next(i for i in first.output if i.type == "function_call")
    assert call.name == "Bash" and json.loads(call.arguments) == {"command": "ls"}
    assert first.status == "completed"

    before = len(engine.prompts)
    second = oa.responses.create(model="tilerl", tools=tools, extra_body=THINKING_ON, input=[
        {"type": "message", "role": "user", "content": "run ls"},
        {"type": "function_call", "call_id": call.call_id, "name": call.name,
         "arguments": call.arguments},
        {"type": "function_call_output", "call_id": call.call_id, "output": "a.txt"},
    ])
    prompt = engine.prompts[before]
    assert "<tool_response>" in prompt and "a.txt" in prompt, prompt
    assert second.output_text == REPLY


def test_responses_max_output_tokens_reports_incomplete(oa):
    r = oa.responses.create(model="tilerl", input="hi", max_output_tokens=4,
                            extra_body=THINKING_ON)
    assert r.status == "incomplete"
    assert r.incomplete_details.reason == "max_output_tokens"


def test_messages_replaying_a_thinking_block_drops_the_reasoning(an, engine):
    """Claude Code sends the thinking block back in the next assistant turn.

    `blocks_to_text` keeps only kinds it knows, so an unknown one vanishes -- and
    that is the wanted behaviour here: the template re-opens <think> only after
    the last real user query, so replaying old reasoning is off-distribution.
    Untested until now, which is the same shape as a silent format regression.
    """
    tools = [{"name": "Bash", "description": "run a command",
              "input_schema": {"type": "object",
                               "properties": {"command": {"type": "string"}}}}]
    first = an.messages.create(model="tilerl", max_tokens=64, tools=tools,
                               messages=[{"role": "user", "content": "run ls"}])
    use = next(b for b in first.content if b.type == "tool_use")

    before = len(engine.prompts)
    second = an.messages.create(
        model="tilerl", max_tokens=64, tools=tools,
        thinking={"type": "enabled", "budget_tokens": 32},
        messages=[
            {"role": "user", "content": "run ls"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should list them.", "signature": ""},
                {"type": "tool_use", "id": use.id, "name": use.name, "input": use.input},
            ]},
            {"role": "user", "content": [{"type": "tool_result",
                                          "tool_use_id": use.id, "content": "a.txt"}]},
        ])
    prompt = engine.prompts[before]
    assert "I should list them." not in prompt, "replayed reasoning reached the prompt"
    # The turn is not dropped wholesale: its tool_use and the result both render.
    assert "<tool_call>" in prompt and "<tool_response>" in prompt and "a.txt" in prompt
    assert _text_of(second) == REPLY


# --- accepted-and-ignored is a format lie: each field must 400 ---------------


@pytest.mark.parametrize("body,field", [
    ({"store": True}, "store"),
    ({"previous_response_id": "resp_1"}, "previous_response_id"),
    ({"include": ["reasoning.encrypted_content"]}, "include"),
    ({"truncation": "auto"}, "truncation"),
    ({"tool_choice": "required"}, "tool_choice"),
    ({"tool_choice": {"type": "function", "name": "Bash"}}, "tool_choice"),
    ({"tools": [{"type": "web_search"}]}, "web_search"),
    ({"tools": [{"type": "code_interpreter", "container": {}}]}, "code_interpreter"),
    ({"tools": [{"type": "file_search", "vector_store_ids": ["x"]}]}, "file_search"),
])
def test_responses_refuses_a_field_it_cannot_honour(oa, body, field):
    """A field taken and ignored is a lie the client pays for one turn later.

    `store=true` means the next request may send only previous_response_id; a
    hosted tool type means the client expects US to run the search. Answering as
    if we had is worse than refusing, so each returns 400 naming the field.
    """
    with pytest.raises(openai.BadRequestError) as exc:
        oa.responses.create(model="tilerl", input="hi", **body)
    assert field in exc.value.body["message"], exc.value.body
    assert exc.value.body["type"] == "invalid_request_error"


def test_responses_still_accepts_what_it_does_honour(oa):
    """The refusal must not be a blanket one.

    truncation="disabled" IS our behaviour, tool_choice auto/none is a hint we
    can respect by rendering or omitting the tools, and metadata is echoed --
    refusing these would break a compliant client for no gain.
    """
    r = oa.responses.create(model="tilerl", input="hi", truncation="disabled",
                            tool_choice="auto", metadata={"k": "v"},
                            parallel_tool_calls=False, extra_body=THINKING_ON)
    assert r.output_text == REPLY
    assert r.metadata == {"k": "v"}


def test_chat_refuses_a_tool_choice_it_cannot_force(oa):
    with pytest.raises(openai.BadRequestError) as exc:
        oa.chat.completions.create(
            model="tilerl", messages=[{"role": "user", "content": "hi"}],
            tool_choice="required",
            tools=[{"type": "function", "function": {"name": "Bash"}}])
    assert "tool_choice" in exc.value.body["message"]


def test_messages_refuses_stop_sequences_and_context_management(an):
    """stop_sequences was accepted and never applied, so stop_reason could never
    be "stop_sequence" -- a client waiting for one waited forever."""
    with pytest.raises(anthropic.BadRequestError) as exc:
        an.messages.create(model="tilerl", max_tokens=64,
                           messages=[{"role": "user", "content": "hi"}],
                           stop_sequences=["\n\n"])
    assert "stop_sequences" in exc.value.body["error"]["message"]
    # An edit we do NOT perform is refused and named.
    with pytest.raises(anthropic.BadRequestError) as exc2:
        an.messages.create(model="tilerl", max_tokens=64,
                           messages=[{"role": "user", "content": "hi"}],
                           extra_body={"context_management": {"edits": [
                               {"type": "clear_tool_uses_20250919"}]}})
    assert "clear_tool_uses" in exc2.value.body["error"]["message"]


def test_messages_accepts_the_context_edit_claude_code_always_sends(an):
    """Claude Code sends context_management on EVERY request, and refusing the
    whole field broke the live rollout gate. Measured by capturing the CLI's
    body: {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]} --
    which asks for behaviour `blocks_to_text` already has, since it drops
    thinking blocks when replaying history. Satisfied by construction, so
    accepting it is not a lie.
    """
    m = an.messages.create(model="tilerl", max_tokens=64,
                           messages=[{"role": "user", "content": "hi"}],
                           extra_body={"context_management": {"edits": [
                               {"type": "clear_thinking_20251015", "keep": "all"}]}})
    assert _text_of(m) == REPLY


def test_chat_refuses_stop_rather_than_dropping_it(oa):
    """`stop` is a core OpenAI field we do not implement.

    It was not even declared, so pydantic discarded it and the client got a 200
    with no sign the stop never applied. Refused with 400 until the engine grows
    multi-token stop ids (errors/2026-09-06-stop-sequences-accepted-and-never-
    applied.md); tracked in OPEN.md rather than treated as settled.
    """
    with pytest.raises(openai.BadRequestError) as exc:
        oa.chat.completions.create(model="tilerl", stop=["\n\n"],
                                   messages=[{"role": "user", "content": "hi"}])
    assert "stop" in exc.value.body["message"]


def test_responses_refuses_stop_rather_than_dropping_it(oa):
    with pytest.raises(openai.BadRequestError) as exc:
        oa.responses.create(model="tilerl", input="hi", extra_body={"stop": ["\n\n"]})
    assert "stop" in exc.value.body["message"]
