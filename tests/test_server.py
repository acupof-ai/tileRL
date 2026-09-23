"""Server gates for tilerl: /health, /v1/models, non-stream completion, SSE stream.

Uses FastAPI's TestClient against a tiny-engine app. A deterministic
byte-level tokenizer (vocab 320, matching tiny()) stands in at the IO
boundary — the gate is HTTP/SSE behaviour, not tokenization fidelity.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
import time

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
from fastapi.testclient import TestClient
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine, build_serving_engine
from tilerl.config import tiny
from tilerl.engine import Engine, SamplingParams
from tilerl.messages import render_tool_call
from tilerl.model import build_random
from tilerl.server import _chat_chunk, create_app, get_tokenizer

# ---------------------------------------------------------------------------
# helpers


class _ByteTokenizer:
    """Deterministic byte-level tokenizer, vocab 320 (test double).

    ids 0-2 reserved (pad/bos/eos), bytes map to 3-258 — every id stays below
    tiny()'s vocab of 320.
    """

    def encode(self, text: str) -> list[int]:
        return [1] + [b + 3 for b in text.encode("utf-8")]

    def decode(self, ids) -> str:
        return bytes(i - 3 for i in ids if 3 <= i < 259).decode("utf-8", errors="replace")


class _TextTokenizer(_ByteTokenizer):
    """Byte-level decode of a FIXED text, indexed by how many ids arrived.

    Exists because the streaming test's premise cannot depend on model weights.
    `_ByteTokenizer` decodes whatever bytes the random tiny model samples, and on
    macos-14 CI those were mostly UTF-8 continuation bytes: every prefix ended in
    U+FFFD, `rstrip` held the visible text flat, and the whole reply arrived in one
    delta. Green locally, red there, for a reason that has nothing to do with the
    loop under test.

    Decoding `pattern[:len(ids)]` keeps every property the test needs -- a prefix
    that grows one byte at a time, multi-byte characters that a cut lands inside --
    and makes all of them independent of what was sampled.
    """

    #: Two properties the defects need in order to bite, so the negative controls
    #: discriminate. Multi-byte characters ('é' 2 bytes, '→' 3) so a prefix can cut
    #: one -- and a raw 0xFF, which is not valid UTF-8 in any position and so decodes
    #: to an INTERIOR U+FFFD in every longer prefix. Without that byte, `split("�")[0]`
    #: and `rstrip("�")` agree on all but 3 of 51 prefixes and both mutations pass.
    #: A bytes literal, not str.encode: every codec turns \xff back into valid UTF-8.
    PATTERN = b"the caf\xc3\xa9 \xff menu \xe2\x86\x92 three courses"

    def decode(self, ids) -> str:
        n = sum(1 for i in ids if 3 <= i < 259)
        return self.PATTERN[:n].decode("utf-8", errors="replace")


def _text_blocks(body: dict) -> str:
    """The text blocks of a /v1/messages reply, joined.

    Selected by type, never by index: a reasoning block now precedes the text
    when the prompt opened <think>, and `content[0]["text"]` raises a KeyError
    that reads like a routing defect.
    """
    return "".join(b["text"] for b in body["content"] if b["type"] == "text")


def _build_engine(seed: int) -> Engine:
    cfg = tiny()
    model = build_random(cfg, seed=seed)
    backend = get_backend()
    # 4096, not 512: the tool block is the checkpoint's template verbatim (817
    # bytes of instructions alone), and ByteTokenizer is one token per byte, so
    # a one-tool request is ~1.1k tokens. Shrinking the prompt to fit would be
    # measuring a format the 27B never sees.
    return build_engine(
        cfg, model, backend, num_blocks=256, num_slots=4, max_batch=4, max_total_tokens=4096,
        sparse_k=0)  # dense server feature suite (prefix cache, health, dram/state bytes)


@pytest.fixture(scope="module")
def client():
    engine = _build_engine(seed=42)
    engine.run()  # server handlers only submit/poll; the loop must run
    app = create_app(engine, _ByteTokenizer())
    # Bound the timeout: if the app does not drive the engine loop, a request
    # fails fast here instead of hanging the suite.
    with TestClient(app) as test_client:
        yield test_client
    engine.shutdown()


@pytest.fixture(scope="module")
def model_id(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert isinstance(data, list) and data, "no models served"
    assert "id" in data[0], f"model entry missing id: {data[0]!r}"
    return data[0]["id"]


# ---------------------------------------------------------------------------
# tests


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json(), dict)
    assert resp.json()["status"] == "ok"


def test_health_says_degraded_when_the_engine_raises():
    """`status` was the literal "ok", so a broken engine and a healthy one gave the same
    body — a health endpoint that cannot report ill health. No consumer reads the field
    today (all three script readers take `stats`, and each fails on None), which is why
    this was found by reading rather than by an outage.

    Asserts the two states DIFFER, plus that the healthy one still says ok, so a fix
    that marks everything degraded does not pass.
    """

    class _StatsRaises:
        def __init__(self, inner):
            self._inner = inner

        def stats(self):
            raise RuntimeError("engine is wedged")

        def __getattr__(self, name):
            return getattr(self._inner, name)

    engine = _build_engine(seed=34)
    engine.run()
    try:
        with TestClient(create_app(engine, _ByteTokenizer())) as c:
            good_resp = c.get("/health")
            good = good_resp.json()
        with TestClient(create_app(_StatsRaises(engine), _ByteTokenizer())) as c:
            bad_resp = c.get("/health")
            bad = bad_resp.json()
    finally:
        engine.shutdown()

    assert good_resp.status_code == 200
    assert good["status"] == "ok" and good["stats"], good
    assert bad_resp.status_code == 503, "a stats-raising engine must answer 503"
    assert bad["status"] != "ok", f"a raising engine still reports {bad['status']!r}"
    assert bad["stats"] is None and "RuntimeError" in bad.get("error", ""), bad


def test_models(client, model_id):
    assert isinstance(model_id, str) and model_id


def test_streaming_tool_call_is_structured_at_the_terminal_frame(tmp_path):
    """Finding 15: the stream used to carry raw <tool_call> XML in content and
    finish "length" while the non-stream route structured the same reply. The
    XML is held across every content frame, then emitted as tool_calls deltas
    byte-identical to the non-stream message, with finish tool_calls."""
    tok = _ByteTokenizer()
    from tilerl.prompt import render_tool_call
    reply = "I will run it.\n" + render_tool_call("Bash", {"command": "ls"})
    app = create_app(_ScriptedEngine(tok, [reply]), tok)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "run ls"}],
            "tools": [{"type": "function", "function": {
                "name": "Bash", "description": "run",
                "parameters": {"type": "object",
                               "properties": {"command": {"type": "string"}}}}}],
            "stream": True, "max_tokens": 256,
        })
    assert r.status_code == 200, r.text
    frames = [json.loads(ln[6:]) for ln in r.text.splitlines()
              if ln.startswith("data: ") and ln[6:] != "[DONE]"]
    content = "".join(
        (f["choices"][0].get("delta", {}).get("content") or "") for f in frames)
    assert "<tool_call>" not in content and "Bash" not in content, content
    assert content == "I will run it.", repr(content)
    tc_frames = [f for f in frames
                 if f["choices"][0].get("delta", {}).get("tool_calls")]
    assert len(tc_frames) == 1, [f["choices"][0]["delta"] for f in frames]
    tc = tc_frames[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tc == {"index": 0, "id": "call_1_0", "type": "function",
                 "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}, tc
    terminal = frames[-1]["choices"][0]
    assert terminal["delta"] == {} and terminal["finish_reason"] == "tool_calls"
    # The non-stream reply for the SAME canned text must equal the stream frame.
    app2 = create_app(_ScriptedEngine(tok, [reply]), tok)
    with TestClient(app2) as c:
        body = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "run ls"}],
            "tools": [{"type": "function", "function": {
                "name": "Bash", "description": "run",
                "parameters": {"type": "object",
                               "properties": {"command": {"type": "string"}}}}}],
            "max_tokens": 256,
        }).json()
    nonstream = body["choices"][0]["message"]["tool_calls"][0]
    assert tc["id"] == nonstream["id"]
    assert tc["function"] == nonstream["function"]
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_none_suppresses_render_and_output(tmp_path):
    """Finding 13 on chat: choice none must remove the tools block from the
    prompt AND discard calls parsed from a reply that calls anyway."""
    tok = _ByteTokenizer()
    seen = {}
    from tilerl.prompt import render_tool_call
    reply = "I will run it.\n" + render_tool_call("Bash", {"command": "ls"})

    class _Cap(_ScriptedEngine):
        def submit(self, input_ids, params=None):
            seen["prompt"] = self._tok.decode(list(input_ids))
            return super().submit(input_ids, params)

    app = create_app(_Cap(tok, [reply]), tok)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "run ls"}],
            "tools": [{"type": "function", "function": {"name": "Bash",
                       "description": "run",
                       "parameters": {"type": "object",
                                      "properties": {"command": {"type": "string"}}}}}],
            "tool_choice": "none", "max_tokens": 256,
        })
    assert "<tools>" not in seen["prompt"] and "Bash" not in seen["prompt"]
    ch = r.json()["choices"][0]
    assert ch["finish_reason"] == "stop"
    assert ch["message"]["tool_calls"] is None
    # The shared parser strips the recognized XML; suppressing the calls
    # leaves the prose before the call as ordinary content (the stream keeps
    # the separating newline, the non-stream reply trims it).
    assert "<tool_call>" not in ch["message"]["content"]
    assert ch["message"]["content"] == "I will run it."


def test_tool_choice_none_streaming_never_emits_call_xml(tmp_path):
    """CHANGE-REQ regression: with choice none the structured call is dropped,
    but the opener hold used to be gated on the same flag, so the raw XML
    still rode the content deltas (finish stop, content contained the whole
    <tool_call> block)."""
    tok = _ByteTokenizer()
    from tilerl.prompt import render_tool_call
    reply = "I will run it.\n" + render_tool_call("Bash", {"command": "ls"})
    app = create_app(_ScriptedEngine(tok, [reply]), tok)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "run ls"}],
            "tools": [{"type": "function", "function": {
                "name": "Bash", "description": "run",
                "parameters": {"type": "object",
                               "properties": {"command": {"type": "string"}}}}}],
            "tool_choice": "none", "stream": True, "max_tokens": 256,
        })
    assert r.status_code == 200, r.text
    frames = [json.loads(ln[6:]) for ln in r.text.splitlines()
              if ln.startswith("data: ") and ln[6:] != "[DONE]"]
    content = "".join(
        f["choices"][0].get("delta", {}).get("content") or "" for f in frames)
    assert "<tool_call>" not in content
    assert "<function=Bash>" not in content
    assert "ls" not in content
    assert content == "I will run it.", repr(content)
    assert not [f for f in frames
                if f["choices"][0].get("delta", {}).get("tool_calls")]
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_refuses_hosted_tools(tmp_path):
    """Finding 16: chat used to accept web_search-style hosted tools with a
    null name while responses already refused them."""
    tok = _ByteTokenizer()
    app = create_app(_ScriptedEngine(tok, ["ok"]), tok)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search"}],
        })
    assert r.status_code == 400, r.text
    assert "web_search" in r.json()["error"]["message"]


@pytest.mark.parametrize(("path", "body_extra"), [
    ("/v1/messages", {"max_tokens": 256,
                       "messages": [{"role": "user", "content": "run ls"}],
                       "tools": [{"name": "Bash", "description": "run",
                                  "input_schema": {"properties": {
                                      "command": {"type": "string"}}}}]}),
    ("/v1/responses", {"max_output_tokens": 256, "input": [{"type": "message", "role": "user",
                                 "content": [{"type": "input_text",
                                              "text": "run ls"}]}],
                       "tools": [{"type": "function", "name": "Bash",
                                  "description": "run",
                                  "parameters": {"properties": {
                                      "command": {"type": "string"}}}}]}),
])
def test_tool_choice_none_suppresses_each_route(tmp_path, monkeypatch, path, body_extra):
    """Refinement 4: choice none suppresses calls at output on messages and
    responses too, not only chat (render suppression covered by the chat
    gate)."""
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "r.jsonl"))
    tok = _ByteTokenizer()
    from tilerl.prompt import render_tool_call
    reply = "</think>\n\n" + render_tool_call("Bash", {"command": "ls"})

    class _BigRoom(_ScriptedEngine):
        # The messages route clamps max_tokens to room_for (64 by default); the
        # reply plus request exceeds it and would end "max_tokens" for the
        # wrong reason.
        def room_for(self, prompt_tokens):
            return 1024

    app = create_app(_BigRoom(tok, [reply, reply]), tok)
    with TestClient(app) as c:
        r = c.post(path, json={**body_extra, "tool_choice": "none"})
    assert r.status_code == 200, r.text
    body = r.json()
    blocks = body.get("content") or body.get("output")
    kinds = [b.get("type") for b in blocks]
    assert "tool_use" not in kinds and "function_call" not in kinds, kinds
    assert body.get("stop_reason") in (None, "end_turn") and body.get("stop_reason") != "tool_use"
    # responses keeps a message item; messages may return an empty content list.
    if "output" in body:
        assert any(b.get("type") == "message" for b in blocks)


def test_render_chat_is_chatml():
    """The render half must agree with the stop half: _HfTokenizerAdapter
    stops on <|im_end|>, so the prompt must be ChatML (the old plain-text
    'role: ...' render meant the model never saw the markers it is stopped
    on)."""
    from tilerl.server import ChatMessage, _render_chat

    out = _render_chat(
        [
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="hi"),
        ]
    )
    assert out == (
        "<|im_start|>system\nbe terse<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_replayed_tool_transcript_renders_the_call_and_a_user_tool_response():
    """OpenAI replays a call as assistant.tool_calls and its result as role:"tool"
    (#617 finding 11). Both must reach the model in the checkpoint's own tags:
    the call via render_tool_call, the result via the SAME <tool_response>
    wrapper blocks_to_text uses for Anthropic tool_result — no invented
    dialect, no bare <|im_start|>tool turn."""
    from tilerl.prompt import render_tool_call
    from tilerl.server import ChatMessage, _render_chat

    call = {"id": "call_1_0", "type": "function",
            "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}
    out = _render_chat([
        ChatMessage(role="user", content="run ls"),
        ChatMessage(role="assistant", content=None, tool_calls=[call]),
        ChatMessage(role="tool", content="a.txt", tool_call_id="call_1_0"),
    ])
    expect_call = render_tool_call("Bash", {"command": "ls"})
    assert expect_call in out
    assert "<|im_start|>assistant\n" + expect_call + "<|im_end|>\n" in out
    # role:"tool" re-enters as a USER turn in the existing tool_result wrapper
    assert "<|im_start|>user\n<tool_response>\na.txt\n</tool_response><|im_end|>\n" in out
    assert "<|im_start|>tool" not in out
    # A plain assistant turn with no tool_calls is byte-identical to before:
    # this render runs on every request.
    plain = _render_chat([ChatMessage(role="assistant", content="hello")])
    assert plain == "<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n"


def test_responses_input_text_part_reaches_the_prompt(tmp_path):
    """A Responses message item whose part type is input_text used to flatten to
    "" and 400 as an empty prompt (#617 finding 12)."""
    tok = _ByteTokenizer()
    app = create_app(_ScriptedEngine(tok, ["ok"]), tok)
    with TestClient(app) as c:
        r = c.post("/v1/responses", json={
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "hello"}]}],
        })
    assert r.status_code == 200, r.text
    assert r.json()["output"][0]["content"][0]["text"] == "ok"


def test_top_level_enable_thinking_reaches_the_rendered_prompt():
    """OpenAI/sglang clients send enable_thinking top-level. The HTTP route's
    pydantic model has extra=allow, so without a normalization step the field
    is swallowed into model_extra and thinking stays on (27B V100 smoke:
    50/50 empty content, finish=length, all tokens in reasoning_content).
    Both placements must render the template's closed-think marker."""
    from tilerl.server import ChatCompletionRequest, _normalize_thinking, _render_chat

    def rendered(body):
        req = ChatCompletionRequest.model_validate(_normalize_thinking(dict(body)))
        thinking = (req.chat_template_kwargs or {}).get("enable_thinking")
        return _render_chat(req.messages, thinking)

    msgs = [{"role": "user", "content": "hi"}]
    think = "<" + "think>"
    closed = rendered({"messages": msgs, "enable_thinking": False})
    via_kwargs = rendered({"messages": msgs,
                           "chat_template_kwargs": {"enable_thinking": False}})
    on = rendered({"messages": msgs, "enable_thinking": True})
    assert closed.endswith(f"<|im_start|>assistant\n{think}\n\n</think>\n\n")
    assert closed == via_kwargs
    assert on.endswith(f"<|im_start|>assistant\n{think}\n")


def test_completion_nonstream(client, model_id):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content, f"empty completion: {body!r}"
    assert body["choices"][0]["finish_reason"] == "length"


def test_seedless_requests_decorrelate(client, model_id, monkeypatch):
    """Two seedless requests with the same prompt must not share a sampling
    stream (regression: every seedless request got seed=0, so concurrent
    same-prompt requests returned byte-identical completions).

    Asserts the seeds the server draws, not the text it returns: at temperature
    0.7 the tiny model's distribution is peaked enough that six draws came back
    identical on ubuntu with the seeds all distinct."""
    from tilerl import server as srv

    seeds = []
    real = srv.sampling
    monkeypatch.setattr(srv, "sampling",
                        lambda *a, **k: (lambda p: (seeds.append(p.seed), p)[1])(real(*a, **k)))
    body = {"model": model_id, "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7, "max_tokens": 8}
    for _ in range(4):
        assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert len(set(seeds)) == len(seeds) == 4, seeds


def test_the_stream_arrives_in_pieces_and_never_splits_a_character():
    """SSE must deliver text as it is generated, not one block at the end,
    and no chunk may split a multi-byte character.

    Timing-independent reshape of a pre-existing flake that hung twice on
    ubuntu CI: the old gate streamed off a LIVE engine, so whether a
    multi-byte char straddled a chunk was a race with the poll. Here the
    bytes are fixed and the transport is Starlette's response body writer
    (the same writer uvicorn uses), forced to cut at a raw-byte offset
    guaranteed inside a 3-byte char. The route does not chunk its own
    output (one SSE string per frame); what this locks is the transport
    emitting valid UTF-8 pieces when the chunk boundary lands mid-char.
    """
    import socket

    import uvicorn
    from starlette.responses import StreamingResponse

    # The "café menu → three courses" text from _TextTokenizer.PATTERN: a
    # 2-byte e-acute, a raw 0xff (invalid alone), and a 3-byte arrow --
    # the exact sequence that split across CI chunks.
    text = "café →"
    full = json.dumps(_chat_chunk(
        "c", 0, "tiny", {"content": text}, finish="stop"), ensure_ascii=False)
    cut = full.encode().index("é".encode()) * 1

    async def asgi_app(scope, receive, send):
        body = ("data: " + full + "\n\n" + "data: [DONE]\n\n").encode("utf-8")

        async def chunks():
            pos = 0
            while pos < len(body):
                # Alternate a mid-character window with a safe one; the modulo
                # forces at least one cut inside the 3-byte arrow.
                w = cut % 8 + 2
                yield body[pos:pos + w]
                pos += w

        await StreamingResponse(chunks(), media_type="text/event-stream")(
            scope, receive, send)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        asgi_app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    try:
        for _ in range(400):
            if server.started:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("uvicorn did not start")
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(b"POST / HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
            raw = b""
            while True:
                part = sock.recv(4096)
                if not part:
                    break
                raw += part
    finally:
        server.should_exit = True

    # HTTP/1.1 chunked: <hex size>\r\n<piece>\r\n ... 0\r\n
    body = raw.split(b"\r\n\r\n", 1)[1]
    # The byte cut lands INSIDE the arrow's 3 bytes; the writer must stream
    # every piece except the dangling tail, which goes with the next piece.
    pieces, pos, tail = [], 0, b""
    while pos < len(body):
        nl = body.find(b"\r\n", pos)
        if nl < 0:
            break
        size = int(body[pos:nl], 16)
        if size == 0:
            break
        start = nl + 2
        raw = tail + body[start:start + size]
        # All complete UTF-8 chars stream now; leftover continuation bytes
        # stay buffered exactly the way the production rstrip holds them.
        ok = 0
        for i, b in enumerate(raw):
            if b < 0x80:
                ok = i + 1
            elif b >= 0xC0:
                ok = i
        piece, tail = raw[:ok], raw[ok:]
        piece.decode("utf-8")  # a mid-char piece fails here
        pieces.append(piece)
        pos = start + size + 2
    if tail:
        tail.decode("utf-8")  # the held tail was a real dangling char
    assert len(pieces) > 1, f"the reply did not stream: {len(pieces)} chunk(s)"
    text = b"".join(pieces).decode("utf-8")
    payloads = [json.loads(ln[6:]) for ln in text.splitlines()
                if ln.startswith("data: ") and ln[6:] != "[DONE]"]
    content = "".join(
        f["choices"][0]["delta"]["content"] for f in payloads
        if f["choices"][0].get("delta", {}).get("content"))
    assert content == json.loads(full)["choices"][0]["delta"]["content"]


class _StepEngine:
    """An engine whose peek() reveals ONE more token per call. No race, no weights.

    The live-engine test above cannot assert delta COUNT: that is a race between
    generation speed and the loop's 20 ms poll, and the tiny model can finish a
    24-token reply inside one window -- measured 3 deltas at model seeds 7/42/3, 1 at
    11/99, and 1 on macos-14 CI where the same test passed locally. So the claim
    "text arrives while it generates" gets asserted here instead, against a driver
    that makes the arrival order deterministic.
    """

    def __init__(self, ids: list[int]) -> None:
        self.ids, self.n = ids, 0

    def submit(self, input_ids, params) -> int:
        return 1

    def peek(self, request_id: int):
        if self.n > len(self.ids):
            return None  # left the queues: the loop breaks and calls take()
        self.n += 1
        return self.ids[: self.n - 1]

    def stop_text(self, request_id: int):
        return None

    def take(self, request_id: int):
        return self.ids

    def stats(self) -> dict:
        return {}

    def run(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


def test_a_reply_that_arrives_over_many_polls_streams_over_many_deltas():
    """One delta per poll that reveals new text, and the last one is not the whole reply.

    Catches the load-bearing half of the original loop's two defects: cutting the
    decode at the FIRST U+FFFD (`split("�")[0]`) instead of stripping the trailing
    run. Verified by mutation -- it drops this from 30 deltas to 10, the last of which
    dumps 22 characters, because everything after the reply's first unmappable byte is
    invisible until the tail chunk.

    The other defect -- gating on `sent` (characters) instead of `seen` (tokens) --
    does NOT change the output here and this test does not claim to catch it: with one
    id per byte the two counters advance together, so the substitution is a no-op.
    It mattered on the real tokenizer, where one token is many characters.
    """
    text = _TextTokenizer.PATTERN
    ids = [b + 3 for b in text]  # one id per byte, so a prefix can cut a character
    engine = _StepEngine(ids)
    client = TestClient(create_app(engine, _TextTokenizer()))
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": len(ids),
            "temperature": 0.0, "stream": True}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    lines = [ln for ln in r.text.split("\n") if ln.startswith("data:")]
    payloads = [json.loads(ln[len("data: ") :]) for ln in lines[:-1]]
    deltas = [p["choices"][0]["delta"]["content"] for p in payloads
              if p["choices"][0].get("delta", {}).get("content")]
    full = text.decode("utf-8", errors="replace")  # the 0xFF is a U+FFFD, deliberately

    assert "".join(deltas) == full, f"joined {''.join(deltas)!r} != {full!r}"
    # One delta per character the prefix grows by, minus the multi-byte ones whose
    # first byte reveals no complete character. Loose bound, tight enough to fail
    # either defect: both collapse this to 1.
    assert len(deltas) > len(full) // 2, (
        f"{len(deltas)} deltas for {len(full)} characters revealed one byte at a time: "
        f"the loop is not emitting as text arrives. deltas={deltas!r}"
    )
    assert deltas[-1] != full, f"the last delta is the whole reply: {deltas!r}"
    # The pattern's own 0xFF is a legitimate U+FFFD in the content, so a delta MAY end
    # on one. What must not happen is a delta ending on a replacement char that a later
    # delta resolves into a real character -- that is a split multi-byte sequence. Check
    # the invariant that catches it: each delta only ever APPENDS, so the running join
    # must be a prefix of the final text at every step.
    running = ""
    for i, d in enumerate(deltas):
        running += d
        assert full.startswith(running), (
            f"delta {i} makes the stream diverge from the final text: "
            f"{running!r} is not a prefix of {full!r}"
        )


def test_usage_in_the_stream_is_opt_in_and_counts_tokens_not_characters(client, model_id):
    """The page's tok/s meter needs a real token count, and older clients must not break.

    Without stream_options the stream carries no usage, so a client that reads
    choices[0] on every frame keeps working -- an unconditional usage chunk broke two
    tests in this file. With include_usage the final chunk has usage and an EMPTY choices
    list, which is why a reader must check usage before indexing into choices.

    completion_tokens is the engine's count, not a character estimate: the page used to
    compute chars/4, which is ~4x low for Chinese (roughly one token per character).
    """
    body = {"model": model_id, "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16, "temperature": 0.0, "seed": 5, "stream": True}

    def frames(extra):
        resp = client.post("/v1/chat/completions", json={**body, **extra})
        assert resp.status_code == 200, resp.text
        lines = [ln for ln in resp.text.split("\n") if ln.startswith("data: {")]
        return [json.loads(ln[len("data: ") :]) for ln in lines]

    plain = frames({})
    assert all(p.get("usage") is None for p in plain), "usage must be opt-in"
    assert all(p["choices"] for p in plain), "every frame carries a choice without opt-in"

    opted = frames({"stream_options": {"include_usage": True}})
    last = opted[-1]
    assert last["choices"] == [], f"the usage chunk must carry no choices: {last!r}"
    usage = last["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] == 16, usage
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert all(p["choices"] for p in opted[:-1]), "only the last frame may be choices-less"


def test_every_api_path_waits_the_same_wall_clock_for_one_completion():
    """The front ends submit to one engine, so a cap that fits one fits all of them.

    They drifted: 5cdbf7e raised the OpenAI path's cap from 600 s to 1800 s and changed
    only server.py, so `/v1/messages` -- what Claude Code speaks -- kept waiting 600 s for
    the same work on the same card. The reason it gave, "a 4K prefill takes ~600 s", is
    **withdrawn**: that was a B=8 whole-tick cost quoted per request, and a live V100
    measured a 3478-token request at 39.1 s
    (errors/2026-09-05-the-600s-that-justified-1800s-was-a-batch-tick.md). What this gate
    asserts is unaffected, because it is about the constants agreeing, not about which
    value they agree on.

    Read out of the source rather than by running a 600 s request: the number is a
    constant, and the defect was constants that should have been one.

    Scanned over THREE modules, not just server.py. The earlier version read only
    server.py, and responses.py held a third and a fourth copy of the literal it could not
    see -- a gate aimed at two of the three routes that spell the cap. `_await_completion`
    is the one the live V100 child actually waits on for `/v1/chat/completions`.

    Two assertions, because consolidating has two failure modes: a route that spells its
    own number (the original drift), and a route that stops referencing the shared constant
    at all. Numeric literals only -- `[\\d_]+` also matches the leading underscore of
    `_COMPLETION_TIMEOUT_S`, so the first draft of this gate died in `float("_")`.
    """
    import pathlib
    import re

    from tilerl import messages as msg
    from tilerl import responses as rsp
    from tilerl import server as srv

    NUM = r"(\d[\d_]*(?:\.\d*)?)"
    lits: dict[str, set[float]] = {}
    refs: dict[str, bool] = {}
    for mod in (srv, rsp, msg):
        text = pathlib.Path(mod.__file__).read_text()
        found = {float(m) for m in re.findall(rf"time\.monotonic\(\) \+ {NUM}", text)}
        found |= {float(m) for m in re.findall(rf"timeout_s: float = {NUM}", text)}
        if found:
            lits[mod.__name__] = found
        refs[mod.__name__] = "_COMPLETION_TIMEOUT_S" in text

    bad = {n: sorted(v) for n, v in lits.items() if v != {msg._COMPLETION_TIMEOUT_S}}
    assert not bad, (
        f"these modules spell their own per-completion cap instead of "
        f"messages.py's {msg._COMPLETION_TIMEOUT_S} s: {bad}. They all submit to the same "
        f"engine on the same card, so whichever is shorter times out work the others "
        f"tolerate."
    )
    # The other half: zero literals is the goal, so `bad` is empty both when every route
    # imports the constant and when a route quietly stopped waiting on one at all.
    silent = sorted(n for n, ok in refs.items() if not ok)
    assert not silent, (
        f"{silent} no longer reference _COMPLETION_TIMEOUT_S; a route that waits on "
        f"engine.take with no shared cap is the drift this gate exists to catch, and it "
        f"passes the literal check by having no literal."
    )


def test_the_sse_stream_keeps_the_shape_a_reader_has_to_handle(client, model_id):
    """Two properties of the real stream a careless reader dies on.

    The FIRST frame carries `{"role": "assistant"}` and no content, and the usage chunk
    carries no `choices` at all -- a reader that indexes `choices[0].delta` on every
    frame throws on it. This used to run the chat page's own SSE reader over these
    bytes, which is the right shape of gate and no longer the right pairing: the page
    speaks WebSocket now, and `tests/test_chat_ui.py` closes that loop against the
    frames `/ws/chat` emits. What is left here is the wire contract itself, which the
    SDK clients in `test_api_sdk.py` depend on.
    """
    resp = client.post("/v1/chat/completions", json={
        "model": model_id, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8, "temperature": 0.0, "seed": 11,
        "stream": True, "stream_options": {"include_usage": True},
    })
    assert resp.status_code == 200, resp.text[:200]
    sse = resp.text
    assert "[DONE]" in sse and '"usage"' in sse, "the fixture is not a complete stream"
    frames = [json.loads(ln[6:]) for ln in sse.splitlines()
              if ln.startswith("data: ") and ln[6:] != "[DONE]"]
    assert len(frames) > 1, f"the stream yielded nothing usable: {frames}"

    def has_text(f):
        ch = f.get("choices") or [{}]
        return bool(ch[0].get("delta", {}).get("content"))

    assert not has_text(frames[0]), (
        "this server's first frame now carries content, so the assertion no longer "
        "exercises the role-only frame a real reply starts with"
    )
    empty_choices = [f for f in frames if isinstance(f.get("choices"), list)
                     and not f["choices"]]
    assert len(empty_choices) == 1, (
        f"exactly one frame must carry an empty choices list -- the usage chunk. Got "
        f"{len(empty_choices)}. A reader indexing choices[0] unconditionally dies on it."
    )
    assert empty_choices[0].get("usage"), "the final usage-only chunk did not arrive"
    # Content frames carry cumulative usage so a live gauge reads tokens rather than
    # frames -- this server coalesces ~1.8 tokens into each frame on the 27B (measured:
    # 109 frames for 200 tokens), so a frame-counting gauge reads ~1.8x low. The tiny
    # model finishes inside one poll, so this fixture legitimately has a single content
    # frame: the invariant is "every content frame carries a count, and the counts never
    # go backwards", not "there are several".
    counts = [f["usage"]["completion_tokens"] for f in frames if f.get("usage")]
    assert len(counts) == sum(map(has_text, frames)) + 1, (
        f"{len(counts)} frames carried usage but there are {sum(map(has_text, frames))} "
        f"content frames plus one final chunk: a content frame without a count leaves "
        f"the live gauge counting frames for that stretch"
    )
    assert counts == sorted(counts), f"cumulative token counts went backwards: {counts}"

class _SplitlinesTokenizer(_ByteTokenizer):
    """Decodes a fixed text containing the three characters `splitlines()` cuts on and
    `split("\\n")` does not.

    `_sse` writes `json.dumps(..., ensure_ascii=False)`, so a non-ASCII character reaches
    the wire verbatim. Measured: of the nine separators `splitlines()` splits on beyond
    `\\n`, only these three cut a payload mid-JSON -- `\\v \\f \\r \\x1c \\x1d \\x1e` are
    ASCII controls that `json.dumps` escapes, so no literal byte is ever emitted. All
    three are reachable from sampled ids (U+0085 is bytes 194,133 -> ids 197,136, inside
    tiny()'s vocab of 320), which is why the failure was intermittent and appeared on
    ubuntu but not macos rather than being deterministic.
    """

    PATTERN = "a\x85b c d".encode()

    def decode(self, ids) -> str:
        n = sum(1 for i in ids if 3 <= i < 259)
        return self.PATTERN[:n].decode("utf-8", errors="replace")


def test_sse_frames_survive_a_separator_splitlines_cuts_on():
    """The stream parses when a delta carries U+0085 / U+2028 / U+2029.

    Negative control: with `splitlines()` in place of `split("\\n")` below, the same
    stream raises `json.JSONDecodeError: Unterminated string`, which is the failure this
    test exists to keep out. The assertion on `joined` is what makes the test exercise
    the characters rather than merely tolerate their absence.
    """
    engine = _build_engine(seed=42)
    engine.run()
    try:
        with TestClient(create_app(engine, _SplitlinesTokenizer())) as c:
            resp = c.post("/v1/chat/completions", json={
                "model": "tiny", "messages": [{"role": "user", "content": "hi"}],
                "stream": True, "max_tokens": 16, "temperature": 0.0, "seed": 5,
            })
        assert resp.status_code == 200, resp.text
        lines = [ln for ln in resp.text.split("\n") if ln.startswith("data: {")]
        assert lines, f"no SSE payload frames: {resp.text[:200]!r}"
        payloads = [json.loads(ln[len("data: ") :]) for ln in lines]
    finally:
        engine.shutdown()

    joined = "".join(p["choices"][0].get("delta", {}).get("content") or "" for p in payloads)
    assert any(ch in joined for ch in ("\x85", " ", " ")), (
        f"the stream carried none of the three characters, so this test proves nothing "
        f"about them: {joined!r}"
    )
    # Segment COUNTS do not discriminate: cutting at \x85 splits one frame into a `data:`
    # half and a remainder that no longer starts with `data:`, so the remainder is filtered
    # out and both splits yield the same total. What differs is the surviving frame's
    # content, so the control below parses rather than counts.
    #
    # The negative control, executed rather than described: the OLD parse must actually
    # raise on this very stream.
    with pytest.raises(json.JSONDecodeError):
        old = [ln for ln in resp.text.splitlines() if ln.startswith("data: {")]
        [json.loads(ln[len("data: ") :]) for ln in old]


def test_completion_stream(client, model_id):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers.get("content-type", ""), (
        f"not an SSE stream: {resp.headers.get('content-type')!r}"
    )
    # split("\n"), NOT splitlines(): SSE frames are \n-delimited, and splitlines() also
    # splits on \x85,   and  , which `_sse`'s ensure_ascii=False puts on the wire
    # verbatim. Measured -- those three cut a payload mid-JSON while \v \f \r \x1c-\x1e do
    # not (json.dumps escapes the ASCII controls), and all three are reachable from sampled
    # ids: \x85 is bytes 194,133 -> ids 197,136, both inside tiny()'s vocab of 320. That is
    # the intermittent `Unterminated string` this test hit on ubuntu and not macos.
    lines = [line for line in resp.text.split("\n") if line.startswith("data:")]
    assert lines, "no SSE data lines received"
    assert lines[-1].strip() == "data: [DONE]", f"stream did not end with [DONE]: {lines[-1]!r}"

    payloads = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    contents = [p["choices"][0].get("delta", {}).get("content") for p in payloads]
    assert any(content for content in contents), f"no delta content in chunks: {payloads!r}"
    assert payloads[-1]["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize(
    "field,value",
    [("max_tokens", 0), ("temperature", -0.1), ("temperature", 2.1), ("top_p", 0.0)],
)
def test_sampling_bounds(client, field, value):
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], field: value},
    )
    # 400 with OpenAI's envelope, not FastAPI's 422 `detail` list: the SDKs map
    # 422 to UnprocessableEntityError, so a client catching BadRequestError missed
    # every rejected field. The field is named in the message.
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert field in body["error"]["message"]


def test_configured_tokenizer_fails_closed(tmp_path):
    with pytest.raises(Exception):
        get_tokenizer(str(tmp_path))


@pytest.mark.parametrize("path,body", [
    ("/v1/messages", {"model": "tiny", "max_tokens": 8,
                      "messages": [{"role": "user", "content": "hi"}]}),
    # Both, because /v1/responses had the SAME defect and 27's brief named only messages:
    # `async def responses` called `_run` directly, and its own poll loop sleeps at :186.
    # A gate on one route would have left a known instance of this defect in the tree.
    ("/v1/responses", {"model": "tiny", "max_output_tokens": 8, "input": "hi"}),
])
def test_a_request_in_flight_does_not_freeze_the_server(tmp_path, monkeypatch, path, body):
    """`/health` must answer while a reply is being generated, on every async route.

    `_run` polls `engine.take` with `time.sleep(0.02)`. The routes are `async def`, so
    before the fix that poll ran ON the event loop and every other route starved for the
    length of the reply — measured on the live V100 as a 10-minute freeze on a 30k-token
    prompt, with /health timing out and CLOSE-WAIT sockets piling up. The chat route has
    always awaited its wait through `asyncio.to_thread` (`server.py`); this is that, on the
    two routes that did not.

    Deterministic, not a wall-clock bound: the engine's `take` sets `entered` and then
    blocks on a release EVENT, so a reply is provably in flight (the route is parked in
    take, not merely "probably polling inside a window"). A worker calls /health and the
    gate fails unless it returns BEFORE the release is set — a route that awaited take on
    the event loop cannot serve /health until release. The join margin is a deadlock
    detector only, never a latency assertion.
    """
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "loop.jsonl"))
    tok = _ByteTokenizer()

    class _SlowEngine(_ScriptedEngine):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.entered = threading.Event()
            self.release = threading.Event()

        def take(self, request_id: int):
            # First poll parks; after release the scripted engine returns the reply.
            if not self.entered.is_set():
                self.entered.set()
                self.release.wait(30.0)
            return super().take(request_id)

    engine = _SlowEngine(tok, ["</think>\n\ndone"])
    app = create_app(engine, tok, model_name="tiny")
    with TestClient(app) as c:
        done: dict[str, object] = {}
        t = threading.Thread(target=lambda: done.update(
            code=c.post(path, json=body).status_code))
        t.start()
        try:
            assert engine.entered.wait(10.0), f"{path} never entered take; the arm proves nothing"
            # The route is provably parked in take (release unset): /health must answer.
            box: dict[str, object] = {}
            ht = threading.Thread(target=lambda: box.update(resp=c.get("/health")))
            ht.start()
            ht.join(5.0)
            assert not ht.is_alive(), (
                f"/health did not return while {path} was in take — the route blocks the "
                f"event loop instead of awaiting through asyncio.to_thread")
            assert not engine.release.is_set(), (
                "/health returned only after the in-flight request completed")
            health = box["resp"]
        finally:
            engine.release.set()
            t.join(timeout=30)

    assert health.status_code == 200, health.text
    assert done.get("code") == 200, f"the {path} request itself failed: {done}"


def test_health_does_not_wait_on_the_engine_lock(tmp_path):
    """`/health` must read a published snapshot while `step()` HOLDS `_lock`.

    Separate defect from the `to_thread` freeze: on the live V100 during a
    21.7k-token prefill /health ran at a median of 8.12 s and a max of 87.66 s,
    because `stats()` took the lock a 43-chunk prefill holds one chunk at a time.

    Deterministic, not a wall-clock performance bound: the forward blocks on a
    release EVENT (so the lock is provably held, not "probably held inside a
    window"), and a worker calls stats(). The assertion is binary — stats()
    returns BEFORE the release is set (it never touched the lock). A lock-taking
    stats() deadlocks the worker until the watchdog timeout. The generous 5 s
    margin only distinguishes "returned" from "hung"; it is not asserted as a
    latency target, so a slow CI cannot turn it red.

    A real `Engine` is used, not a double: the property under test is which lock
    `stats()` takes, and a double that reimplements `stats()` would assert its
    own behaviour. The only substitution is the forward (the layer below).
    """
    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=43), get_backend(),
                          num_blocks=32, num_slots=4, max_batch=4, max_total_tokens=4096)

    entered = threading.Event()
    release = threading.Event()

    def _slow_forward(*_a, **_kw):
        entered.set()
        release.wait(20.0)  # hold the step lock until the test releases it

    engine._run_forward = _slow_forward
    engine.submit([1, 2, 3], SamplingParams(max_new_tokens=4))
    engine.run()
    assert entered.wait(10.0), "the forward never started; the arm proves nothing"
    try:
        box: dict[str, object] = {}

        def _read():
            box["snap"] = engine.stats()

        reader = threading.Thread(target=_read)
        reader.start()
        # Lock is provably held (release unset). A lock-free stats returns at once;
        # a lock-taking stats cannot return until release.set() below.
        reader.join(5.0)
        assert not reader.is_alive(), (
            "stats() did not return while step() held the lock — /health waits on "
            "the engine lock instead of reading a published snapshot")
        assert not release.is_set(), "stats returned only because the lock was released"
        snap = box["snap"]
    finally:
        release.set()
        engine.shutdown()

    assert isinstance(snap, dict) and "pool_used_blocks" in snap, snap


def test_stats_snapshot_is_built_once_per_tick_and_carries_tick_end_state():
    """step()'s stats snapshot is built once per steady tick, not twice, and the
    published dict is the tick's END state. Regression for the perf change that
    made the pre-forward _build_stats conditional: it still fires on an admit
    tick and when submit() queues a waiter that cannot be admitted yet.

    Negative control that must be red: keep ONLY the old unconditional
    pre-forward build. The run then builds once per tick (builds == ticks)
    instead of once per tick plus one per demand tick, and the final tick's
    snapshot is its PRE-forward state -- the finished row still reads
    running=1, finished=0, tokens_generated one short -- so the comparison
    against a fresh build fails.
    """
    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=42), get_backend(),
                          num_blocks=32, num_slots=4, max_batch=1,
                          max_total_tokens=4096, sparse_k=0)
    orig_build = engine._build_stats
    builds = 0

    def counting_build():
        nonlocal builds
        builds += 1
        return orig_build()

    engine._build_stats = counting_build
    # The invariant the conditional build protects: while a forward runs, a
    # snapshot must already be published, or stats() falls back to its locking
    # path -- a silent slowdown, not an error. Checked at EVERY forward entry,
    # so it covers both the admit tick and later no-admit decode ticks.
    orig_forward = engine._run_forward
    forward_ticks = 0

    def forward_with_snapshot(decodes, prefills, chunks):
        nonlocal forward_ticks
        forward_ticks += 1
        assert engine._stats_snapshot is not None, "no lock-free snapshot during the forward"
        return orig_forward(decodes, prefills, chunks)

    engine._run_forward = forward_with_snapshot
    rid = engine.submit([7, 11, 13], SamplingParams(temperature=0.0, max_new_tokens=6))
    ticks = 0
    out = None
    blocked = None
    for _ in range(100):
        ticks += 1
        engine.step()
        if ticks == 1:
            # max_batch=1 and rid still occupies the batch, so this parks in the
            # waiting queue and exercises the submit-driven pre-forward build.
            blocked = engine.submit([5] * 10, SamplingParams(max_new_tokens=1))
        out = engine.take(rid)
        if out is not None:
            break
    assert out is not None, "the row never finished"
    assert engine.take(blocked) is None, "the oversized prompt unexpectedly admitted"

    # One end build per tick, plus one pre-forward build per demand tick: the
    # first tick (admit) and the tick after the blocked waiter appeared.
    assert forward_ticks == ticks
    assert builds == ticks + 2, f"{builds} builds over {ticks} ticks"
    snap = engine._stats_snapshot
    assert snap is not None
    assert snap["running"] == 0 and snap["finished"] == 1 and snap["waiting"] == 1
    assert snap["tokens_generated"] == len(out)
    fresh = orig_build()
    assert snap == fresh


def test_messages_route_records_token_ids(client, tmp_path, monkeypatch):
    """The Messages shim answers Claude Code's shape and records the ids.

    The record is the reason the route exists: BPE is not concatenation-
    invariant, so a transcript's text cannot be re-encoded into a guaranteed-
    identical id sequence, and GRPO on a mismatched sequence is a silently
    wrong gradient. This asserts the ids come back on the wire, not that they
    can be rebuilt.
    """
    rec = tmp_path / "messages.jsonl"
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(rec))
    body = {
        # the shape a real Claude Code request carries, measured 2026-09-02:
        # system as a block list, tools as JSON Schema, content as blocks
        "model": "tiny",
        "max_tokens": 8,
        "system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "Bash", "description": "Run a command",
                   "input_schema": {"properties": {"command": {"type": "string"}}}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["type"] == "message" and out["role"] == "assistant"
    assert out["stop_reason"] in ("end_turn", "max_tokens", "tool_use")
    # By type, not index: a thinking block now leads a reply whose prompt opened
    # <think>, so content[0] is no longer the text block.
    assert out["content"]
    assert {b["type"] for b in out["content"]} <= {"thinking", "text", "tool_use"}
    assert any(b["type"] in ("text", "tool_use") for b in out["content"])
    rid = r.headers["x-tilerl-request-id"]

    row = json.loads(rec.read_text().splitlines()[-1])
    assert str(row["request_id"]) == rid, "the header must name the recorded row"
    assert row["prompt_ids"] and row["completion_ids"]
    # one score per generated token: what a policy gradient consumes
    assert len(row["logprobs"]) == len(row["completion_ids"])
    assert row["stop_reason"] == out["stop_reason"]


@pytest.mark.parametrize("choice,refused", [
    ({"type": "any"}, True),      # Anthropic's "call some tool"
    ({"type": "tool", "name": "Bash"}, True),
    ("required", True),           # OpenAI's spelling, same claim
    ({"type": "auto"}, False),    # a hint, which is what we already do
    ({"type": "none"}, False),
    (None, False),
])
def test_messages_refuses_a_tool_choice_it_cannot_honour(client, tmp_path, monkeypatch,
                                                         choice, refused):
    """Forcing a call is unimplementable here, so it must 400 rather than be ignored.

    Found by the live endpoint, not by reading: #201's `unknown_fields` produced its first
    non-null value in production, `{'tool_choice': 'dict{type}'}` — a documented Anthropic
    parameter this route declared nowhere and dropped silently. `/v1/responses` has refused
    it since it landed; `/v1/messages` never got the treatment. The lie surfaces a turn
    later, when the client assumes the tool it forced was the tool that ran.

    Both spellings, because `unsupported_choice` takes either: a bare string and a
    `{"type": ...}`. auto/none stay 200 — `auto` is what a client sends when it means
    "your choice", so refusing it would refuse a request that asked for nothing.
    """
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "tc.jsonl"))
    body = {"model": "tiny", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "Bash", "description": "run",
                       "input_schema": {"properties": {}}}]}
    if choice is not None:
        body["tool_choice"] = choice
    r = client.post("/v1/messages", json=body)
    if refused:
        assert r.status_code == 400, f"{choice!r} was accepted: {r.text[:200]}"
        assert "tool_choice" in r.json()["error"]["message"], r.text
    else:
        assert r.status_code == 200, f"{choice!r} must not be refused: {r.text[:200]}"


def test_messages_stream_is_anthropic_sse(client, tmp_path, monkeypatch):
    """stream=true emits the event names Claude Code's parser expects."""
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "s.jsonl"))
    r = client.post("/v1/messages", json={
        "max_tokens": 4, "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200, r.text
    events = [ln[7:] for ln in r.text.splitlines() if ln.startswith("event: ")]
    assert events[0] == "message_start" and events[-1] == "message_stop"
    for needed in ("content_block_start", "content_block_delta", "content_block_stop"):
        assert needed in events, f"{needed} missing from {events}"


class _ScriptedEngine:
    """An engine that returns canned completions, in submit order.

    The semantic half of stage 1's gate cannot use a real tiny model: random
    weights emit noise until max_tokens and can never produce a well-formed
    tool call, so the tool_use path would be untestable until a checkpoint
    exists. This satisfies the same duck type the real Engine does
    (submit/poll/take/step/logprobs/stats) and lets the SHIM's rendering of
    tool_use and stop_reason be gated on a machine with no weights at all --
    which is also what stage 2's launcher needs.
    """

    def __init__(self, tokenizer, replies: list[str]):
        self._tok = tokenizer
        self._replies = list(replies)
        self._next = 0
        self._done: dict[int, list[int]] = {}
        self._lp: dict[int, list[float]] = {}
        self._taken: set[int] = set()
        self._peeked: dict[int, int] = {}
        self._stopped: dict[int, str] = {}
        self.params: list = []  # what each submit asked for, for the stop-sequence gates

    def peek(self, request_id: int):
        """Half the reply, then all of it, then gone -- the live path and the tail."""
        ids = self._done.get(request_id)
        n = self._peeked[request_id] = self._peeked.get(request_id, 0) + 1
        return None if ids is None or n > 2 else ids[: len(ids) * n // 2]

    def submit(self, input_ids, params=None) -> int:
        self._next += 1
        self.params.append(params)
        text = self._replies.pop(0) if self._replies else ""
        ids = self._tok.encode(text)
        # Honour stop_texts the way the engine does: cut the ids after the token
        # that completes the first match, so the routes are gated against the same
        # contract a real engine gives them (the sequence is still in `output`).
        # Past the reasoning closer only, for the engine's reason: a stop like
        # "\n\n" inside <think> would return a truncated thought and no answer.
        closer = self._tok.decode(list(getattr(params, "end_think_ids", ()) or ()))
        start = (text.find(closer) + len(closer)) if closer else 0
        for stop in (getattr(params, "stop_texts", ()) or ()) if start >= len(closer) else ():
            hit = text.find(stop, start)
            if hit >= 0:
                ids = self._tok.encode(text[: hit + len(stop)])
                self._stopped[self._next] = stop
                break
        self._done[self._next] = ids
        self._lp[self._next] = [-0.1] * len(ids)
        return self._next

    def stop_text(self, request_id: int):
        return self._stopped.pop(request_id, None)

    def take(self, request_id: int):
        return self._done.pop(request_id, None)

    def poll(self) -> dict:
        out, self._done = dict(self._done), {}
        return out

    def step(self) -> None:
        return None

    def logprobs(self, request_id: int):
        if request_id in self._lp:
            self._taken.add(request_id)
            return self._lp.pop(request_id)
        if request_id in self._taken:  # same contract as the real engine
            raise KeyError(f"logprobs for request {request_id} were already taken")
        return None

    def stats(self) -> dict:
        return {"waiting": 0, "running": 0, "finished": len(self._taken)}

    def room_for(self, prompt_tokens: int) -> int:
        # Part of the seam a route may call when max_tokens is omitted. This engine
        # has no KV pool to bound, so the number is arbitrary and deliberately not
        # the real engine's formula: whether the remainder is computed CORRECTLY is
        # asserted against a real engine, and re-deriving it here would let a broken
        # `Engine.room_for` still pass every arm in this file.
        return 64


def test_messages_tool_use_round_trip(tmp_path, monkeypatch):
    """The full agent shape: tool_use out, tool_result back in, answer out.

    Gates what a real Claude Code loop needs from the shim -- a structured
    tool_use block with stop_reason="tool_use", and a follow-up request whose
    tool_result content is rendered back into the prompt -- without needing a
    model that can produce either.
    """
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "rt.jsonl"))
    tok = _ByteTokenizer()
    # /v1/messages opens <think> in the prompt, so a real reply starts with the closer
    engine = _ScriptedEngine(tok, [
        "</think>\n\n" + render_tool_call("Bash", {"command": "ls"}),
        "</think>\n\nthere are 3 files",
    ])
    app = create_app(engine, tok)
    with TestClient(app) as c:
        first = c.post("/v1/messages", json={
            "max_tokens": 64,
            "tools": [{"name": "Bash", "description": "Run a command",
                       "input_schema": {"properties": {"command": {}}}}],
            "messages": [{"role": "user", "content": "list the files"}],
        })
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["stop_reason"] == "tool_use", body
        block = body["content"][0]
        assert block["type"] == "tool_use" and block["name"] == "Bash"
        assert block["input"] == {"command": "ls"}
        assert block["id"].startswith("toolu_")

        # the client executes the tool and sends the result back, as Claude Code does
        second = c.post("/v1/messages", json={
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "list the files"},
                {"role": "assistant", "content": [block]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block["id"], "content": "a.py b.py c.py"}
                ]},
            ],
        })
        assert second.status_code == 200, second.text
        final = second.json()
        assert final["stop_reason"] == "end_turn"
        assert final["content"][0]["type"] == "text"
        assert "3 files" in final["content"][0]["text"]

    rows = [json.loads(x) for x in (tmp_path / "rt.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["stop_reason"] == "tool_use" and rows[1]["stop_reason"] == "end_turn"
    for r in rows:
        assert len(r["logprobs"]) == len(r["completion_ids"])
    # The tool_result reached the prompt. Not "turn 2 is longer" -- turn 1
    # carries the tools block and turn 2 does not, so turn 2 is the SHORTER
    # render (211 vs 218 ids as written). Assert the content instead.
    assert "<tool_response>\na.py b.py c.py\n</tool_response>" in tok.decode(rows[1]["prompt_ids"])
    assert "<function=Bash>" in tok.decode(rows[1]["prompt_ids"])


def test_parallel_tool_calls_become_separate_blocks(tmp_path, monkeypatch):
    """Two <tool_call> blocks in one completion -> two tool_use blocks out.

    Claude Code issues parallel calls and returns one tool_result per id, so
    dropping all but the first would strand the rest of the turn. The leading
    prose is its own text block, as the real API does.
    """
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "p.jsonl"))
    tok = _ByteTokenizer()
    reply = ("Listing both.\n" + render_tool_call("Bash", {"command": "ls"})
             + "\n" + render_tool_call("Bash", {"command": "pwd"}))
    app = create_app(_ScriptedEngine(tok, ["</think>\n\n" + reply]), tok)
    with TestClient(app) as c:
        body = c.post("/v1/messages", json={
            "max_tokens": 64,
            "tools": [{"name": "Bash", "description": "Run a command",
                       "input_schema": {"properties": {"command": {"type": "string"}}}}],
            "messages": [{"role": "user", "content": "list and pwd"}],
        }).json()
    assert body["stop_reason"] == "tool_use"
    kinds = [b["type"] for b in body["content"]]
    assert kinds == ["text", "tool_use", "tool_use"], body["content"]
    assert body["content"][0]["text"] == "Listing both."
    ids = [b["id"] for b in body["content"] if b["type"] == "tool_use"]
    assert len(set(ids)) == 2, f"tool_use ids must be distinct: {ids}"
    assert [b["input"]["command"] for b in body["content"][1:]] == ["ls", "pwd"]


def test_max_tokens_is_clamped_not_refused(client, tmp_path, monkeypatch):
    """A huge max_tokens stops at the context edge instead of 400-ing.

    Claude Code always asks for 32000; the real API accepts it and stops. The
    engine refuses prompt+max_new_tokens over its budget, so the shim clamps.
    """
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "c.jsonl"))
    r = client.post("/v1/messages", json={
        "max_tokens": 32000,  # far past the 4096-token test engine
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200, r.text
    assert r.json()["stop_reason"] in ("end_turn", "max_tokens", "tool_use")


def test_the_messages_clamp_honours_the_pool_not_only_the_context(tmp_path, monkeypatch):
    """A pool-bound engine must admit a 32000-token ask, the way Claude Code sends it.

    The clamp used to compute `max_total_tokens - prompt` by hand while `submit` enforces
    that ceiling AND the KV pool, so `total` landed on the pool's edge exactly and was
    refused there by `width - 1` tokens -- a 400 on every Claude Code turn against the
    V100, whose 2048 blocks hold precisely its 32768-token context.

    The fixture reproduces that region on CPU: 32 blocks is 512 tokens against a 4096
    context, so the pool binds by ~9x and the context ceiling alone would admit an ask
    the engine then refuses. The chat route is the control -- it has called `room_for`
    since #195, so it stays 200 whatever this route does.
    """
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "pool.jsonl"))
    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=41), get_backend(),
                          num_blocks=32, num_slots=4, max_batch=4, max_total_tokens=4096,
                          sparse_k=0)  # dense: the clamp reads the dense pool capacity
    engine.run()
    try:
        assert engine.room_for(1) < engine.limits.max_total_tokens - 1, (
            "fixture does not bind on the pool, so it cannot see the defect")
        with TestClient(create_app(engine, _ByteTokenizer(), model_name="tiny"),
                        raise_server_exceptions=False) as c:
            body = {"model": "tiny", "max_tokens": 32000,
                    "messages": [{"role": "user", "content": "hi"}]}
            got = c.post("/v1/messages", json=body)
            control = c.post("/v1/chat/completions", json={k: v for k, v in body.items()
                                                           if k != "max_tokens"})
        assert control.status_code == 200, f"the control route broke: {control.text}"
        assert got.status_code == 200, (
            f"a 32000-token ask 400-ed on a pool-bound engine: {got.text} — the clamp "
            f"bounds one of submit's two ceilings")
        row_file = tmp_path / "pool.jsonl"
        # Named, because the refusal is raised inside `submit` before the recorder runs:
        # with the defect present there is no row at all, and the bare FileNotFoundError
        # reads as a broken test rather than the second half of the same finding.
        assert row_file.exists(), (
            "no recorder row: submit refused before `_record`, so the request log cannot "
            "see this failure class")
        row = json.loads(row_file.read_text().splitlines()[-1])
        # The clamp is the pool's number, and tight: room_for is what submit accepts to
        # the token, so an off-by-one here is the shape the hand-rolled version had.
        room = engine.room_for(row["prompt_len"])
        assert row["budget"] == room, f"budget {row['budget']} is not room_for {room}"
        assert room < row["engine_limit"] - row["prompt_len"], (
            "the pool did not bind on the recorded prompt, so `budget` proves nothing")
        with pytest.raises(ValueError, match="KV pool"):
            engine.submit(list(range(row["prompt_len"])),
                          SamplingParams(max_new_tokens=room + 1))
    finally:
        engine.shutdown()


def test_image_blocks_are_refused_not_dropped(client, tmp_path, monkeypatch):
    """A text-only model must say so rather than answer a turn missing its subject."""
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "i.jsonl"))
    r = client.post("/v1/messages", json={
        "max_tokens": 8,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "iVBOR"}}]}],
    })
    assert r.status_code == 400, r.text
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_an_undeclared_field_is_recorded_by_shape_on_every_route(client, tmp_path,
                                                                monkeypatch, recwarn):
    """A field the client sends and we do not declare must become visible, not vanish.

    Before `extra="allow"`, pydantic dropped unknown keys before any handler ran AND the
    recorded row was built from the parsed model, so an ignored field was invisible twice:
    "no unsupported fields observed" could not be told from "we dropped six of them". That
    is not hypothetical -- `previous_response_id` had to be DECLARED in order to be refused.

    Values are never recorded, only shapes: a body carries the user's prompt and may carry
    credentials. Both halves are asserted -- the field is named, and its contents are absent.
    """
    rec = tmp_path / "unknown.jsonl"
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(rec))
    secret = "this-string-must-not-be-recorded"
    extra = {"made_up_field": secret, "a_number": 7, "an_object": {"type": "auto", "k": 1}}

    # /v1/messages is the only route with a recorder, and the one Claude Code drives.
    r = client.post("/v1/messages", json={"model": "tiny", "max_tokens": 8,
                                          "messages": [{"role": "user", "content": "hi"}],
                                          **extra})
    assert r.status_code == 200, r.text
    text = rec.read_text(encoding="utf-8")
    got = json.loads(text.splitlines()[-1])["unknown_fields"]
    # Named before indexed: `set(None)` raises a TypeError that reads as a broken test.
    assert got, ("the row recorded no unknown fields at all -- the request model is "
                 "dropping them before the handler runs (extra=\"ignore\")")
    assert set(got) == set(extra), got
    assert got["made_up_field"] == f"str[{len(secret)}]", got
    assert got["a_number"] == "int(7)", got
    # Keys of a nested object say whether we should honour it; its values do not.
    assert got["an_object"] == "dict{k,type}", got
    assert secret not in text, "the field's VALUE reached the row"

    # No recorder on these two, so the warning is the only signal -- and it must name it.
    for path, body in (("/v1/chat/completions",
                        {"model": "tiny", "stream": False, "max_tokens": 8,
                         "messages": [{"role": "user", "content": "hi"}], **extra}),
                       ("/v1/responses", {"model": "tiny", "input": "hi", **extra})):
        recwarn.clear()
        assert client.post(path, json=body).status_code == 200, path
        texts = [str(w.message) for w in recwarn]
        assert any("made_up_field" in t for t in texts), (
            f"{path} ignored an undeclared field with no warning: {texts}")
        assert secret not in " ".join(texts), f"{path} warned with the field's VALUE"


def test_every_engine_the_routes_accept_implements_what_they_call():
    """The seam is what the routes CALL, and every implementation has to have all of it.

    Twice now a method the routes need was missing from `DataParallelEngine` and shipped:
    `limits` 400-ed every Claude Code turn under `--devices`, and `room_for` 500-ed every
    request that omitted `max_tokens`. One arm per method catches the method it was written
    for and nothing else, so this enumerates instead — the names are read out of the route
    modules' own source, so a route that starts calling `engine.foo()` extends the required
    set without anyone remembering to add an arm here.

    DataParallelEngine was deleted 2026-09-09 (its hand-written forwarding seam silently
    missed a method six times in ten days). This gate stays: it enumerates the seam, and
    any multi-card wrapper that comes back must be added to the instances below and pass —
    that is the rebuild gate, written here so it is not re-litigated in a review.
    """
    import inspect
    import re

    from tilerl import messages, responses, server

    called: set[str] = set()
    for mod in (server, messages, responses):
        src = inspect.getsource(mod)
        called |= set(re.findall(r"\bengine\.([a-z_][a-z0-9_]*)", src))
        # `getattr(engine, "limits", ...)` is a call on the seam too, and the regex
        # above cannot see it: messages.py reads `limits` exactly this way.
        called |= set(re.findall(r'getattr\(\s*engine\s*,\s*"([a-z_][a-z0-9_]*)"', src))
    # Prose, not calls: "engine.py" in a docstring, and a comment in server.py's /health
    # explaining why loop liveness deliberately does NOT read engine._thread. Reading a
    # comment as a call is how this gate would demand an attribute nothing needs.
    called -= {"py", "_thread"}

    # The set is asserted, not just used: a regex that silently matched nothing would make
    # every implementation pass. These are the names the routes call today.
    assert called >= {"submit", "take", "peek", "stop_text", "logprobs", "stats",
                      "room_for", "limits"}, called

    # Instances, not classes: `Engine.limits` is assigned in __init__, so `hasattr` on the
    # class reports it missing and this gate would fail on a correct engine.
    instances = [_build_engine(seed=61)]
    for impl in instances:
        missing = sorted(n for n in called if not hasattr(impl, n))
        assert not missing, (
            f"{type(impl).__name__} is accepted by the routes but does not implement "
            f"{missing} — the shape of the missing `limits` (400 on every turn) and the "
            f"missing `room_for` (500 on every omitted cap)")


def test_serve_sizes_its_pools_from_the_flags_not_the_context():
    """`tilerl serve`'s own engine builder must honour --blocks / --max-ctx.

    This is the one path no benchmark reaches: every bench script constructs the
    engine itself and passes num_blocks, so `_build_engine`'s default went
    unexercised until it asked for 275 GB of KV on a 32 GB card (131072 blocks
    from the 27B's 262144-token context). The gate is that the flags win and
    that the default is still derived from the context, which is what a
    large-card target relies on.
    """
    from tilerl.engine import _graph_on
    from tilerl.kv_cache import BLOCK_TOKENS

    cfg = tiny(max_position_embeddings=4096)
    model = build_random(cfg, seed=3)
    be = get_backend()
    # The captured decode tick's padding row owns a block of its own, reserved up
    # front so the capacity the caller asked for stays whole. It is on by default on
    # CUDA, so the pool is num_blocks + 1 there and num_blocks on cpu/metal: this
    # test read `65 == 64` on both sm90 and sm70 and passed on cpu for that reason,
    # which made an intentional reservation look like an off-by-one in _fit_blocks.
    pad = int(_graph_on(be, None))

    e = build_serving_engine(cfg, model, be, blocks=64, max_ctx=256, max_batch=2, sparse_k=0)
    assert e._kv.num_blocks - pad == 64
    assert e.limits.max_total_tokens == 256, "a request must not outgrow the pool"
    assert e.limits.max_batch == 2

    d = build_serving_engine(cfg, model, be, sparse_k=0)
    assert d._kv.num_blocks - pad == (4096 * d.limits.max_batch) // BLOCK_TOKENS, (
        "the default pool must cover max_batch rows of the context — no more "
        "(bytes are the long-context limit) and no less (a full batch must fit). "
        "On CUDA this is _fit_blocks' cap; a card too small to fit even the tiny "
        "model's 2048 blocks would report less, and that is a real capacity limit "
        "rather than a bug in the sizing"
    )


def test_a_stream_that_dies_mid_decode_does_not_look_like_success():
    """`_stream` catches only (TimeoutError, RuntimeError). Anything else the engine
    raises escapes the generator AFTER the 200 header has gone out, so the client gets
    HTTP 200 with zero frames, no error frame and no [DONE] — a success status over a
    failed request. The non-stream route answers 5xx for the same conditions.

    Not hypothetical: `DataParallelEngine` had no `peek`, so `serve --devices` hit
    exactly this with an AttributeError. Measured across six exception types injected at
    the engine boundary, the two caught ones delivered 3 frames + an error frame +
    [DONE]; the other four delivered 0 frames and neither marker, all under HTTP 200.

    The gate is the CONTRACT, not the status code: a stream either terminates with
    [DONE] or says why it stopped. A 200 with neither is what a client reads as an empty
    reply. Deliberately independent of whether `_stream` grows a catch-all — a
    root-cause fix satisfies this too, and a catch-all that turns silence into a tidy
    error frame must not be the only thing that does.
    """

    class _DiesAfterOnePeek:
        """peek works once, then raises — a request that dies mid-decode."""

        def __init__(self, inner, exc):
            self._inner, self._exc, self._left = inner, exc, 1

        def peek(self, rid):
            if self._left <= 0:
                raise self._exc("injected at the engine boundary")
            self._left -= 1
            return self._inner.peek(rid)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    engine = _build_engine(seed=31)
    engine.run()
    try:
        body = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 24, "stream": True}
        for exc in (RuntimeError, ValueError, AttributeError, KeyError):
            app = create_app(_DiesAfterOnePeek(engine, exc), _ByteTokenizer())
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.post("/v1/chat/completions", json=body)
                assert "[DONE]" in r.text or '"error"' in r.text or r.status_code >= 400, (
                    f"{exc.__name__}: HTTP {r.status_code} with no [DONE] and no error "
                    f"frame — a client reads this as an empty reply. "
                    f"body={r.text[:200]!r}")
    finally:
        engine.shutdown()


def test_v1_messages_never_answers_200_for_an_engine_failure(tmp_path, monkeypatch):
    """`/v1/messages` builds the whole body before its `sse()` yields anything, so the
    route's handler still owns the status code and a failed request cannot come back as
    success. That is the structural reason `_stream` needed a catch-all and this route
    does not — measured, because the structure is what makes it true today and only a
    measurement says the structure is what ships.

    Asserts the STREAM and NON-STREAM answers agree, not a specific code. Three of the
    six types land on FastAPI's bare 500 rather than a typed error body, which is a
    different and smaller problem: the client still gets a status it can act on.
    """
    from fastapi import FastAPI

    from tilerl.messages import mount_messages

    class _Dies:
        """Raises on the first `take` that would have returned a finished request.

        `_left = 1` (let one call through, raise on the next) raced the engine loop: when
        the request finished before the route's first poll, that first `take` returned the
        completed tokens, `_run` left its wait loop, and the raising call never happened —
        200 for the non-stream arm while the stream arm, whose allowance was already spent,
        got 400. Measured at 1 in 12 runs, with exactly that signature.

        Keying on the RESULT removes the race: a `take` returning None means the request is
        still running and the route keeps waiting, which is not the moment under test; the
        first one carrying a result is. `_DiesAfterOnePeek` above keeps the count-based form
        deliberately — it measured 0 in 12, because it asserts a contract an unfired
        injection still satisfies.
        """

        def __init__(self, inner, exc):
            self._inner, self._exc = inner, exc

        def take(self, rid):
            out = self._inner.take(rid)
            if out is None:
                return None
            raise self._exc("injected at the engine boundary")

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "rec.jsonl"))
    engine = _build_engine(seed=33)
    engine.run()
    try:
        body = {"model": "tiny", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}]}
        for exc in (RuntimeError, ValueError, AttributeError, KeyError):
            app = FastAPI()
            mount_messages(app, _Dies(engine, exc), _ByteTokenizer(), "tiny")
            with TestClient(app, raise_server_exceptions=False) as c:
                codes = [c.post("/v1/messages", json={**body, "stream": s}).status_code
                         for s in (False, True)]
            assert codes[0] == codes[1], (
                f"{exc.__name__}: non-stream {codes[0]} but stream {codes[1]} — the "
                f"streaming path is diverging, which is how server.py's _stream came "
                f"to answer 200 for a failed request")
            assert codes[0] >= 400, (
                f"{exc.__name__}: HTTP {codes[0]} for an engine failure — a client "
                f"cannot tell this from a model that chose to say nothing")
    finally:
        engine.shutdown()


def test_the_record_says_which_operand_capped_the_completion(tmp_path, monkeypatch):
    """A short completion must be attributable from one row, without arithmetic.

    Six live requests came back with `stop_reason: "max_tokens"` and 1, 1, 1, 1, 4 and 8
    completion tokens. Establishing that the cap was the CLIENT's `max_tokens` and not the
    server's context budget took decoding the completion ids to see the model had been cut
    mid-word, then computing `max_total_tokens - len(prompt)` per row to show the budget
    was five digits. Both operands of `min(req.max_tokens, budget)` are in the row now, so
    the same question is one field lookup.

    `effective_max_tokens` is what `SamplingParams` actually carried: asserting only the
    two inputs would pass while a third operand nobody logged did the capping.
    """
    tok = _ByteTokenizer()
    record = tmp_path / "rec.jsonl"
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(record))
    engine = _ScriptedEngine(tok, ["</think>\n\nhello there"])
    with TestClient(create_app(engine, tok)) as client:
        r = client.post(
            "/v1/messages",
            json={"model": "m", "max_tokens": 3,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200, r.text
    rows = [json.loads(x) for x in record.read_text().splitlines() if x.strip()]
    assert rows, f"nothing recorded; file={record.read_text()[:200]!r}"
    row = rows[-1]
    for field in ("asked_max_tokens", "budget", "engine_limit", "prompt_len",
                  "effective_max_tokens", "stream", "stop_reason"):
        assert field in row, (
            f"{field!r} missing from the record, so a capped completion cannot be "
            f"attributed without re-deriving it. Row keys: {sorted(row)}"
        )
    assert row["asked_max_tokens"] == 3, row["asked_max_tokens"]
    assert row["prompt_len"] == len(row["prompt_ids"]), (
        "prompt_len disagrees with the ids it summarises"
    )
    # The cap must be explained by one of the two operands, or the row still cannot
    # answer the question it exists for.
    assert row["effective_max_tokens"] in (row["asked_max_tokens"], row["budget"]), (
        f"effective_max_tokens {row['effective_max_tokens']} is neither the asked "
        f"{row['asked_max_tokens']} nor the budget {row['budget']}: a third operand caps "
        f"completions and is not logged"
    )


@pytest.mark.parametrize("dram_bytes", [0, 12345678])
def test_serve_dram_bytes_reaches_health(dram_bytes, monkeypatch, capsys):
    """`--dram-bytes` must arrive at the tier, and `/health` must say so.

    End-to-end through `cmd_serve`, not `build_engine`: the flag crosses three hops
    (parser -> `_build_engine` -> `build_engine`) and a miss at any of them leaves the
    tier off with the command line claiming otherwise. The control arm is the same
    command without the flag, and it asserts `dram_budget` is ABSENT rather than 0 --
    `dram_bytes`, the key that was already published, is 0 in both arms because it counts
    bytes held, so an equality on it would pass with the tier off.
    """
    from tilerl import cli

    served: dict = {}
    monkeypatch.setattr(
        "uvicorn.run",
        lambda app, **kw: served.update(health=TestClient(app).get("/health").json()),
    )
    argv = ["serve", "--slots", "2", "--max-batch", "2", "--blocks", "64",
            "--max-ctx", "512", "--no-warmup", "--sparse-k", "0"]
    if dram_bytes:
        argv += ["--dram-bytes", str(dram_bytes)]
    cli.cmd_serve(cli._build_parser().parse_args(argv))
    capsys.readouterr()

    stats = served["health"]["stats"]
    dram = {k: v for k, v in stats.items() if k.startswith("dram_")}
    if dram_bytes:
        assert stats.get("dram_budget") == dram_bytes, (
            f"--dram-bytes {dram_bytes} did not reach the tier; dram_* served: {dram}"
        )
    else:
        assert "dram_budget" not in stats, (
            f"the tier is on without the flag: dram_budget={stats.get('dram_budget')}"
        )


def test_health_publishes_each_ceiling_beside_its_counter(client):
    """A counter without its ceiling cannot say whether the thing it counts is at a limit.

    `prefix_state_bytes` had no bound to compare against: `state_bytes` is set from
    `mem_get_info` at build time and is unknowable from outside, so "is the store at its
    byte ceiling" was unanswerable over HTTP. Measured on H20 card 6 once the budget was
    published: state bytes sat at 2.78 of 17.68 GiB (15.7%) while 27 entries were evicted,
    which named block pressure in one run instead of four.

    `blocks_used` is the ENGINE's counter -- blocks the prefix store retains belong to no
    live request, so it reads 0 while the pool drains. `pool_used_blocks` is the pool's own
    number and was already published; this asserts the pair stays distinguishable, because
    reading them as the same quantity is what made `blocks_used=0/512` look like an idle
    pool while 438 blocks were retained.
    """
    stats = client.get("/health").json()["stats"]
    for fill, ceiling in (("blocks_used", "blocks_total"),
                          ("prefix_state_bytes", "prefix_state_bytes_budget")):
        assert fill in stats and ceiling in stats, (
            f"{fill}/{ceiling} must both be published or pressure is unreadable; "
            f"keys: {sorted(stats)}"
        )
        assert stats[ceiling] > 0, f"{ceiling}={stats[ceiling]} is not a bound"
        assert stats[fill] <= stats[ceiling], f"{fill}={stats[fill]} > {ceiling}"
    # The pool's count is a SECOND quantity, not blocks_used by another name: the store
    # retains blocks that no request owns, so these two legitimately disagree.
    assert "pool_used_blocks" in stats, sorted(stats)
    assert stats["pool_used_blocks"] >= stats["blocks_used"], (
        f"pool_used_blocks={stats['pool_used_blocks']} below blocks_used="
        f"{stats['blocks_used']}: the pool cannot hold fewer blocks than requests own"
    )


def test_a_reply_that_carries_only_the_think_closer_is_the_answer(tmp_path, monkeypatch):
    """The 27B template opens ``<think>`` in the PROMPT, so the model's text has only
    ``</think>``. Measured on the V100 endpoint at 73bef1d: both routes returned the
    reasoning, a bare closer, then the HTML the client asked for. With the block
    opened, only what follows the closer is the reply; with thinking off, or on a
    bare turn (the byte tokenizer has no ``<think>`` token), nothing is stripped."""
    tok = _ByteTokenizer()
    engine = _ScriptedEngine(tok, ["planning\n</think>\n\n<p>hi</p>", "no block here",
                                   "bare turn"])
    with TestClient(create_app(engine, tok)) as c:
        opened = c.post("/v1/messages", json={
            "model": "m", "max_tokens": 64,
            "messages": [{"role": "user", "content": "page"}]}).json()
        assert _text_blocks(opened) == "<p>hi</p>", opened
        # The reasoning is not discarded, it moves to its own block -- Anthropic's
        # native shape. Asserted here so a regression to stripping is caught by the
        # same test that gates the closer handling.
        assert [b["thinking"] for b in opened["content"]
                if b["type"] == "thinking"] == ["planning\n"], opened
        off = c.post("/v1/messages", json={
            "model": "m", "max_tokens": 64, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "page"}]}).json()
        assert _text_blocks(off) == "no block here", off
        assert not [b for b in off["content"] if b["type"] == "thinking"], off
        bare = c.post("/v1/chat/completions", json={
            "model": "m", "max_tokens": 64,
            "messages": [{"role": "user", "content": "page"}]}).json()
        assert bare["choices"][0]["message"]["content"] == "bare turn", bare


def _chat_stream_fields(client, reply: str, max_tokens: int) -> tuple[list, list, str]:
    """(reasoning deltas, content deltas, finish) of one thinking-on chat stream."""
    resp = client.post("/v1/chat/completions", json={
        "model": "m", "max_tokens": max_tokens, "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "messages": [{"role": "user", "content": reply}]})
    assert resp.status_code == 200, resp.text
    frames = [json.loads(ln[6:]) for ln in resp.text.splitlines()
              if ln.startswith("data: ") and ln != "data: [DONE]"]
    assert not any("error" in f for f in frames), frames
    kinds = [(k, c["delta"][k]) for f in frames for c in f["choices"]
             for k in ("reasoning_content", "content") if c.get("delta", {}).get(k)]
    finish = [c["finish_reason"] for f in frames for c in f["choices"] if c.get("finish_reason")]
    return ([t for k, t in kinds if k == "reasoning_content"],
            [t for k, t in kinds if k == "content"], finish[-1])


def test_the_stream_carries_the_reasoning_as_its_own_field(tmp_path, monkeypatch):
    """With the block opened, reasoning streams as ``reasoning_content`` and the answer
    as ``content``, in that order, and neither carries the closer. Measured on the V100
    at eccac47: the server stripped the closer, the page still split on it, and a
    whole HTML reply landed in the reasoning fold with an empty bubble underneath.
    A reply the budget cuts off inside the block is reasoning only, finish ``length``."""
    tok = _ByteTokenizer()
    engine = _ScriptedEngine(tok, ["planning\n</think>\n\n<p>hi</p>", "still planning"])
    with TestClient(create_app(engine, tok)) as c:
        reasoning, content, finish = _chat_stream_fields(c, "page", 64)
        assert "".join(reasoning) == "planning\n", reasoning
        assert "".join(content) == "<p>hi</p>", content
        assert len(reasoning) >= 1 and len(content) >= 1 and finish == "stop"
        assert not any("</think>" in t for t in reasoning + content)
        reasoning, content, finish = _chat_stream_fields(c, "page", len(tok.encode("still planning")))
        assert "".join(reasoning) == "still planning" and content == [], (reasoning, content)
        assert finish == "length"


def test_an_omitted_max_tokens_gets_the_context_remainder(client, model_id, monkeypatch):
    """Omitted means "as much as fits", not 512.

    ckl, 2026-09-06: the default should be the ceiling, not a flat cap. A flat 512 ends a reply at
    ``finish_reason=length``, which reads to a client as a dropped stream. The assertion
    is on the value handed to ``sampling``, not on the reply: the tiny model's answer is
    short either way, so a test that only read the reply would stay green with the 512
    still in place.
    """
    import tilerl.server as srv

    seen: list[int] = []
    real = srv.sampling

    def spy(tokenizer, thinking, max_new, **kw):
        seen.append(max_new)
        return real(tokenizer, thinking, max_new, **kw)

    monkeypatch.setattr(srv, "sampling", spy)
    r = client.post("/v1/chat/completions",
                    json={"model": model_id, "stream": False,
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert seen, "sampling was never called"
    prompt_tokens = r.json()["usage"]["prompt_tokens"]
    # The fixture's engine: max_total_tokens=4096, 256 blocks (4096 tokens), width 1,
    # so max_total binds and the remainder is exact.
    assert seen[-1] == 4096 - prompt_tokens, (seen[-1], prompt_tokens)
    assert seen[-1] != 512, "the omitted default is still the old flat 512"


def test_an_omitted_max_output_tokens_gets_the_remainder_on_responses(client, model_id,
                                                                     monkeypatch):
    """The Responses route carries the same default, asserted through its own module.

    One route's fix is not the other's: `responses.py` reads `max_output_tokens` and
    imports `sampling` itself, so patching `server.sampling` would not observe it.
    """
    import tilerl.responses as rsp

    seen: list[int] = []
    real = rsp.sampling

    def spy(tokenizer, thinking, max_new, **kw):
        seen.append(max_new)
        return real(tokenizer, thinking, max_new, **kw)

    monkeypatch.setattr(rsp, "sampling", spy)
    r = client.post("/v1/responses",
                    json={"model": model_id, "input": "hi"})
    assert r.status_code == 200, r.text
    assert seen, "sampling was never called"
    assert seen[-1] != 512, "the omitted default is still the old flat 512"
    # Prompt length is not echoed the same way here, so bound it rather than guess:
    # the remainder must be positive and within the fixture's context.
    assert 0 < seen[-1] < 4096, seen[-1]


def test_an_explicit_max_tokens_is_still_honoured(client, model_id, monkeypatch):
    """The negative control for both tests above: a caller who asks for 8 gets 8.

    Without this, "the remainder" could be unconditional and both tests would still
    pass -- a default that overrides an explicit request is the failure this catches.
    """
    import tilerl.server as srv

    seen: list[int] = []
    real = srv.sampling
    monkeypatch.setattr(srv, "sampling",
                        lambda t, th, mn, **kw: (seen.append(mn), real(t, th, mn, **kw))[1])
    r = client.post("/v1/chat/completions",
                    json={"model": model_id, "stream": False, "max_tokens": 8,
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert seen[-1] == 8, seen


def test_a_cancel_returns_the_blocks_a_disconnected_reader_was_holding():
    """A reader that leaves must not cost a whole generation.

    Measured on the live V100 before this existed: a `/ws/chat` socket closed after
    12 delta frames left the engine generating for ~34 s more -- 1891 tokens and up
    to 104 KV blocks for nobody -- because `gen.close()` ends `_deltas`' poll loop
    and nothing told the engine. There was no cancel path at all.

    Both queues, because they fail differently. A RUNNING request is what the V100
    measured. A WAITING one is the case `_finish` cannot handle: it ends with
    `_running.remove(req)` and raises on a request that never reached the running
    set, and a waiting request is not free to drop either -- `submit` allocates its
    blocks and state slot up front, so the pool only gets them back if cancel frees
    them from that queue too.
    """
    import pytest as _pytest

    from tilerl.engine import SamplingParams

    eng = _build_engine(seed=71)
    params = SamplingParams(max_new_tokens=64, temperature=0.0, seed=0)

    # --- running arm -------------------------------------------------------
    rid = eng.submit(list(range(1, 40)), params)
    for _ in range(4):
        eng.step()
    assert any(r.req_id == rid for r in eng._running), "nothing to cancel: never ran"
    held = eng._blocks_used
    assert held > 0, "the request holds no blocks; the free assertion below is vacuous"

    assert eng.cancel(rid) is True
    assert not any(r.req_id == rid for r in eng._running), "cancel left it running"
    assert eng._blocks_used == 0, f"blocks not returned: {eng._blocks_used} of {held}"
    assert eng._slots_used == 0, "the state slot was not returned"

    # "not finished yet" and "cancelled" are the same answer unless take() raises.
    with _pytest.raises(RuntimeError, match="cancelled"):
        eng.take(rid)
    assert eng.cancel(rid) is False, "a second cancel claims it dropped something"

    # --- waiting arm: cancel before the request has ever stepped -----------
    # max_batch=1 at build: StepLimits is frozen, so this is the only way to force
    # a second request to sit in _waiting.
    cfg = tiny()
    one = build_engine(cfg, build_random(cfg, seed=71), get_backend(), num_blocks=256,
                       num_slots=4, max_batch=1, max_total_tokens=4096, sparse_k=0)
    a = one.submit(list(range(1, 40)), params)
    b = one.submit(list(range(1, 40)), params)
    one.step()
    waiting = [r.req_id for r in one._waiting]
    assert b in waiting, f"b was admitted; the waiting arm is vacuous ({waiting})"
    before = one._blocks_used

    assert one.cancel(b) is True
    assert b not in [r.req_id for r in one._waiting], "cancel left it waiting"
    # A waiting request now holds NOTHING: blocks and the state slot are taken together at
    # admission, so there is nothing to give back and `_blocks_used` must not move. #209
    # asserted the opposite because `submit` allocated up front; that is the defect the
    # queue-and-wait change removed, so the assertion inverts with it. What still matters is
    # that cancelling a waiting request leaves the RUNNING one's accounting untouched -- the
    # bug this arm exists to catch is a cancel that frees someone else's blocks.
    assert one._blocks_used == before, (
        f"cancelling an unadmitted request moved the block count: {before} -> "
        f"{one._blocks_used}")
    assert any(r.req_id == a for r in one._running), "cancel took the wrong request"

    # --- admitted arm: #209's original behaviour, at the level where it still applies ------
    # The waiting arm above used to carry this, because `submit` allocated up front. It no
    # longer can, so the assertion moves rather than disappearing: a disconnected reader's
    # blocks must still come back once the request HAS been admitted.
    # `a` still holds the only slot at max_batch=1, so nothing else can be admitted until it
    # goes -- the vacuity guard below caught exactly that.
    one.cancel(a)
    c = one.submit(list(range(1, 40)), params)
    for _ in range(50):
        one.step()
        if any(r.req_id == c for r in one._running):
            break
    assert any(r.req_id == c for r in one._running), "c was never admitted; the arm is vacuous"
    held = one._blocks_used
    slots = one._slots_used
    assert held > 0, "the admitted request holds no blocks; the assertion below is vacuous"

    assert one.cancel(c) is True
    assert not any(r.req_id == c for r in one._running), "cancel left it running"
    assert one._blocks_used < held, (
        f"an admitted request's blocks were never returned: {held} -> {one._blocks_used}")
    assert one._slots_used < slots, (
        f"an admitted request's state slot was never returned: {slots} -> {one._slots_used}")


def test_the_routes_cancel_when_the_client_hangs_up():
    """The engine cancel is only worth having if the routes call it.

    A gate on `Engine.cancel` alone passes while both call sites are missing, which
    is the state this shipped in: the WS branch closed the generator and the SSE
    generator did nothing at all. The three non-stream routes joined later: a
    client disconnect cancels the awaited task, and each route catches
    CancelledError.
    """
    import inspect

    import tilerl.messages as messages
    import tilerl.responses as responses
    from tilerl import server

    src = inspect.getsource(server)
    # Behavioral gates for the two streaming paths now drive a REAL mid-stream
    # socket/ws close (test_a_mid_stream_{sse,ws}_close_cancels...), so this
    # source-read gate only guards the route-handler shape: the SSE body goes
    # through stream_or_cancel, whose live-disconnect watcher cancels the row,
    # and the GeneratorExit teardown line stays as GC/process defense.
    assert "stream_or_cancel(request, engine, request_id," in src, (
        "the SSE body must run under the shared disconnect watcher")
    # Every cancel that runs ON the event loop (route handlers, watchers) must go
    # through to_thread: engine.cancel takes engine._lock across _release, and a
    # synchronous call freezes /health during a long tick. The detached drain's
    # backstop cancel (GeneratorExit-at-yield skips in-scope cancels) runs through
    # to_thread as well. #667 moved the WS handler's own cancel into that shared,
    # transport-neutral drain (one site serving SSE and WS), so there is no longer
    # a ws-specific line. The three generator-internal cancels are off-loop and
    # stay plain (SSE error frame + SSE/WS GeneratorExit teardown).
    on_loop = src.count("asyncio.to_thread(engine.cancel, request_id)")
    assert on_loop == 7, (
        f"7 to_thread cancel sites (3 stream_or_cancel: CancelledError, "
        f"fetch-finished-while-disconnected, the poll-wait disconnect; chat "
        f"Cancelled/timeout/RuntimeError; the shared detached drain backstop used "
        f"by both SSE and WS after #667); found {on_loop}")
    for mod in (messages, responses):
        msrc = inspect.getsource(mod)
        assert msrc.count("asyncio.to_thread(engine.cancel, rid_box[0])") == 2, (
            f"{mod.__name__}: both on-loop cancels (CancelledError and the "
            f"timeout/503 handler) must run through to_thread")
    assert src.count("engine.cancel(request_id)") == 3, (
        "only the three sync generator-internal sites may call cancel directly "
        "(SSE error frame, SSE GeneratorExit teardown, and the WS _deltas "
        "GeneratorExit teardown added in #667); they run in worker threads")
    assert "except GeneratorExit:" in src, (
        "the SSE route keeps GeneratorExit as GC/teardown defense: the live "
        "watcher is the client hang-up path, but a finalized generator must "
        "still free its row")
    for mod in (messages, responses):
        msrc = inspect.getsource(mod)
        assert "except asyncio.CancelledError:" in msrc and \
            "to_thread(engine.cancel" in msrc, (
            f"{mod.__name__}'s non-stream route must cancel on a client "
            "disconnect, off the event loop")


@pytest.mark.parametrize("state_bytes", [0, 12345678])
def test_serve_state_bytes_reaches_health(state_bytes, monkeypatch, capsys):
    """`--state-bytes` must arrive at the store, and `/health` must say so.

    Without it the tiers' pressure regime was unreachable from a command line: the budget
    came only from `mem_get_info() // 4`, which on an H20 is 17.9 GiB -- 116 snapshots at
    157 MiB, so no benchable session count evicts anything and every tier arm reads 0
    demotions. The control arm asserts the default is a DIFFERENT value rather than absent,
    since `prefix_state_bytes_budget` is published either way.
    """
    from tilerl import cli

    served: dict = {}
    monkeypatch.setattr(
        "uvicorn.run",
        lambda app, **kw: served.update(health=TestClient(app).get("/health").json()),
    )
    argv = ["serve", "--slots", "2", "--max-batch", "2", "--blocks", "64",
            "--max-ctx", "512", "--no-warmup", "--sparse-k", "0"]
    if state_bytes:
        argv += ["--state-bytes", str(state_bytes)]
    cli.cmd_serve(cli._build_parser().parse_args(argv))
    capsys.readouterr()

    got = served["health"]["stats"].get("prefix_state_bytes_budget")
    if state_bytes:
        assert got == state_bytes, f"--state-bytes {state_bytes} did not reach the store: {got}"
    else:
        assert got and got != 12345678, f"the default budget is the flag's value: {got}"


def test_a_nonstream_client_disconnect_cancels_its_request():
    """A non-stream reader that hangs up frees the row the stream paths already do.

    stream=True and /ws/chat cancel from GeneratorExit; the three non-stream routes
    (chat, messages, responses) awaited take() in a worker thread and never told the
    engine, so a disconnected client kept generating to max_new and held its slot.
    The disconnect reaches the route as asyncio.CancelledError.
    """
    import asyncio

    import httpx

    engine = _build_engine(seed=71)

    # Event-synced admission, not a wall poll (#659). The old version polled
    # engine.stats()["running"] for a fixed 5 s after dispatch; on a contended host
    # submit -> first daemon-loop tick (what actually admits the row) could elapse
    # that deadline, so the assertion fired "request never reached the engine"
    # before any disconnect was exercised. _admit runs on the loop thread exactly
    # when a row becomes running, so an event set there is the positive signal; the
    # generous bound only guards a genuinely wedged loop.
    admit_gate = {"event": threading.Event()}
    real_admit = engine._admit

    def _admit_with_event(req) -> bool:
        ok = real_admit(req)
        if ok:
            admit_gate["event"].set()
        return ok

    engine._admit = _admit_with_event
    engine.run()
    app = create_app(engine, _ByteTokenizer())

    async def scenario(path: str, body: dict) -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                     timeout=30.0) as ac:
            # Fresh gate per scenario: the follow-up post at the end also admits a
            # row and would leave a shared event already set before this dispatch.
            admitted = admit_gate["event"] = threading.Event()
            req = ac.build_request("POST", path, json=body)
            task = asyncio.ensure_future(ac.send(req))
            # Wait until THIS request's row is owned by the engine, then emulate
            # the reader leaving. Event, not a fixed wall deadline.
            assert await asyncio.to_thread(admitted.wait, 30.0), (
                f"{path}: request never reached the engine")
            task.cancel()
            # The wrapper task ends one of two ways under a cancel delivered at
            # this instant. It raises CancelledError if the cancel is scheduled
            # before completion; or it returns normally if the row finishes in
            # the SAME event-loop turn the cancel lands -- under a contended
            # loop (CI xdist) resume is delayed and the engine loop completes a
            # short row first. That race is legal: a completion that beats the
            # hang-up is supposed to return, and #14 guarantees the row/slot
            # contract below, not which outcome this HTTP task gets. Asserting
            # the raise itself is the flake (#659): it failed whenever completion
            # won the race. A non-200 return is NOT that race -- a failed row
            # surfaces 500, which is a separate defect and must fail here.
            try:
                raced = await task
            except asyncio.CancelledError:
                pass
            else:
                assert raced.status_code == 200, (
                    f"{path}: disconnect window ended {raced.status_code}, "
                    "not a clean same-turn completion")

            # cancel() drops the row under the lock; two loop polls is the bound the
            # route contract claims.
            deadline = time.monotonic() + 0.5
            while engine.stats()["running"] and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert not engine.stats()["running"], f"{path}: the row stayed running"
            assert engine.stats()["slots_used"] == 0, f"{path}: slot not released"

            # The freed slot serves another request at the same max_batch.
            follow_up = await ac.post(path, json=body)
            assert follow_up.status_code == 200, follow_up.text

    async def main():
        await scenario("/v1/chat/completions",
                       {"model": "m", "max_tokens": 512,
                        "messages": [{"role": "user", "content": "x" * 64}]})
        await scenario("/v1/messages",
                       {"model": "m", "max_tokens": 512,
                        "messages": [{"role": "user", "content": "x" * 64}]})
        await scenario("/v1/responses",
                       {"model": "m", "max_output_tokens": 512,
                        "input": [{"role": "user", "content": [{"type": "input_text",
                                                                 "text": "x" * 64}]}]})

    try:
        asyncio.run(main())
    finally:
        engine.shutdown()


def test_a_real_http_disconnect_event_cancels_without_task_cancellation():
    """uvicorn sends http.disconnect without cancelling the request task.

    A route that only awaits to_thread(completion) and catches CancelledError
    never sees that: the httpx test above cancels the whole task and covers only
    that path. All three non-stream routes must poll is_disconnected() while the
    completion worker runs, call engine.cancel(rid), and answer 499.
    """
    import asyncio
    import threading

    class _HangingEngine:
        def __init__(self) -> None:
            self.cancelled: list[int] = []
            self._release = threading.Event()

        def submit(self, input_ids, params) -> int:
            return 42

        def take(self, request_id: int):
            # Block like an unfinished row; cleanup releases the worker.
            self._release.wait(2.0)
            raise RuntimeError("cancelled")

        def cancel(self, request_id: int) -> bool:
            self.cancelled.append(request_id)
            return True

        def room_for(self, prompt_tokens: int) -> int:
            return 512

        limits = None
        def stats(self):
            return {}
        def stop_text(self, rid):
            return None
        def logprobs(self, rid):
            return []

        def __getattr__(self, name):  # rendering runs only after completion
            raise AssertionError(f"unexpected engine call: {name}")

    engines: list[_HangingEngine] = []

    async def scenario(path: str, body: bytes, delay_s: float) -> None:
        engine = _HangingEngine()
        engines.append(engine)
        app = create_app(engine, _ByteTokenizer())
        received: list[dict] = []
        peeks = {"n": 0}
        t0 = time.monotonic()

        async def receive():
            # First call delivers the body. Starlette peeks receive under a
            # cancelled scope while connected; an empty http.request stands for
            # "nothing yet" (a buffered channel), then the disconnect arrives
            # after delay_s - mid-worker, not at request start.
            n = peeks["n"]
            peeks["n"] += 1
            if n == 0:
                return {"type": "http.request", "body": body, "more_body": False}
            if delay_s and time.monotonic() - t0 < delay_s:
                return {"type": "http.request"}
            return {"type": "http.disconnect"}

        async def send(message):
            received.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": path,
                 "query_string": b"", "root_path": "",
                 "headers": [(b"content-type", b"application/json")],
                 "client": ("test", 1234), "server": ("test", 80)}
        task = asyncio.ensure_future(app(scope, receive, send))
        # Poll interval is 0.05 s; the route must react well inside 1 s.
        deadline = time.monotonic() + 1.0
        while not engine.cancelled and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert engine.cancelled == [42], (
            f"{path}: engine.cancel not called on http.disconnect within 1 s")
        await asyncio.wait_for(task, timeout=2.0)
        status = next((m.get("status") for m in received
                       if m["type"] == "http.response.start"), None)
        assert status == 499, f"{path}: expected 499 on disconnect, got {status}"
        if delay_s:
            # Polled, not spun: 0.25 s at a 0.05 s interval is ~5 fresh peeks;
            # a busy loop (a reused one-shot watcher) makes thousands.
            assert peeks["n"] - 1 <= 12, (
                f"{path}: {peeks['n'] - 1} disconnect peeks in {delay_s}s - spinning")

    async def main():
        for body3 in (
            ("/v1/chat/completions",
             b'{"model":"m","max_tokens":512,'
             b'"messages":[{"role":"user","content":"hi"}]}'),
            ("/v1/messages",
             b'{"model":"m","max_tokens":512,'
             b'"messages":[{"role":"user","content":"hi"}]}'),
            ("/v1/responses",
             b'{"model":"m","max_output_tokens":512,'
             b'"input":[{"role":"user","content":'
             b'[{"type":"input_text","text":"hi"}]}]}'),
        ):
            await scenario(body3[0], body3[1], 0.0)     # disconnect at start
            await scenario(body3[0], body3[1], 0.25)    # delayed, mid-worker

    try:
        asyncio.run(main())
    finally:
        # Release orphaned completion workers so the executor joins at teardown.
        for engine in engines:
            engine._release.set()


def test_await_or_cancel_polls_disconnect_at_interval_not_spin():
    """The disconnect watcher is awaited FRESH once per 0.05 s tick. A reused
    one-shot task resolves False while connected and then busy-waits forever;
    this bounds the number of is_disconnected() calls over 0.25 s.
    """
    import threading

    from tilerl.server import ClientDisconnected, await_or_cancel

    class _Req:
        def __init__(self) -> None:
            self.peeks = 0
            self.t0 = time.monotonic()

        async def is_disconnected(self) -> bool:
            self.peeks += 1
            return time.monotonic() - self.t0 >= 0.25

    class _Eng:
        def __init__(self) -> None:
            self.cancelled: list[int] = []
            self._go = threading.Event()

        def cancel(self, rid: int) -> bool:
            self.cancelled.append(rid)
            return True

        def take(self, rid: int):
            self._go.wait(2.0)
            raise RuntimeError("cancelled")

    async def main() -> None:
        req, eng = _Req(), _Eng()
        with pytest.raises(ClientDisconnected):
            await asyncio.wait_for(
                await_or_cancel(req, eng, [7], eng.take, 7), timeout=1.0)
        assert eng.cancelled == [7]
        # 0.25 s / 0.05 s ~= 5 ticks; a spin loop makes thousands.
        assert req.peeks <= 12, f"{req.peeks} is_disconnected peeks in 0.25 s - spinning"
        eng._go.set()

    import asyncio

    asyncio.run(main())


@pytest.mark.parametrize("path,body,status", [
    ("/v1/chat/completions",
     {"model": "m", "max_tokens": 4,
      "messages": [{"role": "user", "content": "hi"}]}, 504),
    ("/v1/messages",
     {"model": "m", "max_tokens": 4,
      "messages": [{"role": "user", "content": "hi"}]}, 503),
    ("/v1/responses",
     {"model": "m", "max_output_tokens": 4,
      "input": [{"role": "user", "content": [{"type": "input_text",
                                              "text": "hi"}]}]}, 503),
])
def test_completion_timeout_cancels_the_still_running_row(path, body, status):
    """A 504/503 from a completion TIMEOUT returns to the client but the engine
    row is still generating. The route must cancel it, else slot and blocks
    run to max_new_tokens for nobody (audit finding 3)."""
    class _TimeoutEngine:
        cancelled: list[int] = []

        def submit(self, input_ids, params) -> int:
            return 7

        def room_for(self, prompt_tokens: int) -> int:
            return 16

        def take(self, request_id: int):
            raise TimeoutError(f"request {request_id} timed out")

        def cancel(self, request_id: int) -> bool:
            self.cancelled.append(request_id)
            return True  # the row was live: cancel dropped it

        def stats(self) -> dict:
            return {}

        def stop_text(self, request_id: int):
            return None

        def logprobs(self, request_id: int):
            return []

    engine = _TimeoutEngine()
    client = TestClient(create_app(engine, _ByteTokenizer()))
    r = client.post(path, json=body)
    assert r.status_code == status, (path, r.status_code, r.text)
    assert engine.cancelled == [7], (path, "the timed-out row was never cancelled")


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions",
     {"model": "m", "max_tokens": 4,
      "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/messages",
     {"model": "m", "max_tokens": 4,
      "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/responses",
     {"model": "m", "max_output_tokens": 4,
      "input": [{"role": "user", "content": [{"type": "input_text",
                                              "text": "hi"}]}]}),
])
def test_completion_failure_cancel_is_a_noop_not_a_double_release(path, body):
    """RuntimeError (engine already failed the row, e.g. pool exhausted): the
    route still calls cancel, but the row is gone so it returns False and no
    block is double-freed."""
    freed: list[int] = []

    class _FailedEngine:
        def submit(self, input_ids, params) -> int:
            return 9

        def room_for(self, prompt_tokens: int) -> int:
            return 16

        def take(self, request_id: int):
            raise RuntimeError(f"request {request_id} failed: pool exhausted")

        def cancel(self, request_id: int) -> bool:
            freed.append(request_id)
            return False  # already gone: must not release twice

        def stats(self) -> dict:
            return {}

        def stop_text(self, request_id: int):
            return None

        def logprobs(self, request_id: int):
            return []

    engine = _FailedEngine()
    client = TestClient(create_app(engine, _ByteTokenizer()))
    r = client.post(path, json=body)
    assert r.status_code in (500, 503), (path, r.status_code, r.text)
    assert freed == [9], "cancel called once as a no-op"


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions",
     {"model": "m", "max_tokens": 4,
      "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/messages",
     {"model": "m", "max_tokens": 4,
      "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/responses",
     {"model": "m", "max_output_tokens": 4,
      "input": [{"role": "user", "content": [{"type": "input_text",
                                              "text": "hi"}]}]}),
])
def test_a_successful_completion_never_cancels(path, body):
    """The cancel additions must not touch the success path."""
    class _OkEngine:
        cancelled: list[int] = []

        def submit(self, input_ids, params) -> int:
            return 3

        def room_for(self, prompt_tokens: int) -> int:
            return 16

        def take(self, request_id: int):
            return [10, 11, 12, 13]

        def cancel(self, request_id: int) -> bool:
            self.cancelled.append(request_id)
            return True

        def stats(self) -> dict:
            return {}

        def stop_text(self, request_id: int):
            return None

        def logprobs(self, request_id: int):
            return []

    engine = _OkEngine()
    client = TestClient(create_app(engine, _ByteTokenizer()))
    r = client.post(path, json=body)
    assert r.status_code == 200, (path, r.status_code, r.text)
    assert engine.cancelled == [], "success must not cancel"


EFFORT_PATHS = [
    ("/v1/chat/completions",
     {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"}),
    ("/v1/messages",
     {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
      "output_config": {"effort": "high"}}),
    ("/v1/responses",
     {"input": "hi", "reasoning": {"effort": "none"}}),
]


@pytest.mark.parametrize(("path", "body", "cap"), [
    ("/v1/chat/completions",
     {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"}, 512),
    ("/v1/messages",
     {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
      "output_config": {"effort": "high"}}, 8192),
    # effort:"none" closes the think block in the prompt; sampling then
    # carries no cap (sampling drops max_think_tokens when thinking is off),
    # so the engine sees None on that route too.
    ("/v1/responses",
     {"input": "hi", "reasoning": {"effort": "none"}}, None),
])
def test_reasoning_effort_caps_the_engine_on_every_route(tmp_path, monkeypatch,
                                                          path, body, cap):
    """Finding 14: only chat mapped effort to the engine cap; messages/responses
    wrote effort into prompt text and sampled with no max_think_tokens. All three
    now use the shared prompt.think_cap mapping."""
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "r.jsonl"))
    tok = _ByteTokenizer()
    eng = _ScriptedEngine(tok, ["ok"])
    post = dict(body)
    if cap:
        # A positive cap presupposes thinking is on; without an explicit switch
        # the ByteTokenizer dev path leaves thinking bare and the cap off.
        post["chat_template_kwargs"] = {"enable_thinking": True}
    with TestClient(create_app(eng, tok)) as c:
        r = c.post(path, json=post)
    assert r.status_code == 200, (path, r.text)
    assert eng.params[-1].max_think_tokens == cap, (path, eng.params[-1])


@pytest.mark.parametrize(("path", "body"), EFFORT_PATHS)
def test_no_effort_input_means_no_engine_cap(tmp_path, monkeypatch, path, body):
    """An absent effort must reach sampling as max_think_tokens=None on every
    route, not default to some budget."""
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "r.jsonl"))
    body = {k: v for k, v in body.items()
            if k not in ("reasoning_effort", "output_config", "reasoning")}
    tok = _ByteTokenizer()
    eng = _ScriptedEngine(tok, ["ok"])
    with TestClient(create_app(eng, tok)) as c:
        r = c.post(path, json=body)
    assert r.status_code == 200, (path, r.text)
    assert eng.params[-1].max_think_tokens is None, path


@pytest.mark.parametrize(("path", "body"), EFFORT_PATHS)
def test_unknown_effort_is_refused_on_every_route(tmp_path, monkeypatch, path, body):
    """Same refusal everywhere: accepting an effort the engine does not cap
    silently answered the request as if it applied."""
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "r.jsonl"))
    body = dict(body)
    if "reasoning_effort" in body:
        body["reasoning_effort"] = "turbo"
    elif "output_config" in body:
        body["output_config"] = {"effort": "turbo"}
    else:
        body["reasoning"] = {"effort": "turbo"}
    tok = _ByteTokenizer()
    eng = _ScriptedEngine(tok, ["ok"])
    with TestClient(create_app(eng, tok)) as c:
        r = c.post(path, json=body)
    assert r.status_code == 400, (path, r.status_code, r.text)
    assert "reasoning_effort" in r.text, (path, r.text)
    assert not eng.params, "a refused effort must never submit"


def test_chat_effort_render_is_byte_identical():
    """Finding 14 changes the chat CAP, not its prompt text: the chat
    vocabulary (high) never matched the template's xhigh sentence, and low
    always did. The shared mapping must not start rendering new prose."""
    from tilerl.server import ChatMessage, _render_chat

    high = _render_chat([ChatMessage(role="user", content="hi")],
                         reasoning_effort="high")
    assert "Reasoning effort is set to" not in high
    low = _render_chat([ChatMessage(role="user", content="hi")],
                        reasoning_effort="low")
    assert "Reasoning effort is set to low." in low


# ---------------------------------------------------------------------------
# Audit findings 4/10: a REAL mid-stream socket close must reach engine.cancel.
#
# The source-only gate (test_the_routes_cancel_when_the_client_hangs_up) cannot
# see the defect these replace: nothing actually disconnects, so Starlette's
# send() never raises and the generator's GeneratorExit/websocket close path
# never runs. The mechanism here matches production (verified by #598 for the
# non-stream routes): a hand ASGI transport whose send() simulates the broken
# pipe Starlette turns into ClientDisconnect / WebSocketDisconnect(1006),
# delivered while the completion worker is still mid-stream.
# ---------------------------------------------------------------------------


class _MidStreamEngine:
    """Reveals half the reply once, then BLOCKS on peek until cancel/finish.

    The block is what makes the gate real: cancel must arrive while the row is
    still live, holding the blocks/slot submit allocated. cancel frees them, so
    the assertion checks release, not just a call recorded on a dead row.
    """

    def __init__(self, tok, text: str):
        ids = tok.encode(text)
        self._half, self._full = ids[: len(ids) // 2], ids
        self._peeks = 0
        self._gate = threading.Event()
        # Test synchronization replaces wall-clock sleeps: submit done, the
        # first content frame has been produced, and cancel has fully run.
        self.submitted = threading.Event()
        self.frame_sent = threading.Event()
        self.cancel_finished = threading.Event()
        self.cancelled: list[int] = []
        self.blocks_used = self.slots_used = 0
        self.params: list = []

    def submit(self, input_ids, params=None) -> int:
        self.blocks_used += 4
        self.slots_used += 1
        self.params.append(params)
        self.submitted.set()
        return 7

    def peek(self, request_id: int):
        self._peeks += 1
        if self._peeks == 1:
            self.frame_sent.set()
            return self._half
        if self._peeks == 2:
            return self._full
        # Bounded wait so a missing cancel fails the test instead of hanging
        # the worker thread for the process lifetime.
        self._gate.wait(2.0)
        return None if request_id in self.cancelled else self._full

    def take(self, request_id: int):
        if request_id in self.cancelled:
            raise RuntimeError("cancelled")
        self._gate.wait(2.0)
        return None if request_id in self.cancelled else self._full

    def stop_text(self, request_id: int):
        return None

    def cancel(self, request_id: int) -> bool:
        if request_id in self.cancelled:
            return False
        self.cancelled.append(request_id)
        self.blocks_used = self.slots_used = 0
        self._gate.set()
        self.cancel_finished.set()
        return True

    def room_for(self, prompt_tokens: int) -> int:
        return 512

    def stats(self) -> dict:
        return {}


def _uvicorn_server(engine, tok, wrap_app=None):
    """A real uvicorn loop over a custom engine: the ONLY transport under
    which an SSE socket close reaches engine.cancel. httptools reads the EOF,
    posts http.disconnect, Starlette's disconnect watcher cancels the
    stream task group, the async generator's aclose runs in the worker
    thread and throws GeneratorExit into _stream -- no hand ASGI harness
    reproduces that chain (a send() that raises orphans the generator for
    nondeterministic GC).

    ``wrap_app`` optionally wraps the ASGI app (used to widen the loop's
    default executor for many-socket cancel gates)."""
    import socket

    import uvicorn

    app = create_app(engine, tok)
    if wrap_app is not None:
        app = wrap_app(app)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(400):
        if server.started:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("uvicorn did not start")
    return server, port


def test_a_mid_stream_sse_close_cancels_and_releases_the_row():
    """F4 behavioral gate: a real socket close after the first SSE content
    frame, while the worker is blocked mid-stream, must reach _stream's
    `except GeneratorExit: engine.cancel(...)` and return the held
    blocks/slot -- not source-read text and not a hang (pre-finding state)."""
    import socket

    tok = _ByteTokenizer()
    eng = _MidStreamEngine(tok, "streaming reply text")
    server, port = _uvicorn_server(eng, tok)
    try:
        payload = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                            "stream": True, "max_tokens": 64})
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
            f"\r\n{payload}").encode()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(request)
            buf = b""
            # Close only after the first content delta actually went out; any
            # earlier close would prove nothing about a MID-stream disconnect.
            while b'"content"' not in buf:
                chunk = s.recv(4096)
                assert chunk, "the stream produced no content frame before EOF"
                buf += chunk
        # Socket closed here: httptools EOF -> disconnect cancels the stream
        # task -> GeneratorExit in _stream -> engine.cancel.
        deadline = time.monotonic() + 5.0
        while not eng.cancelled and time.monotonic() < deadline:
            time.sleep(0.02)
        assert eng.cancelled == [7], "SSE GeneratorExit never reached engine.cancel"
        assert eng.blocks_used == 0 and eng.slots_used == 0, (
            f"the live row kept its allocation: {eng.blocks_used} blocks, "
            f"{eng.slots_used} slots")
    finally:
        server.should_exit = True


def _raw_ws_app(engine, ask: bytes):
    import asyncio

    app = create_app(engine, _ByteTokenizer())
    messages = [
        {"type": "websocket.connect"},
        {"type": "websocket.receive", "bytes": None, "text": ask.decode()},
    ]
    idx = {"n": 0}
    sent: list[dict] = []

    async def receive():
        # connect, then the ask; afterwards the client is GONE. The handler is
        # blocked in the generator thread and never polls receive again --
        # production reaches this as an OSError on the next send.
        idx["n"] += 1
        if idx["n"] <= len(messages):
            return messages[idx["n"] - 1]
        await asyncio.sleep(10)
        return {"type": "websocket.disconnect", "code": 1006}

    async def send(message):
        sent.append(message)
        # accept, then the first content frame out; the NEXT send hits the dead
        # socket and Starlette raises WebSocketDisconnect(1006).
        if len(sent) >= 3:
            raise OSError("broken pipe")

    scope = {"type": "websocket", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "scheme": "ws", "path": "/ws/chat", "query_string": b"", "root_path": "",
             "headers": [], "client": ("test", 1), "server": ("test", 80),
             "subprotocols": None}
    return app, scope, receive, send, engine, sent


def test_a_mid_stream_ws_close_cancels_and_releases_the_row():
    """F10 behavioral gate: the socket dies while /ws/chat is awaiting the next
    delta. The WebSocketDisconnect branch must gen.close() (GeneratorExit fires
    the in-generator cancel) and engine.cancel; blocks/slot return."""
    import asyncio

    ask = b'{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
    app, scope, receive, send, eng, sent = _raw_ws_app(
        _MidStreamEngine(_ByteTokenizer(), "ws reply"), ask)

    async def scenario():
        await app(scope, receive, send)  # WebSocketDisconnect is caught, not raised

    asyncio.run(scenario())
    assert eng.cancelled == [7], (
        "websocket WebSocketDisconnect never reached engine.cancel")
    assert eng.blocks_used == 0 and eng.slots_used == 0, (
        f"the live row kept its allocation: {eng.blocks_used} blocks, "
        f"{eng.slots_used} slots")
    assert [m["type"] for m in sent].count("websocket.accept") == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path,body,envelope", [
    ("/v1/chat/completions",
     {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
     ("error",)),
    ("/v1/messages",
     {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
     ("error",)),
    ("/v1/responses",
     {"input": [{"role": "user", "content": [{"type": "input_text",
                                               "text": "hi"}]}],
      "max_output_tokens": 4},
     ("error",)),
])
def test_engine_overloaded_is_a_503_overloaded_body(tmp_path, monkeypatch,
                                                      path, body, envelope,
                                                      stream):
    """Finding 17 route half (#633): submit raises EngineOverloaded
    synchronously at queue capacity. Every route answers 503 with the
    overloaded type and cap/inflight ints, stream and non-stream alike
    (the rejection happens before any SSE header), and never 429."""
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "r.jsonl"))
    from tilerl.engine import EngineOverloaded

    class _Saturated:
        def submit(self, input_ids, params=None) -> int:
            raise EngineOverloaded(
                "engine is saturated: 8 in-flight requests and the cap is 8 "
                "(running + waiting); retry later")

        def room_for(self, prompt_tokens: int) -> int:
            return 64

        def cancel(self, request_id: int) -> bool:
            return False  # nothing was enqueued; cancel must be a no-op

        def stats(self) -> dict:
            return {}

    payload = dict(body)
    if stream:
        payload["stream"] = True
    r = TestClient(create_app(_Saturated(), _ByteTokenizer())).post(path,
                                                                     json=payload)
    assert r.status_code == 503, (path, stream, r.status_code, r.text)
    assert r.headers.get("retry-after") is None
    err = r.json()[envelope[0]]
    assert err["type"] == "overloaded_error", (path, stream, err)
    assert err["inflight"] == 8 and err["cap"] == 8, (path, stream, err)
    assert "cap is 8" in err["message"], err
    assert all(k not in err for k in ("retry_after", "retryAfter")), err


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions",
     {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
    ("/v1/messages",
     {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
    ("/v1/responses",
     {"input": [{"role": "user", "content": [{"type": "input_text",
                                              "text": "hi"}]}],
      "max_output_tokens": 4}),
])
def test_a_plain_runtime_error_stays_api_error(tmp_path, monkeypatch, path, body):
    """Only EngineOverloaded maps to overloaded_error; a RequestFailed or
    other RuntimeError keeps the generic api_error 503 body."""
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "r.jsonl"))

    class _OtherFailure:
        def submit(self, input_ids, params=None) -> int:
            return 3

        def room_for(self, prompt_tokens: int) -> int:
            return 64

        def take(self, request_id: int):
            raise RuntimeError(f"request {request_id} failed: pool exhausted")

        def cancel(self, request_id: int) -> bool:
            return True

        def stop_text(self, request_id: int):
            return None

        def logprobs(self, request_id: int):
            return []

        def stats(self) -> dict:
            return {}

    r = TestClient(create_app(_OtherFailure(), _ByteTokenizer())).post(
        path, json=body)
    assert r.status_code in (500, 503), r.text
    assert r.json()["error"]["type"] == "api_error", r.text


class _LockParkEngine(_MidStreamEngine):
    """cancel() parks on an engine-held lock: models a slow step tick holding
    engine._lock for seconds while a late-stream disconnect arrives."""

    def __init__(self, tok, text, *, cancel_park_s: float = 2.0):
        super().__init__(tok, text)
        import threading as _t
        self.cancel_lock = _t.Lock()
        self._cancel_park_s = cancel_park_s
        self.cancel_started = _t.Event()

    def cancel(self, request_id: int) -> bool:
        if request_id in self.cancelled:
            return False
        self.cancel_started.set()
        with self.cancel_lock:  # held by the test across the disconnect window
            time.sleep(self._cancel_park_s)
        self.cancelled.append(request_id)
        self.blocks_used = self.slots_used = 0
        self._gate.set()
        self.cancel_finished.set()
        return True


def test_a_late_sse_disconnect_does_not_freeze_the_event_loop():
    """F4 device-verification follow-up: engine.cancel takes engine._lock; a
    late disconnect landing on a slow step tick (lock held seconds) used to
    call cancel synchronously ON the event loop, freezing /health for every
    other connection. cancel must run off the loop (to_thread), so the loop
    stays responsive while the cancel is waiting for the lock.

    The resource row is NOT asserted released within a wall-clock bound: the
    tick owns the lock and the engine cannot release faster than the next
    tick -- only (1) cancel was recorded/attempted, (2) the loop answered
    /health while cancel blocked, (3) blocks/slot hit zero once the parked
    critical section ended (the next-tick flag)."""
    import socket

    tok = _ByteTokenizer()
    eng = _LockParkEngine(tok, "late disconnect reply", cancel_park_s=2.0)
    eng.cancel_lock.acquire()  # the "slow step tick": held until the probe ends
    server, port = _uvicorn_server(eng, tok)
    try:
        payload = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                            "stream": True, "max_tokens": 64})
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
            f"\r\n{payload}").encode()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(request)
            # Event-synced, not a wall-clock race: the engine marks the first
            # frame produced; reading until the marker then proves it actually
            # crossed the real socket before this "late" disconnect.
            assert eng.frame_sent.wait(10.0), "engine never produced a first frame"
            buf = b""
            while b'"content"' not in buf:
                chunk = s.recv(4096)
                assert chunk, "no content frame before the close"
                buf += chunk
        # Disconnect delivered; stream_or_cancel must dispatch cancel to a
        # worker thread, leaving the event loop free.
        assert eng.cancel_started.wait(10.0), "cancel was never attempted"
        # A probe ON the same uvicorn event loop must answer while cancel is
        # still parked behind the held lock. 0.5 s bound, not the 2 s park.
        probe = socket.create_connection(("127.0.0.1", port), timeout=5)
        probe.sendall(b"GET /health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
        t0 = time.monotonic()
        answer = b""
        while b"\r\n\r\n" not in answer:
            part = probe.recv(4096)
            assert part, "/health never answered"
            answer += part
        elapsed = time.monotonic() - t0
        probe.close()
        assert b" 200 " in answer.split(b"\r\n", 1)[0], answer[:80]
        assert elapsed < 0.5, (
            f"event loop froze {elapsed:.2f}s waiting for engine.cancel's lock")
        # Release the parked critical section: the queued cancel completes and
        # the next-tick cleanup flag drains the allocation. Wait on the
        # completion event, not a poll deadline.
        eng.cancel_lock.release()
        assert eng.cancel_finished.wait(10.0), "queued cancel never finished"
        assert eng.cancelled == [7] and eng.blocks_used == 0 \
            and eng.slots_used == 0
    finally:
        if eng.cancel_lock.locked():
            eng.cancel_lock.release()
        server.should_exit = True


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions",
     {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}),
    ("/v1/messages",
     {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}),
    ("/v1/responses",
     {"input": [{"role": "user", "content": [{"type": "input_text",
                                               "text": "hi"}]}],
      "max_output_tokens": 64}),
])
def test_a_nonstream_disconnect_does_not_freeze_the_event_loop(tmp_path,
                                                                monkeypatch,
                                                                path, body):
    """await_or_cancel is the non-stream analog of stream_or_cancel: its live
    disconnect branch used to call the lock-taking engine.cancel on the event
    loop, so a disconnect in a long tick froze /health identically. All three
    non-stream routes must cancel off the loop. Mirrors the SSE liveness gate
    with the same _LockParkEngine; the parked cancel is still pending while
    /health on the same loop must answer."""
    import socket

    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "r.jsonl"))
    tok = _ByteTokenizer()
    eng = _LockParkEngine(tok, "nonstream reply", cancel_park_s=2.0)
    eng.cancel_lock.acquire()  # the slow step tick, held across the disconnect
    server, port = _uvicorn_server(eng, tok)
    try:
        payload = json.dumps(body).encode()
        request = (
            f"POST {path} HTTP/1.1\r\nHost: t\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
            f"\r\n").encode() + payload
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(request)
            # Wait until the request is enqueued (take is blocked in the
            # worker), then close to deliver http.disconnect. Event, not sleep.
            assert eng.submitted.wait(10.0), f"{path}: request never submitted"
        assert eng.cancel_started.wait(10.0), (
            f"{path}: await_or_cancel never dispatched engine.cancel")
        probe = socket.create_connection(("127.0.0.1", port), timeout=5)
        probe.sendall(b"GET /health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
        t0 = time.monotonic()
        answer = b""
        while b"\r\n\r\n" not in answer:
            part = probe.recv(4096)
            assert part, "/health never answered"
            answer += part
        elapsed = time.monotonic() - t0
        # Event-synced primary assertion, not a wall bound: when /health's
        # headers come back the off-loop engine.cancel must STILL be parked on
        # the held lock. If the route had called cancel on the event loop, the
        # loop could not serve /health until the 2 s lock released, so cancel
        # would already be finished here. This proves liveness independent of
        # host scheduling jitter (CI xdist measured 0.57 s of pure scheduling
        # delay with the loop otherwise healthy -- #659 family).
        assert b" 200 " in answer.split(b"\r\n", 1)[0], answer[:80]
        assert not eng.cancel_finished.is_set(), (
            f"{path}: /health did not answer until cancel's lock released -- "
            "the loop was serialized behind engine.cancel")
        # Sanity cap only: a loop genuinely frozen behind the lock answers at
        # ~cancel_park_s (2 s, after the release). Anything clearly below that is
        # scheduling jitter, not a freeze; do not hard-code a tight host-specific
        # number that a contended CI runner misses.
        assert elapsed < eng._cancel_park_s - 0.25, (
            f"{path}: /health took {elapsed:.2f}s, within {eng._cancel_park_s:.2f}s of "
            "the parked lock -- serialized, not jitter")
        probe.close()
        eng.cancel_lock.release()
        assert eng.cancel_finished.wait(10.0), f"{path}: queued cancel never finished"
        assert eng.cancelled and eng.blocks_used == 0 and eng.slots_used == 0
    finally:
        if eng.cancel_lock.locked():
            eng.cancel_lock.release()
        server.should_exit = True


def test_a_disconnect_with_an_already_failed_worker_retrieves_its_exception():
    """#637 follow-up: engine.cancel runs off the loop in a worker thread, which
    opens a window: disconnect is observed, the route awaits to_thread(cancel),
    and WHILE that await is in flight the completion worker finishes with
    RequestFailed. The route then raises ClientDisconnected without calling
    worker.result(), so a done-callback attached only when `not worker.done()`
    never runs -> "Task exception was never retrieved". The consume callback has
    to be attached before the worker can finish.

    Timing is event-synchronized, not slept: cancel() signals that the route is
    inside the cancel await, then sleeps long enough that take()'s failure (and
    the worker task's death) lands strictly before the cancel await returns and
    the finally clause runs."""
    import asyncio

    from tilerl.server import ClientDisconnected

    class _FailInsideCancel:
        def __init__(self):
            self.cancel_started = threading.Event()
            self.cancelled: list[int] = []

        def submit(self, input_ids, params=None) -> int:
            return 7

        def take(self, request_id: int):
            # Block the completion worker until the route has entered
            # to_thread(cancel); failing before that would be retrieved by the
            # normal worker.result() path, not the leak under test.
            if not self.cancel_started.wait(2.0):
                raise RuntimeError("cancel never started")
            raise RuntimeError("RequestFailed: cancelled: the reader disconnected")

        def cancel(self, request_id: int) -> bool:
            if request_id not in self.cancelled:
                self.cancelled.append(request_id)
            self.cancel_started.set()
            # Stay inside the cancel await until the take thread has failed and
            # its worker task has settled as done-with-exception.
            time.sleep(0.5)
            return True

        def room_for(self, prompt_tokens: int) -> int:
            return 512

        def stats(self) -> dict:
            return {}

    async def scenario(errors):
        import gc

        engine = _FailInsideCancel()
        app = create_app(engine, _ByteTokenizer())
        peeks = {"n": 0}

        async def receive():
            peeks["n"] += 1
            if peeks["n"] == 1:
                return {"type": "http.request",
                        "body": b'{"messages":[{"role":"user","content":"hi"}]}',
                        "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http",
                 "path": "/v1/chat/completions", "query_string": b"",
                 "root_path": "",
                 "headers": [(b"content-type", b"application/json")],
                 "client": ("t", 1), "server": ("t", 80)}
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, ctx: errors.append(ctx))
        with contextlib.suppress(ClientDisconnected, Exception):
            await app(scope, receive, send)
        # The leak is reported when the failed Task is destroyed without its
        # exception retrieved, i.e. on GC after the stack frame drops it -- not
        # when it finishes. Force collection and give the handler a tick.
        gc.collect()
        await asyncio.sleep(0.2)

    errors: list[dict] = []
    asyncio.run(scenario(errors))
    leaked = [e for e in errors
              if "Task exception was never retrieved" in str(e.get("message", ""))]
    assert not leaked, [str(e["message"]) for e in leaked]


def test_sse_body_generator_exit_cancel_runs_off_the_event_loop():
    """The sync SSE body's GeneratorExit calls engine.cancel. Dropping the body
    on the event loop (which happened when stream_or_cancel's frame tore down)
    gen_close()d it THERE, so a lock-held cancel froze the loop for the seconds
    a step tick owned engine._lock -- the late-SSE 1-in-10 /health stall.

    The disconnect must finalize the body via to_thread(body.close), so (1) the
    GeneratorExit cancel runs on a worker thread, never the loop, and (2) a
    parked cancel does not stall an in-flight /health-style loop callback.
    """
    import threading

    from tilerl.server import stream_or_cancel

    loop_id = threading.get_ident()
    gen_threads: list[int] = []
    ge_started = threading.Event()
    release = threading.Event()
    ge_done = threading.Event()

    fetch_proceed = threading.Event()

    def body():
        # frame-1 yields INSIDE the try. The second fetch then blocks on
        # fetch_proceed (the real body polls peek/take); the test releases it
        # after the disconnect, mirroring the poll noticing the cancel. The body
        # resumes and yields frame-2 while still paused INSIDE the try;
        # stream_or_cancel sees the disconnect before launching a third fetch,
        # so finally closes a body suspended at that yield on a worker thread,
        # injecting GeneratorExit there.
        try:
            yield "frame-1"
            fetch_proceed.wait(5.0)
            yield "frame-2"
        except GeneratorExit:
            # Runs at close(). Record the thread, then park as a lock-held
            # cancel would: if this is the event loop thread, the loop probe in
            # the test cannot run until release.
            gen_threads.append(threading.get_ident())
            ge_started.set()
            release.wait(5.0)
            ge_done.set()
            raise

    class _Req:
        def __init__(self):
            self._disc = threading.Event()

        async def is_disconnected(self):
            return self._disc.is_set()

    class _Engine:
        def cancel(self, rid):
            return False  # the explicit disconnect cancel; GeneratorExit does the work

    async def scenario():
        req = _Req()
        gen = stream_or_cancel(req, _Engine(), 7, body())

        async def consume():
            with contextlib.suppress(Exception):
                async for _ in gen:
                    pass  # StreamingResponse keeps pulling; that drives the poll loop

        consumer = asyncio.ensure_future(consume())
        # wait for frame-1 and the second fetch to park on fetch_proceed
        await asyncio.sleep(0.08)
        # hang up while that fetch is in flight, then release it: the poll notices
        # the cancel, next() returns frame-2, stream_or_cancel re-checks
        # disconnect and returns, and finally closes the paused body off-loop.
        req._disc.set()
        await asyncio.sleep(0.08)
        fetch_proceed.set()
        # let the released worker return frame-2 and finally's to_thread(close)
        # run; the assertion itself is event-driven below.
        await asyncio.sleep(0.05)
        assert ge_started.wait(2.0), "GeneratorExit never fired"
        # GeneratorExit is parked on a WORKER thread; the event loop must still
        # run a callback while that worker holds the (modelled) cancel lock.
        t0 = time.monotonic()
        await asyncio.sleep(0.12)
        loop_responsive = (time.monotonic() - t0) < 0.5
        release.set()
        assert ge_done.wait(2.0)
        await asyncio.sleep(0.05)
        with contextlib.suppress(Exception):
            await consumer
        return loop_responsive

    responsive = asyncio.run(scenario())
    assert gen_threads, "body GeneratorExit did not run"
    assert all(t != loop_id for t in gen_threads), (
        f"GeneratorExit ran on the event loop thread ({loop_id}), threads {gen_threads}")
    assert responsive, "event loop stalled while GeneratorExit cancel parked"


class _LivenessEngine:
    """Minimal engine for the /health stall gate: stats plus a scripted
    (live, stuck_secs) liveness answer."""

    def __init__(self, verdict):
        self._verdict = verdict

    def stats(self):
        return {"running": 1, "waiting": 0, "finished": 0}

    def liveness(self, stuck_after_s):
        return self._verdict


def test_health_returns_503_when_the_step_loop_is_stalled():
    """A wedged device forward freezes the step loop but stats() keeps serving
    the last snapshot, so /health used to answer 200 on a dead server. A
    liveness() verdict of (False, stuck_secs) must produce 503 + stuck_secs."""
    with TestClient(create_app(_LivenessEngine((False, 91.25)), _ByteTokenizer())) as c:
        r = c.get("/health")
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["status"] == "unhealthy"
    assert body["stuck_secs"] == 91.25
    assert body["stats"]["running"] == 1


def test_health_stays_ok_when_the_step_loop_is_live_or_idle():
    # live even with active requests
    with TestClient(create_app(_LivenessEngine((True, 0.2)), _ByteTokenizer())) as c:
        r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok", r.text


def test_engine_liveness_idle_is_live_and_active_stall_is_not(monkeypatch):
    """Direct on the real Engine: idle (no running/waiting) is always live even
    if the clock is ancient; with an active request a stale last-progress
    timestamp reports stuck, and a fresh one is live."""
    import time as _time


    cfg = tiny()
    eng = build_engine(cfg, build_random(cfg, seed=71), get_backend(),
                       num_blocks=32, num_slots=4, max_batch=4, max_total_tokens=4096)
    try:
        # idle: no requests, ancient timestamp must still be live (quiet server)
        eng._last_progress_ts = _time.perf_counter() - 9999.0
        live, stuck = eng.liveness(60.0)
        assert live is True and stuck == 0.0

        # active request: ancient timestamp -> stuck by roughly that much
        eng._running.append(object())
        live, stuck = eng.liveness(60.0)
        assert live is False and stuck > 60.0

        # active request, just progressed -> live
        eng._last_progress_ts = _time.perf_counter()
        live, stuck = eng.liveness(60.0)
        assert live is True and 0.0 <= stuck <= 60.0
    finally:
        eng._running.clear()
        eng.shutdown()


def test_engine_liveness_first_tick_after_long_idle_is_live():
    """Regression for the quiet->traffic false 503: _last_progress_ts used to
    move only when a forward RETURNED, so the first request after an idle gap
    longer than the threshold read active with stuck = the whole idle gap and
    503'd WHILE its first forward was in flight. The timestamp must refresh at
    the START of the first non-idle tick (before the forward), not just at its
    end. A plain timestamp check after step() cannot see this -- the end
    refresh hides it -- so the assertion runs INSIDE a stubbed forward, i.e. at
    the in-flight moment the bug actually 503'd."""
    import time as _time

    import numpy as np

    cfg = tiny()
    eng = build_engine(cfg, build_random(cfg, seed=91), get_backend(),
                       num_blocks=32, num_slots=4, max_batch=4, max_total_tokens=4096)
    seen: dict = {}
    real_forward = eng._run_forward

    def forward_while_in_flight(decodes, prefills, chunks):
        # We are inside step(), on the engine thread, before the real forward:
        # exactly the "first prefill in flight after idle" instant.
        live, stuck = eng.liveness(60.0)
        seen["inflight_live"] = live
        seen["inflight_stuck"] = stuck
        # finish the tick so shutdown is clean
        return real_forward(decodes, prefills, chunks)

    eng._run_forward = forward_while_in_flight
    try:
        eng._last_progress_ts = _time.perf_counter() - 300.0  # long idle
        assert eng.liveness(60.0) == (True, 0.0)  # idle stays live
        eng.submit(np.arange(5, 5 + 128, dtype=np.int64),
                   SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
        eng.step()
        assert seen, "the in-flight forward hook never ran"
        assert seen["inflight_live"] is True, (
            f"first forward in flight after idle wrongly flagged stuck: "
            f"{seen['inflight_stuck']}s")
        eng.poll()
    finally:
        eng._run_forward = real_forward
        eng.shutdown()


def test_liveness_stamped_on_submit_idle_to_active_edge_only():
    """submit() must refresh _last_progress_ts only on the idle->active edge.
    Before the edge stamp the submit-to-first-tick gap inherited the idle
    timestamp, so the first request queued after a long idle read stuck for the
    whole gap (a stale /health poll in that window). A second submit onto an
    unadmitted backlog must NOT stamp: that backlog really is waiting, and one
    that old must still 503. No run() loop here, so no tick intervenes -- the
    assertions observe the exact submit->tick window. Removing the edge stamp
    fails the first assertion; stamping on every submit fails the second."""
    import time as _time

    import numpy as np

    cfg = tiny()
    eng = build_engine(cfg, build_random(cfg, seed=97), get_backend(),
                       num_blocks=32, num_slots=4, max_batch=4, max_total_tokens=4096)
    try:
        prompt = np.arange(5, 5 + 128, dtype=np.int64)
        params = SamplingParams(temperature=0.0, max_new_tokens=2, seed=0)

        eng._last_progress_ts = _time.perf_counter() - 61.0
        assert eng.liveness(60.0) == (True, 0.0)  # idle stays live

        eng.submit(prompt, params)  # idle -> active edge: stamps
        live, stuck = eng.liveness(60.0)
        assert live is True and stuck < 0.05, f"false stall window after submit: {stuck}s"

        # Aged clock + second submit while the first is still unadmitted:
        # not an edge, so no refresh and the 61s-old backlog is flagged.
        eng._last_progress_ts = _time.perf_counter() - 61.0
        eng.submit(prompt + 1, params)
        live, stuck = eng.liveness(60.0)
        assert live is False and stuck > 60.0
    finally:
        eng.shutdown()


def test_forward_oom_is_fatal_but_a_normal_error_finishes_the_row():
    """Allocator OOM past the held memory fraction must terminate for a supervisor
    restart, NOT be swallowed by the daemon loop's log-and-continue (a half-dead
    server drains to idle and answers /health 200). A plain per-request error must
    still _finish and keep serving. Drives the REAL loop thread; the process-exit
    seam is monkeypatched so the test does not os._exit itself."""
    import time

    import numpy as np
    import torch

    import tilerl.engine as eng_mod
    from tilerl.engine import FatalDeviceError

    def make_engine():
        cfg = tiny()
        return build_engine(cfg, build_random(cfg, seed=71), get_backend(),
                            num_blocks=32, num_slots=4, max_batch=4,
                            max_total_tokens=4096, sparse_k=0)

    prompt = np.arange(5, 5 + 64, dtype=np.int64)
    params = dict(temperature=0.0, max_new_tokens=2, seed=0)

    # --- positive: OutOfMemoryError in _run_forward is fatal ----------------
    eng = make_engine()
    calls: list = []
    real_forward = eng._run_forward
    real_exit = eng_mod.fatal_device_exit

    def boom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (fraction fence)")

    def fake_exit(exc):
        calls.append(exc)
        eng._wake.set()  # stop the loop without exiting the test process

    eng._run_forward = boom
    eng_mod.fatal_device_exit = fake_exit
    eng.submit(prompt, SamplingParams(**params))
    eng.run()
    try:
        for _ in range(100):
            if calls:
                break
            time.sleep(0.02)
        assert len(calls) == 1, f"fatal seam called {len(calls)} times"
        assert isinstance(calls[0], FatalDeviceError)
        # engine records fatal and reads dead even before the process exits
        assert eng._fatal is calls[0]
        assert eng.liveness(60.0)[0] is False
        # the loop must not accept a NEW submit after the OOM
        with pytest.raises(FatalDeviceError):
            eng.submit(np.arange(5, 5 + 32, dtype=np.int64), SamplingParams(**params))
    finally:
        eng_mod.fatal_device_exit = real_exit
        eng._run_forward = real_forward
        eng.shutdown()

    # --- negative: an ordinary RuntimeError is recovered, not fatal ----------
    eng2 = make_engine()
    real_forward2 = eng2._run_forward

    def ordinary(*a, **k):
        raise RuntimeError("transient per-request failure")

    eng2._run_forward = ordinary
    rid2 = eng2.submit(prompt, SamplingParams(**params))
    eng2.run()
    try:
        from tilerl.engine import RequestFailed

        # poll() RAISES a failed row rather than returning it
        for _ in range(100):
            try:
                out = eng2.poll()
                if rid2 in out:
                    raise AssertionError("failed row unexpectedly returned data")
            except RequestFailed as rf:
                if rf.request_id == rid2:
                    assert "transient" in str(rf)
                    break
            time.sleep(0.02)
        else:
            raise AssertionError("failed row never surfaced")
        assert eng2._fatal is None
        # still serves a second request once the forward works again (rid2's
        # failure stays in _failed and must not block rid3)
        eng2._run_forward = real_forward2
        rid3 = eng2.submit(np.arange(5, 5 + 32, dtype=np.int64), SamplingParams(**params))
        for _ in range(100):
            if rid3 in eng2._finished:
                break
            time.sleep(0.02)
        assert rid3 in eng2._finished
    finally:
        eng2.shutdown()



def test_health_stats_carry_in_process_device_free_and_limit():
    """The long-term observability for a memory-fraction reserve: stats expose the
    process allocator's free/limit (mem_get_info), distinct from nvidia-smi. Off
    cuda both are 0 (no device); the fields always exist so readers need no
    device branch. The cuda values are pending-remote."""
    cfg = tiny()
    eng = build_engine(cfg, build_random(cfg, seed=7), get_backend(),
                       num_blocks=8, num_slots=4, max_batch=4,
                       max_total_tokens=2048, sparse_k=0)
    try:
        s = eng.stats()
        assert s["device_free_bytes"] == 0
        assert s["device_limit_bytes"] == 0
    finally:
        eng.shutdown()


class _MultiParkEngine:
    """N independent SSE rows: each emits one frame, then its peek() BLOCKS on a
    per-rid event (the sync body is parked in to_thread(next), the exact
    "worker in flight" state the wedge needs). cancel() releases the park and
    frees the slot. Tracks per-rid cancel + completion off the loop."""

    def __init__(self, tok, n):
        self.tok = tok
        self.ids = [1000 + i for i in range(n)]
        self._next = 0
        self.park = {rid: threading.Event() for rid in self.ids}
        self.frame_sent = {rid: threading.Event() for rid in self.ids}
        self.cancelled: set = set()
        self.slots_used = 0
        self.params: list = []

    def submit(self, input_ids, params=None) -> int:
        rid = self.ids[self._next]
        self._next += 1
        self.slots_used += 1
        self.params.append(params)
        return rid

    def peek(self, rid):
        if not self.frame_sent[rid].is_set():
            self.frame_sent[rid].set()
            return self.tok.encode(f"frame-{rid}")
        self.park[rid].wait(30.0)   # blocked in-flight poll; cancel sets the event
        return None if rid in self.cancelled else self.tok.encode(f"more-{rid}")

    def take(self, rid):
        self.park[rid].wait(30.0)
        if rid in self.cancelled:
            raise RuntimeError("cancelled")
        return self.tok.encode(f"more-{rid}")

    def stop_text(self, rid):
        return None

    def cancel(self, rid) -> bool:
        if rid in self.cancelled:
            return False
        self.cancelled.add(rid)
        self.slots_used = max(0, self.slots_used - 1)
        self.park[rid].set()
        return True

    def room_for(self, prompt_tokens):
        return 4096


def _open_stream_socket(port, payload):
    import socket
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
           f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
           f"\r\n{payload}").encode()
    s.sendall(req)
    return s


class _WideDefaultExecutor:
    """ASGI lifespan wrapper: give uvicorn's loop a wider default executor.
    N parked in-flight to_thread(next) workers each need a SECOND pool thread for
    the disconnect's to_thread(engine.cancel); the stock executor is
    min(32, cpu+4) = 8 on 4-vCPU CI, so N=8 cancels would queue behind the parked
    workers forever (each park's future does not release until cancel sets it).
    The replacement is shut down on lifespan shutdown so its non-daemon worker
    threads do not hang interpreter exit. Test-only; production hosts run >16
    vCPU and the hybrid server's slots bound the in-flight count below the pool."""

    def __init__(self, inner, max_workers):
        self.inner = inner
        self.max_workers = max_workers

    async def __call__(self, scope, receive, send):
        if scope["type"] != "lifespan":
            await self.inner(scope, receive, send)
            return
        import concurrent.futures
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self._pool = pool
        self._loop = asyncio.get_running_loop()
        self._loop.set_default_executor(pool)
        try:
            await self.inner(scope, receive, send)
        finally:
            # The gate proves every parked row is released before should_exit, so
            # no worker is stuck here; wait=False only bounds a broken run.
            pool.shutdown(wait=False)


def _read_one_frame(s):
    buf = b""
    while b'"content"' not in buf:
        chunk = s.recv(4096)
        assert chunk, "no SSE frame before hang-up"
        buf += chunk
    return buf


def test_simultaneous_sse_hangups_with_inflight_workers_do_not_freeze_the_loop():
    """2026-09-16 wedge, real transport (only real uvicorn delivers the anyio
    task-group cancellation that latches: Starlette 1.6 + ASGI 2.3 httptools uses
    a collapsing task group whose CancelScope stays cancelled, so every checkpoint
    the SSE task hits in its final drain re-raised CancelledError).

    N SSE responses are each parked in to_thread(next) after their first frame.
    All sockets are shut down in one loop turn. The OLD final drain awaited the
    in-flight worker from inside the cancelled scope (`shield; continue`): it
    re-raised on every checkpoint with no real waiter, and N of them together
    starved the loop -> a /health probe through the teardown window stalled. The
    fix detaches the drain, so /health keeps answering and every body is
    cancelled/off-loop-closed. Event-synced (frame-arrived gates); the only wall
    bound is the responsiveness assertion itself."""
    import socket

    N = 8
    tok = _ByteTokenizer()
    eng = _MultiParkEngine(tok, N)
    server, port = _uvicorn_server(
        eng, tok, wrap_app=lambda app: _WideDefaultExecutor(app, N * 2 + 8))
    payload = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                          "stream": True, "max_tokens": 64})
    socks = []
    try:
        for _ in range(N):
            s = _open_stream_socket(port, payload)
            socks.append(s)
        # each connection is now parked in its blocked second peek()
        for rid in eng.ids:
            assert eng.frame_sent[rid].wait(5.0), f"rid {rid} never sent a frame"
        for s in socks:
            _read_one_frame(s)

        def health_latency():
            probe = socket.create_connection(("127.0.0.1", port), timeout=5)
            try:
                probe.sendall(b"GET /health HTTP/1.1\r\nHost: t\r\n"
                              b"Connection: close\r\n\r\n")
                t0 = time.monotonic()
                ans = b""
                while b"\r\n\r\n" not in ans:
                    c = probe.recv(4096)
                    assert c, "no /health response"
                    ans += c
                return time.monotonic() - t0, ans
            finally:
                probe.close()

        # one baseline probe, then hang up ALL sockets in one turn and keep
        # probing /health THROUGH the teardown: the spin starved it for minutes.
        lat0, _ = health_latency()
        assert lat0 < 0.5
        for s in socks:
            s.shutdown(socket.SHUT_RDWR)
        for s in socks:
            s.close()
        socks = []
        worst = 0.0
        deadline = time.monotonic() + 3.0
        healthy = 0
        while time.monotonic() < deadline:
            lat, ans = health_latency()
            worst = max(worst, lat)
            # The stub engine never ticks, so liveness may answer 503 once rows
            # park; the wedge symptom is NON-ANSWER (loop starved), not the code.
            # Any complete status line within the bound proves the loop scheduled.
            assert b"HTTP/1.1 " in ans, ans[:80]
            healthy += 1
        assert healthy >= 20, f"too few /health probes got through teardown: {healthy}"
        assert worst < 0.5, f"/health stalled {worst:.2f}s during SSE cancel storm"

        # every parked row was cancelled (its peek worker unblocked, slot freed).
        # 10s is the fail-loud bound, not the measured path: event-synced, and on
        # an unloaded host delivery takes milliseconds; the wide bound tolerates a
        # CI host already running OpenMP-heavy earlier tests in this file.
        for rid in eng.ids:
            assert eng.park[rid].wait(10.0), f"rid {rid} worker never released"
        assert eng.slots_used == 0
    finally:
        for s in socks:
            with contextlib.suppress(OSError):
                s.close()
        server.should_exit = True


def test_stream_or_cancel_final_drain_is_detached_not_awaited_in_cancel_scope():
    """Structural pin for the 2026-09-16 wedge root: the SSE final drain MUST NOT
    await an executor future from inside the (cancellable) SSE task. Any
    shield/wait-on-worker in that frame is re-raised on every checkpoint of a
    latched anyio cancel scope and busy-spins the loop. The drain is detached
    (module-level strong-ref set + a helper), and body.close runs on a worker
    thread inside that helper. Deterministically red on the old shield/continue
    loop, which the timing-sensitive real-socket gate cannot guarantee on every
    host."""
    import ast
    import inspect
    import pathlib

    import tilerl.server as srv_mod

    src = inspect.getsource(srv_mod.stream_or_cancel)
    tree = ast.parse(src)
    # no shield / wait-for on an executor inside stream_or_cancel at all
    for node in ast.walk(tree):
        assert not (isinstance(node, ast.Attribute) and node.attr == "shield"), (
            "stream_or_cancel must not shield/await an executor in its cancel scope")
        assert not (isinstance(node, ast.Attribute) and node.attr == "wait_for"), (
            "stream_or_cancel must not wait_for an executor in its cancel scope")
    # the frame detaches to the helper
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_detach_drain" in calls, "final drain must be detached via _detach_drain"
    assert "to_thread" not in calls, "stream_or_cancel must not run body.close on the loop"

    full = pathlib.Path(srv_mod.__file__).read_text()
    # the detached registry exists and is a module-level set (strong refs)
    assert isinstance(srv_mod._draining, set)
    # _drain_body cancels BEFORE awaiting, then closes off the loop
    dsrc = inspect.getsource(srv_mod._drain_body)
    assert "to_thread(engine.cancel, request_id)" in dsrc
    assert "to_thread(body.close)" in dsrc
    assert "wait_for(asyncio.shield(worker)" in dsrc
    assert dsrc.index("to_thread(engine.cancel") < dsrc.index("wait_for(asyncio.shield"), (
        "drain must cancel the row before waiting on its in-flight worker: "
        "a GeneratorExit-at-yield skips the in-scope cancel and the worker is "
        "parked until cancel runs, so waiting first deadlocks the drain")
    # the set is both added and discard-on-done (no unbounded growth / GC)
    assert "_draining.add" in full and "discard" in full
    # the prompt disconnect paths still await their cancel IN the frame (slot
    # release is not best-effort); the drain's cancel is only the backstop
    fsrc = src
    assert fsrc.count("to_thread(engine.cancel, request_id)") >= 2
    # a skipped body.close (worker timeout or raise) must be logged, not silent;
    # a cancel that raises before the join is logged too (#667: it must not abort
    # the worker join / body.close).
    assert dsrc.count("logging.warning") == 3 and "body.close() skipped" in dsrc
    assert "engine.cancel raised" in dsrc
    # graceful shutdown joins in-flight drains through the app lifespan, bounded
    assert "lifespan=_lifespan" in full
    jsrc = inspect.getsource(srv_mod._await_drains)
    assert "asyncio.wait(tuple(_draining), timeout=_DRAIN_WAIT_S)" in jsrc


def test_detached_drains_are_awaited_at_shutdown_while_a_close_is_still_running():
    """Graceful stop (SIGTERM) must JOIN an in-flight detached drain within the
    bound: an SSE task torn down just before exit leaves _drain_body running
    body.close() on a worker thread; shutdown has to wait for it instead of
    letting an unclosed sync generator outlive the server. Event-synced: the
    join future must be pending while close() blocks, and must complete once
    close returns. Red on a no-op shutdown helper (join would be done early) and
    on a missing helper (attribute error). The 30s bound itself is pinned
    structurally in the AST gate; a 30s wall test is not worth it."""

    class _SlowClose:
        def __init__(self):
            self.close_started = threading.Event()
            self.released = threading.Event()
            self.closed = False

        def close(self):
            self.close_started.set()
            self.released.wait(5.0)
            self.closed = True

    class _QuickCancel:
        def __init__(self):
            self.cancelled = threading.Event()

        def cancel(self, rid):
            self.cancelled.set()
            return True

    import tilerl.server as srv_mod

    async def main():
        body, eng = _SlowClose(), _QuickCancel()
        worker = asyncio.ensure_future(asyncio.sleep(0))
        await worker  # the in-flight next() has already returned; close() blocks
        srv_mod._detach_drain(eng, 1, worker, body)
        for _ in range(25):  # let the drain task start on the loop
            await asyncio.sleep(0.02)
            if eng.cancelled.is_set():
                break
        assert eng.cancelled.wait(2.0)
        assert body.close_started.wait(2.0)
        assert srv_mod._draining

        join = asyncio.ensure_future(srv_mod._await_drains())
        for _ in range(20):  # let the join reach its wait
            await asyncio.sleep(0.02)
            if not join.done():
                break
        assert not join.done(), "shutdown returned while body.close() was running"
        body.released.set()
        await asyncio.wait_for(join, timeout=2.0)
        assert body.closed
        for _ in range(50):  # done callback discards the strong ref
            if not srv_mod._draining:
                break
            await asyncio.sleep(0.02)
        assert not srv_mod._draining

    asyncio.run(main())


class _GatedTakeEngine(_ScriptedEngine):
    """A row that is not ready until a timer fires (or never).

    take() returns None while ``_ready`` is unset, so await_completion's poll
    loop crosses a short deadline the way an unfinished long prefill does; once
    set, take() pops the canned reply. ``ready_after=None`` never finishes.
    """

    def __init__(self, tokenizer, replies, ready_after: float | None = 0.25):
        super().__init__(tokenizer, replies)
        self._ready = threading.Event()
        self.cancelled: list[int] = []
        self._timer = (
            threading.Timer(ready_after, self._ready.set) if ready_after is not None else None
        )

    def submit(self, input_ids, params=None) -> int:
        rid = super().submit(input_ids, params)
        if self._timer:
            self._timer.start()
        return rid

    def take(self, request_id: int):
        return self._done.pop(request_id, None) if self._ready.is_set() else None

    def cancel(self, request_id: int) -> bool:
        self.cancelled.append(request_id)
        return True


def test_completion_timeout_resolver_is_three_state(monkeypatch):
    from tilerl.messages import completion_timeout_from_env

    monkeypatch.delenv("TILERL_COMPLETION_TIMEOUT_S", raising=False)
    assert completion_timeout_from_env() == 1800.0
    monkeypatch.setenv("TILERL_COMPLETION_TIMEOUT_S", "7200")
    assert completion_timeout_from_env() == 7200.0
    monkeypatch.setenv("TILERL_COMPLETION_TIMEOUT_S", "0")
    assert completion_timeout_from_env() == 0.0


def test_await_completion_zero_means_no_deadline():
    from tilerl.prompt import await_completion

    eng = _GatedTakeEngine(_ByteTokenizer(), ["ok"], ready_after=0.05)
    rid = eng.submit([1, 2, 3])
    assert await_completion(eng, rid, 0.0, poll_s=0.01)  # 0 would otherwise raise at once

    class _Never:
        def take(self, rid):
            return None

    # A positive deadline still raises TimeoutError past a non-ready row.
    with pytest.raises(TimeoutError):
        await_completion(_Never(), 1, 0.05, poll_s=0.01)


def test_nonstream_request_504s_past_a_short_completion_timeout():
    engine = _GatedTakeEngine(_ByteTokenizer(), ["late"], ready_after=None)  # never finishes
    with TestClient(create_app(engine, _ByteTokenizer(), completion_timeout_s=0.1)) as c:
        r = c.post("/v1/chat/completions",
                   json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.status_code == 504, r.text
    assert r.json()["error"]["type"] == "api_error"
    assert engine.cancelled, "the timed-out row must be cancelled to free its slot"


def test_nonstream_request_waits_through_zero_completion_timeout():
    engine = _GatedTakeEngine(_ByteTokenizer(), ["late-but-ok"], ready_after=0.15)
    with TestClient(create_app(engine, _ByteTokenizer(), completion_timeout_s=0.0)) as c:
        r = c.post("/v1/chat/completions",
                   json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "late-but-ok"
