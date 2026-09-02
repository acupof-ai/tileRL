"""Server gates for tilerl: /health, /v1/models, non-stream completion, SSE stream.

Uses FastAPI's TestClient against a tiny-engine app. A deterministic
byte-level tokenizer (vocab 320, matching tiny()) stands in at the IO
boundary — the gate is HTTP/SSE behaviour, not tokenization fidelity.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
from fastapi.testclient import TestClient

from tilerl.config import tiny
from tilerl.engine import Engine, build_engine
from tilerl.model import build_random
from tilerl_kernels.backend import get_backend
from tilerl.server import create_app, get_tokenizer


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


def _build_engine(seed: int) -> Engine:
    cfg = tiny()
    model = build_random(cfg, seed=seed)
    backend = get_backend()
    return build_engine(
        cfg, model, backend, num_blocks=64, num_slots=4, max_batch=4, max_total_tokens=512
    )


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


def test_models(client, model_id):
    assert isinstance(model_id, str) and model_id


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


def test_seedless_requests_decorrelate(client, model_id):
    """Two seedless requests with the same prompt must not share a sampling
    stream (regression: every seedless request got seed=0, so concurrent
    same-prompt requests returned byte-identical completions)."""
    json_body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "max_tokens": 8,
    }
    # Six draws, not two: on the tiny model's small vocabulary two 8-token
    # samples collide often enough to fail a green build. The property is that
    # the stream is not SHARED — one repeat is chance, six identical is a bug.
    seen = {
        client.post("/v1/chat/completions", json=json_body).json()["choices"][0]["message"][
            "content"
        ]
        for _ in range(6)
    }
    assert len(seen) > 1, seen


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
    lines = [line for line in resp.text.splitlines() if line.startswith("data:")]
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
    assert resp.status_code == 422


def test_configured_tokenizer_fails_closed(tmp_path):
    with pytest.raises(Exception):
        get_tokenizer(str(tmp_path))


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
    assert out["content"] and out["content"][0]["type"] in ("text", "tool_use")
    rid = r.headers["x-tilerl-request-id"]

    row = json.loads(rec.read_text().splitlines()[-1])
    assert str(row["request_id"]) == rid, "the header must name the recorded row"
    assert row["prompt_ids"] and row["completion_ids"]
    # one score per generated token: what a policy gradient consumes
    assert len(row["logprobs"]) == len(row["completion_ids"])
    assert row["stop_reason"] == out["stop_reason"]


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

    def submit(self, input_ids, params=None) -> int:
        self._next += 1
        text = self._replies.pop(0) if self._replies else ""
        ids = self._tok.encode(text)
        self._done[self._next] = ids
        self._lp[self._next] = [-0.1] * len(ids)
        return self._next

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


def test_messages_tool_use_round_trip(tmp_path, monkeypatch):
    """The full agent shape: tool_use out, tool_result back in, answer out.

    Gates what a real Claude Code loop needs from the shim -- a structured
    tool_use block with stop_reason="tool_use", and a follow-up request whose
    tool_result content is rendered back into the prompt -- without needing a
    model that can produce either.
    """
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "rt.jsonl"))
    tok = _ByteTokenizer()
    engine = _ScriptedEngine(tok, [
        '{"tool": "Bash", "input": {"command": "ls"}}',
        "there are 3 files",
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
    assert "<tool_result>a.py b.py c.py</tool_result>" in tok.decode(rows[1]["prompt_ids"])
    assert '"tool": "Bash"' in tok.decode(rows[1]["prompt_ids"])
