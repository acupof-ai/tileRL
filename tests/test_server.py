"""Server gates for tilerl: /health, /v1/models, non-stream completion, SSE stream.

Uses FastAPI's TestClient against a tiny-engine app. A deterministic
byte-level tokenizer (vocab 320, matching tiny()) stands in at the IO
boundary — the gate is HTTP/SSE behaviour, not tokenization fidelity.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
from fastapi.testclient import TestClient
from tilerl_kernels.backend import get_backend

from tilerl.config import tiny
from tilerl.engine import Engine, build_engine
from tilerl.messages import render_tool_call
from tilerl.model import build_random
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


def _build_engine(seed: int) -> Engine:
    cfg = tiny()
    model = build_random(cfg, seed=seed)
    backend = get_backend()
    # 4096, not 512: the tool block is the checkpoint's template verbatim (817
    # bytes of instructions alone), and ByteTokenizer is one token per byte, so
    # a one-tool request is ~1.1k tokens. Shrinking the prompt to fit would be
    # measuring a format the 27B never sees.
    return build_engine(
        cfg, model, backend, num_blocks=256, num_slots=4, max_batch=4, max_total_tokens=4096
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
    """SSE must deliver text as it is generated, not one block at the end.

    Two things must hold at once and they pull against each other. Deltas must
    arrive as separate chunks (streaming), AND concatenating them must equal the
    non-streamed text exactly (correctness). The trap is decoding per token: one
    token is not one character, so a per-token decode splits multi-byte UTF-8.
    _ByteTokenizer makes that reachable -- one id per BYTE, so any multi-byte
    character is guaranteed to span tokens.

    Uses _TextTokenizer, whose decode is a function of the id COUNT against a fixed
    string, so the premise does not depend on which bytes the random weights sample.
    With plain _ByteTokenizer this passed locally and failed on macos-14: the reply
    there was mostly UTF-8 continuation bytes, so every prefix ended in U+FFFD, the
    rstrip held the visible text flat, and one delta carried everything.

    Builds its own engine at seed 7 rather than taking the module `client`, whose
    seed 42 is the one seed measured where the two-defect loop still produced two
    deltas -- i.e. where this test cannot see the bug. At seed 7 the broken loop
    puts the entire reply in the final chunk.
    """
    engine = _build_engine(seed=7)
    engine.run()
    try:
        client = TestClient(create_app(engine, _TextTokenizer()))
        model_id = client.get("/v1/models").json()["data"][0]["id"]
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 24,
            "temperature": 0.0,
            "seed": 7,
        }
        _assert_stream_is_incremental(client, body)
    finally:
        engine.shutdown()


def _assert_stream_is_incremental(client, body) -> None:
    streamed = client.post("/v1/chat/completions", json={**body, "stream": True})
    assert streamed.status_code == 200, streamed.text
    lines = [ln for ln in streamed.text.splitlines() if ln.startswith("data:")]
    payloads = [json.loads(ln[len("data: ") :]) for ln in lines[:-1]]
    deltas = [
        p["choices"][0]["delta"]["content"]
        for p in payloads
        if p["choices"][0].get("delta", {}).get("content")
    ]

    # Correctness first, because it holds unconditionally: the pieces must reassemble
    # into the same text the non-streamed path returns for the same request.
    plain = client.post("/v1/chat/completions", json={**body, "stream": False})
    assert plain.status_code == 200, plain.text
    expected = plain.json()["choices"][0]["message"]["content"]
    assert "".join(deltas) == expected, (
        f"stream != non-stream:\n  joined  {''.join(deltas)!r}\n  expected {expected!r}"
    )

    # No incremental delta may END on U+FFFD: that is where a multi-byte character was
    # cut in half, and holding the trailing replacement run until its bytes arrive is
    # the whole point of the loop's rstrip.
    #
    # Deliberately not "contains no U+FFFD": tiny() has random weights, so its bytes
    # are mostly not valid UTF-8 and the non-streamed reply carries interior
    # replacement chars on all four model seeds measured -- an assertion against
    # containment is unsatisfiable for any loop that actually streams, and the earlier
    # one passed only because the loop emitted nothing until the end.
    #
    # How MUCH streamed is asserted by
    # test_a_reply_that_arrives_over_many_polls_streams_over_many_deltas, against a
    # stepped engine, and deliberately NOT here: the delta count off a live engine is
    # a race between generation speed and the loop's 20 ms poll, and the tiny model
    # can finish a 24-token reply inside one window. Measured 3 deltas at seeds
    # 7/42/3 and 1 at 11/99. An assertion that the last delta is not the whole reply
    # therefore fails on timing, not on a defect -- it did, 2 runs in 6 of this file,
    # while passing every time the test ran alone. The stepped engine reveals one
    # token per poll by construction and asserts the same invariant deterministically.
    assert not any(d.endswith("�") for d in deltas[:-1]), (
        f"an incremental delta ends mid-character: {deltas!r}"
    )


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
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:")]
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
        lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: {")]
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


def test_the_chat_page_reads_the_stream_this_server_sends(client, model_id):
    """The server's own SSE bytes, through the chat page's own reader.

    Every other gate holds one side still: the SSE tests here assert on bytes no page
    reads, and `tests/test_chat_ui.py` feeds the page frames no server wrote. So a
    change to the frame shape breaks the UI silently -- and the UI shipped broken twice
    on 2026-09-04 in the other direction, with the server correct throughout.

    Two properties of the real stream a careless reader dies on: the FIRST frame carries
    `{"role": "assistant"}` and no content, and the usage chunk carries no `choices` at
    all. Feeding the bytes whole and then one byte at a time also pins that frame
    boundaries do not depend on how the transport split them.

    Scope, measured by mutation: this catches a first frame that starts carrying content
    (CAUGHT) and a reader that stops buffering partial frames (CAUGHT). It does NOT catch
    usage becoming unconditional, because it always opts in --
    `test_usage_in_the_stream_is_opt_in_and_counts_tokens_not_characters` owns that half
    and was verified to fail on it.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; the page's reader cannot be executed")
    from tilerl.server import _CHAT_UI

    resp = client.post("/v1/chat/completions", json={
        "model": model_id, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8, "temperature": 0.0, "seed": 11,
        "stream": True, "stream_options": {"include_usage": True},
    })
    assert resp.status_code == 200, resp.text[:200]
    sse = resp.text
    assert "[DONE]" in sse and '"usage"' in sse, "the fixture is not a complete stream"

    js = _CHAT_UI.split("<script>", 1)[1].split("</script>", 1)[0]
    reader = js[js.index("async function readSSE") : js.index("\nasync function sendChat")]
    harness = reader + (
        "\nconst SSE = " + json.dumps(sse) + ";\n"
        "const enc = new TextEncoder();\n"
        "async function drive(parts) {\n"
        "  const frames = [], q = [...parts];\n"
        "  const resp = { body: { getReader: () => ({ read: async () =>\n"
        "    q.length ? {value: enc.encode(q.shift()), done: false} : {done: true} }) } };\n"
        "  await readSSE(resp, (o) => frames.push(o));\n"
        "  return frames;\n"
        "}\n"
        "const whole = await drive([SSE]), bytewise = await drive([...SSE]);\n"
        # `choices` is present but EMPTY on the usage frame, so `choices[0].delta` throws.
        # Optional chaining here for the same reason the page uses it: this harness made
        # the exact mistake the test exists to catch.
        "const hasText = (o) => !!o.choices?.[0]?.delta?.content;\n"
        "console.log(JSON.stringify({\n"
        "  whole: whole.length, bytewise: bytewise.length,\n"
        "  content: whole.filter(hasText).length,\n"
        "  firstHasContent: hasText(whole[0]),\n"
        "  emptyChoices: whole.filter((o) => Array.isArray(o.choices) && !o.choices.length).length,\n"
        "  usage: whole.filter((o) => o.usage).length,\n"
        "}));\n"
    )
    out = subprocess.run([node, "--input-type=module", "-e", harness],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, f"the page's reader threw on real bytes: {out.stderr[:500]}"
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["whole"] > 1, f"the reader yielded nothing usable: {got}"
    assert got["whole"] == got["bytewise"], (
        f"frame count depends on chunk boundaries ({got['whole']} vs {got['bytewise']}): "
        f"the reader does not buffer partial frames"
    )
    assert got["content"] >= 1, f"no content frame survived the reader: {got}"
    assert not got["firstHasContent"], (
        "this server's first frame now carries content, so the assertion no longer "
        "exercises the role-only frame a real reply starts with"
    )
    assert got["emptyChoices"] == 1, (
        f"exactly one frame must carry an empty choices list -- the usage chunk. Got "
        f"{got['emptyChoices']}. A reader indexing choices[0] unconditionally dies on it, "
        f"which this test's own harness did before optional chaining: {got}"
    )
    assert got["usage"] == 1, f"the usage chunk did not arrive exactly once: {got}"


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
        render_tool_call("Bash", {"command": "ls"}),
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
    app = create_app(_ScriptedEngine(tok, [reply]), tok)
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


def test_serve_sizes_its_pools_from_the_flags_not_the_context():
    """`tilerl serve`'s own engine builder must honour --blocks / --max-ctx.

    This is the one path no benchmark reaches: every bench script constructs the
    engine itself and passes num_blocks, so `_build_engine`'s default went
    unexercised until it asked for 275 GB of KV on a 32 GB card (131072 blocks
    from the 27B's 262144-token context). The gate is that the flags win and
    that the default is still derived from the context, which is what a
    large-card target relies on.
    """
    from tilerl import cli
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

    e = cli._build_engine(cfg, model, be, blocks=64, max_ctx=256, max_batch=2)
    assert e._kv.num_blocks - pad == 64
    assert e.limits.max_total_tokens == 256, "a request must not outgrow the pool"
    assert e.limits.max_batch == 2

    d = cli._build_engine(cfg, model, be)
    assert d._kv.num_blocks - pad == (4096 * d.limits.max_batch) // BLOCK_TOKENS, (
        "the default pool must cover max_batch rows of the context — no more "
        "(bytes are the long-context limit) and no less (a full batch must fit). "
        "On CUDA this is _fit_blocks' cap; a card too small to fit even the tiny "
        "model's 2048 blocks would report less, and that is a real capacity limit "
        "rather than a bug in the sizing"
    )
