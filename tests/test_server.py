"""Server gates for tilerl: /health, /v1/models, non-stream completion, SSE stream.

Uses FastAPI's TestClient against a tiny-engine app. A deterministic
byte-level tokenizer (vocab 320, matching tiny()) stands in at the IO
boundary — the gate is HTTP/SSE behaviour, not tokenization fidelity.
"""

from __future__ import annotations

import json
import os
import threading
import time

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
from fastapi.testclient import TestClient
from tilerl_kernels.backend import get_backend

from tilerl.config import tiny
from tilerl.engine import Engine, SamplingParams, build_engine
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
            good = c.get("/health").json()
        with TestClient(create_app(_StatsRaises(engine), _ByteTokenizer())) as c:
            bad = c.get("/health").json()
    finally:
        engine.shutdown()

    assert good["status"] == "ok" and good["stats"], good
    assert bad["status"] != "ok", f"a raising engine still reports {bad['status']!r}"
    assert bad["stats"] is None and "RuntimeError" in bad.get("error", ""), bad


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
    lines = [ln for ln in streamed.text.split("\n") if ln.startswith("data:")]
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


@pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="a wall-duration verdict on a live HTTP round-trip is machine load, not code: "
    "the same flakiness class as the GIL-yield ratio that went red at 1.47 on a healthy "
    "shared runner (errors/2026-09-11-flaky-wallclock-test-inventory.md). The non-blocking "
    "property this covers is run locally/dedicated.",
)
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

    The gate is a second request answering while the first is still in flight, which is the
    property the defect broke. A slow engine, not a slow model: `take` returning None for a
    fixed number of polls is the same shape as a long generation and costs the suite ~1 s.
    """
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "loop.jsonl"))
    tok = _ByteTokenizer()

    class _SlowEngine(_ScriptedEngine):
        #: ~1.4 s of polling at _run's 0.02 s interval — long enough that a blocked loop
        #: cannot answer /health inside the 1 s assertion, short enough for the suite.
        POLLS = 70

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._left: dict[int, int] = {}

        def take(self, request_id: int):
            self._left.setdefault(request_id, self.POLLS)
            if self._left[request_id] > 0:
                self._left[request_id] -= 1
                return None
            return super().take(request_id)

    engine = _SlowEngine(tok, ["</think>\n\ndone"])
    app = create_app(engine, tok, model_name="tiny")
    with TestClient(app) as c:
        done: dict[str, object] = {}
        t = threading.Thread(target=lambda: done.update(
            code=c.post(path, json=body).status_code))
        t.start()
        try:
            # Wait for the request to be IN FLIGHT, else /health answers before the poll
            # loop starts and the arm passes on a server that was never busy.
            for _ in range(200):
                if engine.params:
                    break
                time.sleep(0.01)
            assert engine.params, f"{path} never reached submit; the arm proves nothing"
            t0 = time.monotonic()
            health = c.get("/health")
            elapsed = time.monotonic() - t0
        finally:
            t.join(timeout=30)

    assert health.status_code == 200, health.text
    assert elapsed < 1.0, (
        f"/health took {elapsed:.2f}s while {path} was generating — the route blocks the "
        f"event loop instead of awaiting through asyncio.to_thread")
    assert done.get("code") == 200, f"the {path} request itself failed: {done}"


@pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="the 100 ms wall verdict on a lock-holding forward is machine load, not code; "
    "the 0.1 s bound is tighter than the GIL ratio that already went red at 1.47 on a "
    "healthy shared runner (errors/2026-09-11-flaky-wallclock-test-inventory.md). The "
    "lock-free-snapshot property is verified locally/dedicated.",
)
def test_health_does_not_wait_on_the_engine_lock(tmp_path):
    """`/health` must answer while `step()` holds `_lock` across a forward.

    Separate defect from the `to_thread` freeze above, and the reason both gates exist: that
    one was the event loop, this one is the lock, and fixing the loop did not fix this. On
    the live V100 during a 21.7k-token prefill /health ran at a median of 8.12 s and a max of
    **87.66 s**, against 0.002 s idle — four orders of magnitude on identical code, because
    `stats()` took the lock that a 43-chunk prefill holds one chunk at a time.

    A real `Engine` is used, not a double: the property under test is which lock `stats()`
    takes, and a double that reimplements `stats()` would assert its own behaviour. The
    forward is replaced by a sleep so the tick is slow without needing a model — that is the
    only substitution, and it is at the layer below the one being measured.
    """
    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=43), get_backend(),
                          num_blocks=32, num_slots=4, max_batch=4, max_total_tokens=4096)

    held = threading.Event()

    def _slow_forward(*_a, **_kw):
        held.set()
        time.sleep(2.0)  # 20x the 100 ms assertion, so a lock-taking reader cannot pass

    engine._run_forward = _slow_forward
    engine.submit([1, 2, 3], SamplingParams(max_new_tokens=4))
    engine.run()
    try:
        assert held.wait(10.0), "the forward never started; the arm proves nothing"
        t0 = time.monotonic()
        snap = engine.stats()
        elapsed = time.monotonic() - t0
    finally:
        engine.shutdown()

    assert isinstance(snap, dict) and "pool_used_blocks" in snap, snap
    assert elapsed < 0.1, (
        f"stats() took {elapsed:.2f}s while step() held the lock across a forward — "
        f"/health waits on the engine lock instead of reading a published snapshot")


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
                          num_blocks=32, num_slots=4, max_batch=4, max_total_tokens=4096)
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
            "--max-ctx", "512", "--no-warmup"]
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
                       num_slots=4, max_batch=1, max_total_tokens=4096)
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
    generator did nothing at all.
    """
    import inspect

    from tilerl import server

    src = inspect.getsource(server)
    assert src.count("engine.cancel(request_id)") == 2, (
        "expected the WebSocketDisconnect branch and the SSE generator's GeneratorExit "
        f"to cancel; found {src.count('engine.cancel(request_id)')}"
    )
    assert "except GeneratorExit:" in src, (
        "the SSE route needs GeneratorExit: starlette closes the generator when the "
        "client hangs up, and without it an abandoned SSE stream runs to its cap"
    )


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
            "--max-ctx", "512", "--no-warmup"]
    if state_bytes:
        argv += ["--state-bytes", str(state_bytes)]
    cli.cmd_serve(cli._build_parser().parse_args(argv))
    capsys.readouterr()

    got = served["health"]["stats"].get("prefix_state_bytes_budget")
    if state_bytes:
        assert got == state_bytes, f"--state-bytes {state_bytes} did not reach the store: {got}"
    else:
        assert got and got != 12345678, f"the default budget is the flag's value: {got}"
