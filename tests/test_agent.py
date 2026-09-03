"""Agent loop + /v1/agent SSE endpoint — hermetic, no model.

The agent.py self-check covers the loop/tool guards; this pins the server
wiring: the endpoint drives generate() through the engine contract and streams
thought/action/observation/answer as SSE.
"""

import json

from fastapi.testclient import TestClient

from tilerl.server import create_app

# generate() renders messages -> tokens -> engine.submit/take -> decode. A byte
# tokenizer round-trips text, so a canned per-call script stands in for a model.
_SCRIPT = [
    '{"thought": "run it", "tool": "shell", "args": {"command": "echo hi"}}',
    '{"answer": "the shell said hi"}',
]


class _FakeEngine:
    def __init__(self):
        self.i = 0

    def submit(self, ids, params):
        rid = self.i
        self.i += 1
        return rid

    def take(self, rid):
        return list(_SCRIPT[rid].encode())

    def stats(self):
        return {}


class _ByteTok:
    stop_token_ids = ()

    def encode(self, t):
        return list(t.encode())

    def decode(self, ids):
        return bytes(ids).decode("utf-8", "replace")


def _events(text):
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            out.append(json.loads(line[6:]))
    return out


def test_agent_endpoint_streams_react_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("TILERL_AGENT_TOOLS", str(tmp_path))  # opt in, server-pinned root
    app = create_app(_FakeEngine(), _ByteTok(), "tiny")
    client = TestClient(app)
    r = client.post("/v1/agent", json={"message": "say hi"})
    assert r.status_code == 200
    kinds = [e["type"] for e in _events(r.text)]
    assert kinds == ["thought", "action", "observation", "answer"], kinds
    evs = _events(r.text)
    assert evs[1]["payload"]["tool"] == "shell"
    assert evs[2]["payload"] == "hi"  # the shell actually ran
    assert evs[3]["payload"] == "the shell said hi"


def test_agent_endpoint_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TILERL_AGENT_TOOLS", raising=False)  # not opted in
    app = create_app(_FakeEngine(), _ByteTok(), "tiny")
    client = TestClient(app)
    r = client.post("/v1/agent", json={"message": "say hi"})
    assert r.status_code == 403  # shell tools off unless the operator enables them


def test_agent_tools_jail_and_deny(tmp_path):
    from tilerl.agent import Tools

    tools = Tools(str(tmp_path))
    # path jail
    try:
        tools.read_file("../../etc/passwd")
        raise AssertionError("path jail failed")
    except ValueError:
        pass
    # deny list
    try:
        tools.shell("rm -rf /")
        raise AssertionError("deny list failed")
    except ValueError:
        pass
    # write/read round-trip inside the jail
    assert "wrote" in tools.write_file("a.txt", "hi")
    assert tools.read_file("a.txt") == "hi"
