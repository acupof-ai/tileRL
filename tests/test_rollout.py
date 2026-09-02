"""Stage-2 gate: one command starts a server and drives `claude -p` through it.

The engine is scripted, not a real model -- the launcher's job is the wiring
(port, base URL, sandbox settings, episode tag, record rows), and gating it on a
real model would make it depend on weights this machine does not have. That is
the same reason stage 1's semantic half uses a scripted engine.

Skipped where the `claude` CLI is absent (CI, the pod), because the thing under
test IS the CLI's contract with our server.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import shutil

import pytest

from tilerl import rollout as rollout_mod
from tilerl.messages import render_tool_call
from tilerl.server import create_app

from test_server import _ByteTokenizer, _ScriptedEngine

pytestmark = pytest.mark.skipif(shutil.which("claude") is None,
                                reason="stage-2 gate needs the claude CLI")


def test_sandbox_settings_refuse_rather_than_run_bare():
    s = rollout_mod.sandbox_settings("127.0.0.1", 9000)["sandbox"]
    # The one key that matters: a host that cannot sandbox must not run the
    # agent unconfined -- an unisolated rollout is worse than a missing one.
    assert s["enabled"] and s["failIfUnavailable"]
    assert s["allowUnsandboxedCommands"] is False
    assert s["network"]["allowedDomains"] == ["127.0.0.1:9000"]
    assert s["network"]["strictAllowlist"] is True


def test_rollout_records_carry_the_episode_tag(tmp_path, monkeypatch):
    """A `claude -p` episode against our server, joined to its rows by tag."""
    record = tmp_path / "rec.jsonl"
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(record))
    tok = _ByteTokenizer()
    app = create_app(_ScriptedEngine(tok, ["done: two files"]), tok)
    base, _ = rollout_mod.serve_app(app)

    # sandbox off: the sandbox blocks localhost by name resolution on some
    # hosts, and what this gate measures is the record wiring, not isolation.
    res = rollout_mod.run_rollout("say done", cwd=str(tmp_path), base_url=base,
                                  tag="ep-1", sandbox=False, timeout_s=120.0)
    assert res["returncode"] == 0, res.get("stderr")

    rows = rollout_mod.read_records(str(record), "ep-1")
    assert rows, f"no record rows for ep-1; file={record.read_text()[:400]!r}"
    for r in rows:
        assert r["rollout"] == "ep-1"
        assert len(r["logprobs"]) == len(r["completion_ids"])
        assert r["stop_reason"] in {"end_turn", "max_tokens", "tool_use"}
    # A different tag shares the file and must not be picked up.
    assert rollout_mod.read_records(str(record), "ep-2") == []
    # The episode's rows are the trajectory, in served order.
    assert [r["request_id"] for r in rows] == sorted(r["request_id"] for r in rows)


def test_records_are_partitioned_by_tag(tmp_path, monkeypatch):
    record = tmp_path / "rec.jsonl"
    record.write_text("\n".join(json.dumps(r) for r in [
        {"request_id": 1, "rollout": "a", "completion_ids": [1], "logprobs": [-0.1]},
        {"request_id": 2, "rollout": "b", "completion_ids": [2], "logprobs": [-0.2]},
        {"request_id": 3, "rollout": "a", "completion_ids": [3], "logprobs": [-0.3]},
    ]) + "\n")
    assert [r["request_id"] for r in rollout_mod.read_records(str(record), "a")] == [1, 3]
    assert [r["request_id"] for r in rollout_mod.read_records(str(record), "b")] == [2]


@pytest.mark.skipif(not rollout_mod.sandbox_available()[0],
                    reason=f"no sandbox here: {rollout_mod.sandbox_available()[1]}")
def test_sandbox_confines_writes_to_the_rollout_dir(tmp_path, monkeypatch):
    """The sandbox blocks a write outside cwd -- with its own negative control.

    Asserting only that the escape fails would pass just as well if the command
    never ran, so the SAME scripted escape runs twice: sandboxed (must not
    appear) and unsandboxed (must appear). One assertion without the other
    measures nothing.
    """
    monkeypatch.setenv("TILERL_MESSAGES_RECORD", str(tmp_path / "rec.jsonl"))
    outside = tmp_path / "outside" / "ESCAPED.txt"
    outside.parent.mkdir()
    escape = render_tool_call("Bash", {"command": f"echo pwned > {outside} && echo wrote"})

    def attempt(sandbox: bool, tag: str) -> None:
        tok = _ByteTokenizer()
        app = create_app(_ScriptedEngine(tok, [escape, "reported"]), tok)
        base, _ = rollout_mod.serve_app(app)
        work = tmp_path / f"work-{tag}"
        work.mkdir()
        rollout_mod.run_rollout("write the file", cwd=str(work), base_url=base, tag=tag,
                                sandbox=sandbox, tools=["Bash"], timeout_s=240.0)

    attempt(sandbox=True, tag="sb")
    assert not outside.exists(), "sandboxed rollout wrote outside its directory"
    attempt(sandbox=False, tag="bare")
    assert outside.exists(), (
        "negative control failed: the escape did not write even unsandboxed, so "
        "the sandboxed assertion above proves nothing"
    )
