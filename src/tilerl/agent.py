"""A minimal tool-calling agent loop over the serving engine.

The model runs a ReAct-ish loop: it emits ONE JSON action per turn — either a
tool call ``{"tool": "...", "args": {...}}`` or a final answer
``{"answer": "..."}``. The server executes the tool, appends the result as a
new turn, and re-prompts, up to ``max_steps``.

SECURITY: ``shell`` runs arbitrary commands on the server host. The deny list
and CWD jail are speed bumps, NOT a sandbox — ``shell`` is trivially escapable
(subshells, interpreters, symlinks). The real containment is that the endpoint
is off unless the operator opts in (``TILERL_AGENT_TOOLS``) and that the root
is server-pinned, never client-supplied. Do not expose this to untrusted
callers; run it only where arbitrary code execution is already acceptable.

The loop is generator-based: it yields ``(kind, payload)`` events
(``thought`` / ``action`` / ``observation`` / ``answer`` / ``error``) so the
server can stream them to the UI as they happen.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterator
from typing import Any

# A deny list is a speed bump, not a sandbox — shell=True is escapable a dozen
# ways. It only catches the obvious accidental foot-guns, not a hostile caller.
_DENY = ("rm -rf", "mkfs", "dd if=", ":(){", "shutdown", "reboot", "> /dev", "curl", "wget")


class Tools:
    """Backend tools rooted at ``root`` (server-pinned, never client-supplied).
    read/write paths are jailed to the root; shell is time-boxed but NOT
    sandboxed — see the module security note."""

    def __init__(self, root: str, timeout_s: float = 20.0) -> None:
        self.root = os.path.realpath(root)
        self.timeout_s = timeout_s

    def _resolve(self, path: str) -> str:
        p = os.path.realpath(os.path.join(self.root, path))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ValueError(f"path {path!r} escapes the agent root")
        return p

    def shell(self, command: str) -> str:
        if any(bad in command for bad in _DENY):
            raise ValueError(f"command refused (matched deny list): {command!r}")
        r = subprocess.run(
            command, shell=True, cwd=self.root, capture_output=True, text=True,
            timeout=self.timeout_s,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:4000] if out else f"(exit {r.returncode}, no output)"

    def read_file(self, path: str) -> str:
        with open(self._resolve(path)) as f:
            return f.read()[:8000]

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
        return f"wrote {len(content)} bytes to {path}"

    def dispatch(self, tool: str, args: dict) -> str:
        fn = {"shell": self.shell, "read_file": self.read_file, "write_file": self.write_file}
        if tool not in fn:
            raise ValueError(f"unknown tool {tool!r}; have {sorted(fn)}")
        return fn[tool](**args)


_SYSTEM = """You are a tool-using agent. Each turn, reply with ONE JSON object and nothing else:
- to use a tool: {"thought": "...", "tool": "shell"|"read_file"|"write_file", "args": {...}}
    shell:      args={"command": "<sh>"}          runs in the working dir
    read_file:  args={"path": "<rel path>"}
    write_file: args={"path": "<rel path>", "content": "<text>"}
- to finish:   {"thought": "...", "answer": "<final answer to the user>"}
The observation from each tool is fed back to you as the next turn. Keep going until you can answer."""


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first balanced {...} object out of the model's text."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in model output")


def run_agent(
    user_msg: str,
    generate: Callable[[list[dict]], str],
    tools: Tools,
    max_steps: int = 8,
) -> Iterator[tuple[str, Any]]:
    """Drive the ReAct loop. ``generate(messages) -> assistant_text`` is the
    one-shot completion call the server wires to the engine. Yields
    ``(kind, payload)`` events for streaming."""
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user_msg}]
    for _ in range(max_steps):
        raw = generate(messages)
        messages.append({"role": "assistant", "content": raw})
        try:
            action = _extract_json(raw)
        except ValueError as e:
            yield "error", f"could not parse action: {e}"
            return
        if action.get("thought"):
            yield "thought", action["thought"]
        if "answer" in action:
            yield "answer", action["answer"]
            return
        tool, args = action.get("tool"), action.get("args", {})
        yield "action", {"tool": tool, "args": args}
        try:
            obs = tools.dispatch(tool, args)
        except Exception as e:  # noqa: BLE001 - tool errors are fed back to the model
            obs = f"error: {e}"
        yield "observation", obs
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})
    yield "error", f"agent did not finish within {max_steps} steps"


if __name__ == "__main__":
    # Self-check: the loop parses actions, jails paths, and terminates. A canned
    # `generate` walks write_file -> read_file -> answer with no engine.
    import tempfile

    td = tempfile.mkdtemp()
    tools = Tools(td)
    script = [
        '{"thought": "make a file", "tool": "write_file", "args": {"path": "a.txt", "content": "hi"}}',
        '{"thought": "read it", "tool": "read_file", "args": {"path": "a.txt"}}',
        '{"answer": "done"}',
    ]
    it = iter(script)
    events = list(run_agent("go", lambda _m: next(it), tools))
    kinds = [k for k, _ in events]
    assert kinds == ["thought", "action", "observation", "thought", "action", "observation", "answer"], kinds
    assert ("answer", "done") in events
    # path jail
    try:
        tools.read_file("../../etc/passwd")
        raise SystemExit("path jail failed")
    except ValueError:
        pass
    # deny list
    try:
        tools.shell("rm -rf /")
        raise SystemExit("deny list failed")
    except ValueError:
        pass
    print("agent self-check OK")
