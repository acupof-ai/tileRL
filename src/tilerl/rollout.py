"""One command: serve tileRL, point Claude Code at it, collect the trajectory.

Stage 2 of the Claude Code RL line. A rollout is ``claude -p`` run against a
task checkout with ``ANTHROPIC_BASE_URL`` on our own server, and what comes back
is not the CLI's text but the shim's record rows -- the exact token ids that
were sampled, joined to the episode by a per-rollout header.

Three things are deliberate:

* **The server runs in this process, on a thread.** A rollout needs the engine
  and the recorder to agree about which request produced which ids; a second
  process would put a serialization hop between them for no gain. The port is
  ephemeral so K rollouts can run at once without a port map.
* **No Docker.** The pod is already a container and cannot nest one, and the
  Mac has no bwrap. Isolation is Claude Code's own sandbox (Seatbelt on macOS,
  bubblewrap on Linux) configured through ``--settings``; ``failIfUnavailable``
  makes it refuse to run rather than run bare. See :func:`sandbox_settings`.
* **The tag is a header, not ``metadata.user_id``.** Measured 2026-09-02:
  ``ANTHROPIC_CUSTOM_HEADERS`` reaches the server verbatim.

# ponytail: one rollout per process-with-a-thread. K parallel rollouts are K
# processes today (each with its own engine); a shared engine behind one server
# is the upgrade when engine startup, not generation, dominates.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
from typing import Any

__all__ = ["free_port", "sandbox_settings", "sandbox_available", "serve_app",
           "serve_background", "run_rollout", "read_records"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def sandbox_available() -> tuple[bool, str]:
    """Whether Claude Code's sandbox can actually run here, and why not.

    macOS always can (Seatbelt ships with the OS). Linux needs bubblewrap plus
    a usable user namespace; probing ``unshare -Ur`` is the honest test because
    a container can carry the binary and still refuse the syscall.
    """
    if platform.system() == "Darwin":
        return bool(shutil.which("sandbox-exec")), "seatbelt"
    if not shutil.which("bwrap"):
        return False, "bubblewrap not installed"
    try:
        rc = subprocess.run(["unshare", "-Ur", "true"], capture_output=True, timeout=10).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"unshare probe failed: {exc}"
    return (rc == 0), "bubblewrap + userns" if rc == 0 else "user namespaces denied"


def sandbox_settings(host: str, port: int) -> dict[str, Any]:
    """Claude Code's sandbox config for a rollout: our server and nothing else.

    ``failIfUnavailable`` is the important key -- without it a host that cannot
    sandbox runs the agent bare, and a rollout that touched the real filesystem
    is worse than a rollout that did not happen.
    """
    return {
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "enableWeakerNestedSandbox": True,
            "filesystem": {"denyRead": ["~/.claude", "~/.claude.json"]},
            "network": {"allowedDomains": [f"{host}:{port}"], "strictAllowlist": True},
        }
    }


def serve_app(app: Any, host: str = "127.0.0.1", port: int | None = None,
              timeout_s: float = 60.0) -> tuple[str, threading.Thread]:
    """Run any ASGI app on a daemon thread; return (base_url, thread) once it accepts.

    Split out from :func:`serve_background` so a test can serve a scripted
    engine over a real socket -- the launcher's gate must not need weights.
    """
    import uvicorn

    port = port or free_port()
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return f"http://{host}:{port}", thread
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"server did not accept connections on port {port} within {timeout_s}s")


def serve_background(model: str = "tiny-agent", host: str = "127.0.0.1",
                     port: int | None = None) -> tuple[str, Any, threading.Thread]:
    """Start the tileRL server on a daemon thread; return (base_url, engine, thread)."""
    from tilerl_kernels.backend import get_backend

    from .cli import _build_engine, _build_model
    from .server import create_app, get_tokenizer

    cfg, model_obj = _build_model(model, seed=0, fuse_projections=True)
    engine = _build_engine(cfg, model_obj, get_backend())
    app = create_app(engine, get_tokenizer(), model_name=cfg.name)
    engine.run()
    base, thread = serve_app(app, host, port)
    return base, engine, thread


def run_rollout(task: str, cwd: str, base_url: str, tag: str, *, sandbox: bool = True,
                tools: list[str] | None = None, timeout_s: float = 900.0,
                max_turns: int | None = None) -> dict[str, Any]:
    """Run one `claude -p` episode against ``base_url``; return the CLI's result.

    ``tag`` is echoed into every record row this episode writes, so the token
    ids can be grouped into a trajectory afterwards without parsing transcripts.
    """
    host_port = base_url.split("://", 1)[-1]
    host, _, port = host_port.partition(":")
    cmd = ["claude", "-p", task, "--output-format", "json"]
    if tools:
        cmd += ["--allowed-tools", *tools]
    if max_turns is not None:
        cmd += ["--max-turns", str(max_turns)]
    if sandbox:
        ok, why = sandbox_available()
        if not ok:
            raise RuntimeError(
                f"sandbox unavailable ({why}); pass sandbox=False to run unisolated"
            )
        cmd += ["--settings", json.dumps(sandbox_settings(host, int(port or 80)))]
    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "tilerl-local",
        "ANTHROPIC_CUSTOM_HEADERS": f"x-tilerl-rollout: {tag}",
        "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1",
    }
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout_s)
    try:
        result = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        result = {"is_error": True, "result": proc.stdout}
    result["returncode"] = proc.returncode
    result["stderr"] = proc.stderr[-2000:]
    result["rollout"] = tag
    return result


def read_records(path: str, tag: str) -> list[dict[str, Any]]:
    """The record rows belonging to one episode, in the order they were served."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("rollout") == tag:
                rows.append(row)
    return rows


if __name__ == "__main__":  # pragma: no cover - self-check
    s = sandbox_settings("127.0.0.1", 8000)["sandbox"]
    assert s["failIfUnavailable"] and not s["allowUnsandboxedCommands"]
    assert s["network"]["allowedDomains"] == ["127.0.0.1:8000"]
    p = free_port()
    assert 1024 < p < 65536
    print("rollout self-check ok:", sandbox_available())
