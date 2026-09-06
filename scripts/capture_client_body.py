"""Capture one real Claude Code request body and name every field we do not declare.

Why this exists: `unknown_fields` (#201) reports what a request carried, so it finds a
field only once a client happens to send it -- `tool_choice` surfaced that way, one field
at a time, on a route that had been live for days. This drives the real CLI against a stub
that records the whole body, so the answer is the full set in one shot instead of a
discovery per turn.

No GPU and no engine: the stub answers with a minimal valid Messages reply, which is all
the CLI needs to send its first request.

Usage:
    uv run python scripts/capture_client_body.py            # prints the field diff
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_BODIES: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n)
        try:
            _BODIES.append(json.loads(raw))
        except json.JSONDecodeError:
            _BODIES.append({"_unparseable": raw[:200].decode("utf-8", "replace")})
        # A minimal valid reply: the CLI only needs one to have sent its first request.
        body = json.dumps({
            "id": "msg_stub", "type": "message", "role": "assistant", "model": "stub",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # /v1/models and friends
        self.send_response(200)
        self.send_header("content-length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_a) -> None:
        return  # the field diff is the output; access lines are noise


def _declared() -> set[str]:
    from tilerl.messages import MessagesRequest
    return set(MessagesRequest.model_fields)


def main() -> int:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    env = {**os.environ, "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
           "ANTHROPIC_API_KEY": "capture", "ANTHROPIC_MODEL": "stub"}
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["claude", "-p", "say ok"], cwd=d, env=env,
                       capture_output=True, timeout=180)
    srv.shutdown()

    if not _BODIES:
        print("no request captured -- the CLI never reached the stub", file=sys.stderr)
        return 1

    declared = _declared()
    seen: dict[str, str] = {}
    for b in _BODIES:
        for k, v in b.items():
            seen.setdefault(k, type(v).__name__)
    undeclared = {k: t for k, t in sorted(seen.items()) if k not in declared}

    print(f"requests captured: {len(_BODIES)}")
    print(f"fields the CLI sent ({len(seen)}): {', '.join(sorted(seen))}")
    print(f"declared on MessagesRequest ({len(declared)}): {', '.join(sorted(declared))}")
    print(f"UNDECLARED ({len(undeclared)}): {json.dumps(undeclared, indent=2)}")
    # Not an assert: a new CLI version legitimately adds fields, and this script is how
    # you find out. The count is the finding.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
