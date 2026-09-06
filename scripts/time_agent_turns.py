"""Time an agent's turns against a tileRL endpoint, one row per request.

Why a proxy and not the recorder: `runs/messages_requests.jsonl` carries no timestamp
(keys checked on the live V100, 2026-09-07), so "wall clock per turn" has no server-side
evidence source. It also cannot see a request the engine REFUSED -- the 400 is raised
inside `submit`, before `_record`, which is how the first trial's failing turns left no
trace at all.

This sits between the client and the endpoint, so it times what the client experienced and
records every request including the refused ones. Read-only with respect to the server: it
forwards the body verbatim and returns the response verbatim.

Usage:
    uv run python scripts/time_agent_turns.py --upstream http://10.37.2.27:8000 &
    ANTHROPIC_BASE_URL=http://127.0.0.1:8787 ANTHROPIC_API_KEY=x claude -p "..."
    # then read the printed summary, or the JSONL at --out

One row per request: seconds, status, prompt/completion sizes, stop_reason, and whether the
reply carried a tool_use block -- which is what makes the round-trip count a count rather
than an impression.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ROWS: list[dict] = []
_UPSTREAM = ""
_OUT = ""


class _Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, method: str) -> None:
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else None
        url = _UPSTREAM.rstrip("/") + self.path
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection")}
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=1900) as r:
                payload, status = r.read(), r.status
                ctype = r.headers.get("content-type", "application/json")
        except urllib.error.HTTPError as e:
            payload, status, ctype = e.read(), e.code, "application/json"
        except Exception as exc:  # a transport failure is a turn outcome too
            payload = json.dumps({"proxy_error": str(exc)}).encode()
            status, ctype = 599, "application/json"
        secs = time.monotonic() - t0

        # `t0_epoch`/`t1_epoch`, not just the duration: a turn has to be alignable against
        # the pool sampler's rows, and a duration alone cannot say which samples fell inside.
        row: dict = {"path": self.path, "status": status, "secs": round(secs, 3),
                     "t1_epoch": round(time.time(), 3),
                     "t0_epoch": round(time.time() - secs, 3)}
        if body:
            try:
                sent = json.loads(body)
                row["asked_max_tokens"] = sent.get("max_tokens")
                row["messages"] = len(sent.get("messages") or [])
                row["tools"] = len(sent.get("tools") or [])
            except json.JSONDecodeError:
                pass
        # stop_reason and tool_use make the round-trip count a count. A streamed reply is
        # SSE, so parse only what is JSON and leave the rest unclaimed rather than guessed.
        try:
            got = json.loads(payload)
            row["stop_reason"] = got.get("stop_reason")
            row["tool_use"] = sum(1 for b in got.get("content") or []
                                  if b.get("type") == "tool_use")
            if status >= 400:
                row["error"] = str(got.get("error") or got)[:200]
        except (json.JSONDecodeError, AttributeError):
            row["streamed"] = True
        _ROWS.append(row)
        # Appended as it happens, not at exit: the first live trial lost every client-side
        # number because the rows were written only after serve_forever returned, and the
        # signal that would have ended it never reached this process under `uv run`+nohup.
        if _OUT:
            with open(_OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        self._forward("POST")

    def do_GET(self) -> None:
        self._forward("GET")

    def log_message(self, *_a) -> None:
        return


def main() -> int:
    global _UPSTREAM, _OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--out", default="/tmp/agent-turns.jsonl")
    a = ap.parse_args()
    _UPSTREAM, _OUT = a.upstream, a.out
    open(_OUT, "w").close()  # truncate: rows are appended per request from here on

    # SIGTERM too, not only ctrl-c: under `uv run`/nohup the interrupt reaches the wrapper
    # and this process dies unhandled, which is how the first trial lost its summary.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), _Proxy)
    print(f"proxy on http://127.0.0.1:{a.port} -> {a.upstream}; rows stream to {a.out}",
          flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()

    posts = [r for r in _ROWS if r["path"].startswith("/v1/")]
    ok = [r for r in posts if r["status"] == 200]
    print(f"\nrequests {len(posts)}  ok {len(ok)}  failed {len(posts) - len(ok)}")
    if ok:
        secs = sorted(r["secs"] for r in ok)
        print(f"seconds per turn: min {secs[0]:.2f}  median {secs[len(secs) // 2]:.2f}  "
              f"max {secs[-1]:.2f}  total {sum(secs):.2f}")
    print(f"tool_use blocks returned: {sum(r.get('tool_use') or 0 for r in _ROWS)}")
    for r in posts:
        if r["status"] != 200:
            print(f"  FAILED {r['status']} {r.get('error', '')[:120]}")
    print(f"rows: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
