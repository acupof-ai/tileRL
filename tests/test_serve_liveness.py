"""Gates for scripts/serve_liveness.py's restart decision.

The decision is the pure restart_action(); its two inputs (a 200 and a fatal
marker) are scripted, so no server is needed except the two poll_health
transport branches that cannot be faked: a real 503 and a connection nobody
accepts. Both must read as not-healthy with the SAME answer as a stall, since
wedged and gone need the same restart.
"""

from __future__ import annotations

import http.server
import importlib.util
import socketserver
import threading
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "serve_liveness.py"


def _load():
    spec = importlib.util.spec_from_file_location("serve_liveness", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_healthy_poll_does_not_restart_and_resets_the_streak():
    m = _load()
    assert m.restart_action(0, True, False) == (0, "")
    # one miss already counted, then a 200: a recovery resets, not a restart
    assert m.restart_action(1, True, False) == (0, "")


def test_one_miss_is_tolerated_two_consecutive_misses_restart():
    m = _load()
    fails, action = m.restart_action(0, False, False)
    assert (fails, action) == (1, "")
    fails, action = m.restart_action(fails, False, False)
    assert (fails, action) == (2, "restart-health")


def test_fatal_marker_restarts_immediately_even_while_healthy():
    m = _load()
    # marker wins over a 200 and over a streak of zero
    assert m.restart_action(0, True, True) == (0, "restart-marker")
    assert m.restart_action(4, False, True)[1] == "restart-marker"


class _503(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(503)
        self.end_headers()
        self.wfile.write(b'{"status":"unhealthy"}')

    def log_message(self, *a):
        pass


def test_poll_health_503_and_refused_connection_both_read_unhealthy():
    m = _load()

    httpd = socketserver.TCPServer(("127.0.0.1", 0), _503)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        assert m.poll_health(f"http://127.0.0.1:{port}", timeout=5) == (False, None)
    finally:
        httpd.shutdown()

    # A closed local port refuses immediately; that answer must be identical to
    # the 503, or a crashed child would look different from a wedged one.
    s = socketserver.TCPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    closed_port = s.server_address[1]
    s.server_close()
    assert m.poll_health(f"http://127.0.0.1:{closed_port}", timeout=5) == (False, None)
