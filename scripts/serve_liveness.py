#!/usr/bin/env python3
"""Out-of-process liveness guard for scripts/serve_hybrid_v100.sh.

A /health 200 alone is not liveness: a wedged device launch froze the step loop
while stats() kept serving the last snapshot -- the false-200 the server now
answers as 503 with stuck_secs (#650). This guard is the recovery half: it polls
/health and tells the supervisor (by exit code) to kill and restart when

* two consecutive polls do not answer 200 -- HTTP 503 (step loop stalled), a
  refused connection and a timeout all count the same, because wedged and gone
  need the same restart; one failure then a 200 resets with no action;
* a fatal CUDA marker appears in the serve log -- immediate, no poll wait;
* three real short completions fail in a row while a slot is free -- fallback
  for a leak that answers /health but cannot generate.

Exit 10 = health/chat failure, 11 = fatal marker. The decision itself is the
pure restart_action(); restart_action()'s inputs are trivial to script, so the
gates need no server.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

HEALTH_TIMEOUT_S = float(os.environ.get("LIVENESS_HEALTH_TIMEOUT_S", "5"))
HEALTH_FAILS = int(os.environ.get("LIVENESS_HEALTH_FAILS", "2"))
CHAT_FAILS = int(os.environ.get("LIVENESS_CHAT_FAILS", "3"))
POLL_S = float(os.environ.get("LIVENESS_POLL_S", "60"))
MODEL = os.environ.get("LIVENESS_MODEL", "qwen38-27b")
MARKERS = ("CUDA error", "illegal memory access", "out of memory", "captures_underway")


def restart_action(health_fails, health_ok, marker, threshold=HEALTH_FAILS):
    """Restart decision for one poll.

    (consecutive health failures so far, this poll answered 200, fatal marker
    seen in new log bytes) -> (new failure count, action). Action is
    "restart-marker", "restart-health" or "". A marker wins over everything;
    a 200 resets the streak; a miss restarts only at the threshold, so one
    failure followed by a 200 is a recovery, not a restart.
    """
    if marker:
        return health_fails, "restart-marker"
    if health_ok:
        return 0, ""
    health_fails += 1
    return health_fails, ("restart-health" if health_fails >= threshold else "")


def poll_health(base, timeout=HEALTH_TIMEOUT_S):
    """(answered 200, stats dict or None). 503, any other HTTP error, a refused
    connection and a timeout all return (False, None) -- the caller's streak
    cannot distinguish wedged from gone, and must not."""
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as r:
            body = json.load(r)
        return 200 <= r.status < 300, body.get("stats")
    except urllib.error.HTTPError:
        return False, None
    except (urllib.error.URLError, OSError, TimeoutError):
        return False, None


def chat_ok(base, timeout=60.0):
    d = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Liveness {uuid.uuid4().hex}. Reply ok."}],
        "temperature": 0,
        "max_tokens": 4,
        "enable_thinking": False,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(d).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return bool(json.load(r)["choices"][0]["message"].get("content"))


def main(argv):
    log = argv[1]
    base = argv[2] if len(argv) > 2 else os.environ.get("LIVENESS_BASE", "http://127.0.0.1:8000")
    pos = os.path.getsize(log) if os.path.exists(log) else 0
    health_fails = 0
    gen_fails = 0
    while True:
        time.sleep(POLL_S)
        try:
            with open(log) as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            chunk = ""
        marker = any(m in chunk for m in MARKERS)
        ok, stats = poll_health(base)
        health_fails, action = restart_action(health_fails, ok, marker)
        if action:
            print(f"liveness: {action} at {time.strftime('%FT%T')}", flush=True)
            return 11 if action == "restart-marker" else 10
        # Slot-leak fallback: only while healthy and a slot is free, so four
        # long requests do not read as a broken server.
        if stats is not None and stats.get("slots_used", 0) < stats.get("slots_total", 1):
            try:
                good = chat_ok(base)
            except Exception:
                good = False
            if good:
                gen_fails = 0
                continue
            gen_fails += 1
            print(
                f"liveness: generation failure {gen_fails}/{CHAT_FAILS} "
                f"at {time.strftime('%FT%T')}",
                flush=True,
            )
            if gen_fails >= CHAT_FAILS:
                print("liveness: generation cannot progress, requesting restart", flush=True)
                return 10
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
