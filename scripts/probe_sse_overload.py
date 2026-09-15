#!/usr/bin/env python3
"""Real-curl verification on the live hybrid serve (reads :8000, no deploy):

1. In-flight client disconnect -> engine.cancel, slot + blocks eventually
   released (pre-fix the row leaked to max_new). TWO arms, because #637 moves
   every engine.cancel off the event loop, not just SSE:
     * SSE: one first-frame rep plus late-frame (frame ~45) reps;
     * NON-STREAM: stream=false, the reply is never read, socket closed
       mid-decode (await_or_cancel / CancelledError).
   Both share two hard PASS gates: in-flight /health latency stays < 0.5 s
   (event loop alive) and running/slots/blocks end at zero (zero leak, no
   wall-clock cap). Release delay is recorded as a distribution only -- after
   cancel moves off the event loop it can still wait behind a multi-second
   engine tick; that is the separate slow-tick OPEN defect, not a failure of
   this fix.
   The /ws/chat arm is NOT in this stdlib probe (the standard library has no
   websocket client); run it manually with the third-party `websockets`
   package using the same two gates (see the errors entry for the after run).
2. inflight pinned at the cap (cap = 2*usable_slots = 8); each further submit on
   chat/messages/responses, stream and non-stream, must be HTTP 503
   overloaded_error with integer inflight/cap = 8, never 429, no retry_after.
3. Normal short chat 200 before and after; capacity recovers after the pin.

Memory envelope: holder rows are 250 words (~1k dense prompt tokens) x 16 new
tokens -- the proven-clean smoke_dense_d1_boundary shape (32/32 rows, four
concurrent decodes max). Two probe-sizing mistakes OOM-killed the 31.7 GiB
V100's serve during development (three liveness restarts, all
torch.OutOfMemoryError with tens of MiB free -- never a code defect in the
serve):
  * max_tokens 512 on four concurrent rows grew past the marginal headroom;
  * max_tokens 32-64 ALSO OOMed when the pin churned (rows finishing and
    re-prefilling continuously) -- the danger is concurrent decode length plus
    refill prefill churn, not just per-row length. Waiting rows hold no KV, so
    the cap=8 inflight (4 running + 4 waiting) costs no more than 4 rows.
Do not lengthen holder generations or add extra concurrent submitters.

Probe rules (all learned the hard way on this box):
  * Before the "9th submit" probes, read /health and require EXACTLY
    running=4 waiting=4, then fire all six probes through one barrier. Holder
    rows are only 16 tokens and finish fast; once the waiting deque empties the
    next submit is admitted and answers 200, which is a probe timing bug, not a
    503 regression.
  * A streamed response checks the STATUS LINE only: read one SSE line and
    stop. Never read()/drain the whole SSE body for a status assertion -- that
    blocks until the row (or the whole pin) drains and turns an expected 503/200
    handshake into a multi-second hang. post() already enforces this.
  * The cancel/SSE timing is the real latency signal; run it after the
    to_thread cancel fix lands and require in-flight /health latency < 0.5 s
    during a disconnect (pre-fix event-loop freeze measured up to 5.57 s).

Stdlib only. Env: BASE (default http://127.0.0.1:8000), MODEL.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "qwen38-27b")
_host, _port = BASE.split("://")[1].split(":")
PORT = int(_port)
PROMPT = "serendipity " * 250  # ~1k dense prompt tokens


def log(s: str) -> None:
    print(f"{time.time():.3f} {s}", flush=True)


def _safe_close(c: http.client.HTTPConnection | None) -> None:
    if c is None:
        return
    with contextlib.suppress(Exception):  # noqa: BLE001 - probe teardown is best-effort
        c.close()


def _hangup(c: http.client.HTTPConnection) -> None:
    """Hard client hang-up: shutdown then close, both best-effort."""
    with contextlib.suppress(Exception):  # noqa: BLE001
        c.sock.shutdown(2)
    _safe_close(c)


def stats(timeout: float = 30.0) -> dict:
    c = http.client.HTTPConnection(_host, PORT, timeout=timeout)
    c.request("GET", "/health")
    r = c.getresponse()
    d = json.loads(r.read())
    c.close()
    return d["stats"]


def post(path: str, body: dict, stream: bool, timeout: float = 30.0):
    """A streamed 200 reads only the handshake: the refusal is the status line."""
    c = http.client.HTTPConnection(_host, PORT, timeout=timeout)
    c.request("POST", path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = c.getresponse()
    if stream and r.status == 200:
        r.readline()
        data = b""
    else:
        data = r.read()
    return r.status, {k.lower(): v for k, v in r.getheaders()}, data, c


def short_sanity(tag: str) -> bool:
    s0 = stats()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"nonce {uuid.uuid4().hex} say ok"}],
        "temperature": 0,
        "max_tokens": 4,
        "stream": False,
    }
    st, _, data, c = post("/v1/chat/completions", body, False)
    c.close()
    ntok = None
    if st == 200:
        ntok = json.loads(data)["usage"]["completion_tokens"]
    s1 = stats()
    ok = st == 200 and ntok and ntok >= 1
    log(
        f"sanity {tag}: HTTP {st} tokens={ntok}; "
        f"running {s0['running']}->{s1['running']} blocks {s0['blocks_used']}->{s1['blocks_used']}"
    )
    return ok


def _wait_zero(t0: float, tag: str, base_blocks: int) -> float | None:
    """Release is NOT wall-clock gated: after cancel moves off the event loop a
    multi-second slow engine tick still postpones row reclamation (the slow
    OPEN). Wait generously for the IDLE state; only a hard 60 s cap guards a
    genuine hang, which would be a real failure.

    blocks returns to the pre-arm IDLE baseline, not necessarily literal zero:
    a completed request can legitimately leave a resident prefix block pinned
    (blocks_used=1 with no running row or slot is normal after traffic), so
    absolute zero is the wrong no-leak signal on a warm serve."""

    def idle(s: dict) -> bool:
        return s["running"] == 0 and s.get("slots_used", 0) == 0 and s["blocks_used"] == base_blocks

    for _ in range(1200):  # 60 s, 0.05 s poll
        s = stats()
        if idle(s):
            dt = time.time() - t0
            time.sleep(0.5)
            if not idle(stats()):
                log(f"{tag} FAIL: counters reached idle then rose again")
                return None
            return dt
        time.sleep(0.05)
    log(
        f"{tag} FAIL: row never released to idle baseline within 60 s "
        f"(true leak/hang; base blocks {base_blocks})"
    )
    return -1.0


def sse_once(frames: int = 1) -> float | None:
    """Seconds from socket close to idle state; None on handshake failure.
    frames=1 disconnects on the first content frame (the classic SSE leak);
    larger values disconnect mid-generation (the late-tick lock tail)."""
    base_blocks = stats()["blocks_used"]
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"nonce {uuid.uuid4().hex} {PROMPT}"}],
        "temperature": 0,
        "max_tokens": 400,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    c = http.client.HTTPConnection(_host, PORT, timeout=60)
    c.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    r = c.getresponse()
    if r.status != 200:
        log(f"SSE FAIL handshake HTTP {r.status}: {r.read()[:200]!r}")
        c.close()
        return None
    seen = 0
    while True:
        line = r.readline()
        if not line:
            log("SSE FAIL: closed before the target content frame")
            c.close()
            return None
        if line.startswith(b"data:") and b"content" in line:
            seen += 1
            if seen >= frames:
                break
    t0 = time.time()
    _hangup(c)  # hard hang-up right after the target content frame
    return _wait_zero(t0, f"SSE(frame {frames})", base_blocks)


def nonstream_once(dwell_s: float = 3.0) -> float | None:
    base_blocks = stats()["blocks_used"]
    """Submit a NON-stream request with a long generation and hang up while it
    is parked mid-decode (never reading the JSON reply). The server's
    await_or_cancel path must observe http.disconnect and cancel off the event
    loop. Returns seconds from socket close to zero state."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"nonce {uuid.uuid4().hex} {PROMPT}"}],
        "temperature": 0,
        "max_tokens": 400,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    c = http.client.HTTPConnection(_host, PORT, timeout=60)
    c.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    # Wait until the row is admitted and decoding, then dwell into the late
    # decode region where the long ticks appeared (~frame 40).
    end = time.time() + 30
    while time.time() < end:
        if stats()["running"] >= 1:
            break
        time.sleep(0.05)
    else:
        log("NONSTREAM FAIL: request never became running before disconnect")
        c.close()
        return None
    time.sleep(dwell_s)
    t0 = time.time()
    _hangup(c)
    return _wait_zero(t0, "NONSTREAM", base_blocks)


_SAMPLER_SRC = (
    "import http.client,json,sys,time\n"
    "h,p=sys.argv[1].split('://')[1].split(':');p=int(p)\n"
    "while True:\n"
    " t=time.time()\n"
    " try:\n"
    "  c=http.client.HTTPConnection(h,p,timeout=10);c.request('GET','/health')\n"
    "  json.loads(c.getresponse().read());c.close()\n"
    " except Exception: pass\n"
    # emit "<epoch-of-response-start> <latency-s>" so stalls align to wall clock
    " sys.stdout.write(f'{t:.3f} {time.time()-t:.3f}\\n');sys.stdout.flush()\n"
    " time.sleep(0.05)\n"
)


class HealthSampler:
    """Polls /health ~every 50 ms and records worst latency from a SEPARATE
    subprocess. It must be its own process, not a thread: under the probe's own
    holder/client thread load the GIL stalled an in-process sampler to 0.94 s
    while an independent curl loop measured 0.044 s -- client-side contention,
    not the server. stats() is lock-free, so a healthy event loop answers in
    tens of ms even while a long engine tick holds the lock; a slow independent
    answer means the LOOP itself is blocked (pre-fix the on-loop engine.cancel
    froze it, measured up to 5.57 s). Gate: /health < 0.5 s during disconnects."""

    def __init__(self, keep_path: str = "") -> None:
        self._keep = keep_path
        if keep_path:
            self._path = keep_path
            with contextlib.suppress(FileNotFoundError):
                os.remove(keep_path)
        else:
            fd, self._path = tempfile.mkstemp(suffix=".log", prefix="hlth")
            os.close(fd)
        # Popen dup's the fd at spawn; closing our copy with `with` leaves the
        # child holding its own write end.
        with open(self._path, "w") as out:
            self.proc = subprocess.Popen(
                [sys.executable, "-c", _SAMPLER_SRC, BASE], stdout=out, stderr=subprocess.DEVNULL
            )
        self._start = len(self._read())

    def _read(self) -> list[tuple[float, float]]:
        vals: list[tuple[float, float]] = []
        with open(self._path) as f:
            for line in f:
                parts = line.split()
                try:
                    if len(parts) == 2:
                        vals.append((float(parts[0]), float(parts[1])))
                    else:  # tolerate an old single-latency line
                        vals.append((0.0, float(parts[0])))
                except (ValueError, IndexError):
                    pass
        return vals

    def samples(self) -> list[tuple[float, float]]:
        return self._read()[self._start :]

    def latencies(self) -> list[float]:
        return [lat for _, lat in self.samples()]

    def mark(self) -> int:
        # index into the post-construction list samples() returns
        return len(self.samples())

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if not self._keep:
            os.unlink(self._path)


def _summarize(
    label: str, times: list[float], n_plan: int, lat_window: list[float], health_gate: float
) -> bool:
    worst = max(lat_window, default=0.0)
    health_ok = worst < health_gate
    leak_ok = len(times) == n_plan and all(t is not None for t in times)
    slow = [round(t, 3) for t in times if t > 2.0]
    log(
        f"{label} release times (s): {[round(t, 3) for t in times]} "
        f"min={min(times):.3f} max={max(times):.3f}"
    )
    log(f"{label} /health worst latency during disconnects: {worst:.3f}s (gate <{health_gate}s)")
    if slow:
        # Not a failure: moving cancel off the event loop does not bound an
        # engine tick. Report these as measured evidence for the slow-tick OPEN.
        log(f"{label} release >2s samples (slow-tick OPEN evidence, not gated): {slow}")
    ok = leak_ok and health_ok
    log(
        f"{label} -> {'PASS' if ok else 'FAIL'} "
        f"(all rows released to zero={leak_ok}, in-flight health<{health_gate}s={health_ok})"
    )
    return ok


def disconnect_check(reps: int = 6, nonstream_reps: int = 3, health_gate: float = 0.5) -> bool:
    # One SSE sampler spans BOTH arms: the gate is that the event loop stays
    # alive across every disconnect path #637 moved off the loop.
    sampler = HealthSampler()
    overall = True
    try:
        # SSE: rep 0 on frame 1 (the original leak shape), the rest late in
        # decode (frame ~45), where the cancel-on-event-loop lock tail showed up.
        frame_plan = [1] + [45] * (reps - 1)
        times = []
        m0 = sampler.mark()
        for i, frames in enumerate(frame_plan):
            dt = sse_once(frames)
            if dt is None or dt < 0:
                log(f"SSE rep {i} (frame {frames}): FAIL ({dt})")
                overall = False
                break
            times.append(dt)
            log(f"SSE rep {i} (frame {frames}): released {dt:.3f}s after close")
            time.sleep(0.5)
        if len(times) == len(frame_plan):
            overall &= _summarize(
                "SSE", times, len(frame_plan), sampler.latencies()[m0:], health_gate
            )
        else:
            overall = False

        # Non-stream: submit with stream=false, never read the reply, hang up
        # mid-decode (await_or_cancel / CancelledError path).
        ntimes = []
        mn = sampler.mark()
        for i in range(nonstream_reps):
            dt = nonstream_once(dwell_s=3.0)
            if dt is None or dt < 0:
                log(f"NONSTREAM rep {i}: FAIL ({dt})")
                overall = False
                break
            ntimes.append(dt)
            log(f"NONSTREAM rep {i}: released {dt:.3f}s after close")
            time.sleep(0.5)
        if len(ntimes) == nonstream_reps:
            overall &= _summarize(
                "NONSTREAM", ntimes, nonstream_reps, sampler.latencies()[mn:], health_gate
            )
        else:
            overall = False
    finally:
        sampler.stop()
    return overall


class Pool:
    """Keeps inflight pinned at cap. Holders stream short (250-word, 16-token)
    rows and immediately re-submit when one drains. Backstops hammer submit in
    a tight loop: every 503 is ignored, and while the waiting deque never
    empties, running+waiting is structurally the cap (a finish admits one
    waiting in the same step)."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.stop = threading.Event()
        self.refused = 0
        self.conns: list[http.client.HTTPConnection] = []
        self._clock = threading.Lock()
        self.threads = [threading.Thread(target=self._holder, daemon=True) for _ in range(cap)]
        self.threads += [threading.Thread(target=self._backstop, daemon=True) for _ in range(4)]

    def _holder(self) -> None:
        while not self.stop.is_set():
            c = None
            try:
                c = http.client.HTTPConnection(_host, PORT, timeout=120)
                body = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": f"nonce {uuid.uuid4().hex} {PROMPT}"}],
                    "temperature": 0,
                    "max_tokens": 16,
                    "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                c.request(
                    "POST",
                    "/v1/chat/completions",
                    json.dumps(body).encode(),
                    {"Content-Type": "application/json"},
                )
                r = c.getresponse()
                if r.status != 200:
                    r.read()
                    c.close()
                    if r.status == 503:
                        with self._clock:
                            self.refused += 1
                    self.stop.wait(0.1)
                    continue
                with self._clock:
                    self.conns.append(c)
                while not self.stop.is_set() and r.readline():
                    pass
                c.close()
            except Exception:  # noqa: BLE001 - tight envelope, re-loop
                _safe_close(c)
                self.stop.wait(0.1)

    def _backstop(self) -> None:
        while not self.stop.is_set():
            c = None
            try:
                c = http.client.HTTPConnection(_host, PORT, timeout=30)
                body = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": f"nonce {uuid.uuid4().hex} {PROMPT}"}],
                    "temperature": 0,
                    "max_tokens": 16,
                    "stream": False,
                }
                c.request(
                    "POST",
                    "/v1/chat/completions",
                    json.dumps(body).encode(),
                    {"Content-Type": "application/json"},
                )
                r = c.getresponse()
                r.read()
                c.close()
                if r.status == 503:
                    with self._clock:
                        self.refused += 1
                else:
                    # Admitted (pin momentarily below cap): its completion still
                    # counts, just sleep it off to avoid adding load.
                    self.stop.wait(0.2)
            except Exception:  # noqa: BLE001
                _safe_close(c)
                self.stop.wait(0.1)

    def start(self) -> None:
        for t in self.threads:
            t.start()
            time.sleep(0.05)

    def shutdown(self) -> None:
        self.stop.set()
        with self._clock:
            conns = list(self.conns)
        for c in conns:  # SSE hang-up on every parked stream -> cancel
            _hangup(c)


def ninth_probe(
    gate: threading.Event, barrier: threading.Barrier, spec: tuple, results: dict
) -> None:
    name, mode, path, body = spec
    gate.wait()  # main opens it only at a fresh running=4 waiting=4 reading
    barrier.wait(2.0)  # all six hit submit within the same saturated window
    try:
        st, hdr, data, c = post(path, body, bool(body.get("stream")), timeout=15)
        c.close()
        try:
            err = json.loads(data).get("error", {})
        except Exception:  # noqa: BLE001
            err = {"_raw": data[:200].decode(errors="replace")}
        r = {
            "status": st,
            "type": err.get("type"),
            "inflight": err.get("inflight"),
            "cap": err.get("cap"),
            "retry_after": "retry-after" in hdr,
            "raw": data[:200].decode(errors="replace"),
        }
    except Exception as e:  # noqa: BLE001
        r = {
            "status": -1,
            "type": None,
            "inflight": None,
            "cap": None,
            "retry_after": False,
            "raw": f"{type(e).__name__}: {e}",
        }
    results[(name, mode)] = r


def overload(cap: int = 8) -> bool:
    pool = Pool(cap)
    pool.start()
    pinned = None
    end = time.time() + 60
    while time.time() < end:
        s = stats()
        if s["running"] == 4 and s["waiting"] == 4:
            pinned = s
            break
        time.sleep(0.1)
    if pinned is None:
        s = stats()
        log(
            f"OVERLOAD FAIL: never read running=4 waiting=4: "
            f"running={s['running']} waiting={s['waiting']}"
        )
        pool.shutdown()
        return False
    log(
        f"pinned: running={pinned['running']} waiting={pinned['waiting']} "
        f"slots_used={pinned['slots_used']}"
    )

    nonce = f"nonce {uuid.uuid4().hex}"
    specs = [
        (
            "chat",
            "ns",
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": nonce}],
                "max_tokens": 1,
                "stream": False,
            },
        ),
        (
            "chat",
            "s",
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": nonce}],
                "max_tokens": 1,
                "stream": True,
            },
        ),
        (
            "messages",
            "ns",
            "/v1/messages",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": nonce}],
                "max_tokens": 1,
                "stream": False,
            },
        ),
        (
            "messages",
            "s",
            "/v1/messages",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": nonce}],
                "max_tokens": 1,
                "stream": True,
            },
        ),
        (
            "responses",
            "ns",
            "/v1/responses",
            {"model": MODEL, "input": nonce, "max_output_tokens": 1, "stream": False},
        ),
        (
            "responses",
            "s",
            "/v1/responses",
            {"model": MODEL, "input": nonce, "max_output_tokens": 1, "stream": True},
        ),
    ]
    results: dict = {}
    gate = threading.Event()  # main sets it only on running=4 waiting=4
    barrier = threading.Barrier(6)
    workers = [
        threading.Thread(target=ninth_probe, args=(gate, barrier, spec, results), daemon=True)
        for spec in specs
    ]
    for w in workers:
        w.start()
    # Open the gate only at a fresh exact running=4 waiting=4 reading, then let
    # all six through simultaneously; a 16-token holder row cannot drain in the
    # milliseconds between the reading and the six submits.
    deadline = time.time() + 30
    while time.time() < deadline:
        s = stats()
        if s["running"] == 4 and s["waiting"] == 4:
            gate.set()
            break
        time.sleep(0.02)
    else:
        log("OVERLOAD FAIL: never read running=4 waiting=4 before the barrier")
        gate.set()
        pool.shutdown()
        return False
    for w in workers:
        w.join(30)
    pool.shutdown()

    allok = True
    for name, mode, _, _ in specs:
        r = results.get((name, mode))
        if r is None:
            log(f"9th {name:9s} {mode}: NO RESULT -> FAIL")
            allok = False
            continue
        ok = (
            r["status"] == 503
            and r["type"] == "overloaded_error"
            and r["inflight"] == cap
            and r["cap"] == cap
            and isinstance(r["inflight"], int)
            and not r["retry_after"]
        )
        allok &= ok
        log(
            f"9th {name:9s} {mode}: HTTP {r['status']} type={r['type']} "
            f"inflight={r['inflight']} cap={r['cap']} retry_after={r['retry_after']} "
            f"-> {'PASS' if ok else 'FAIL ' + r['raw']}"
        )

    drained = False
    end = time.time() + 30
    while time.time() < end:
        s = stats()
        if s["running"] == 0 and s["waiting"] == 0 and s.get("slots_used", 0) == 0:
            drained = True
            break
        time.sleep(0.1)
    log(
        f"after shutdown: drained={drained} running={s['running']} waiting={s['waiting']} "
        f"slots={s.get('slots_used')} blocks_used={s['blocks_used']} "
        f"(pool observed {pool.refused} 503s)"
    )
    return allok and drained


def normal_gen_once(gen: int = 120) -> int:
    """Stream a full generation to completion with NO disconnect; returns the
    number of SSE lines consumed. Paired with the HealthSampler it is the
    control that separates a disconnect-correlated /health stall from a generic
    long prefill/decode tick."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"nonce {uuid.uuid4().hex} {PROMPT}"}],
        "temperature": 0,
        "max_tokens": gen,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    conn = http.client.HTTPConnection(_host, PORT, timeout=120)
    conn.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    r = conn.getresponse()
    n = 0
    if r.status == 200:
        for _ in range(gen * 4):
            if not r.readline():
                break
            n += 1
    conn.close()
    return n


def long_prefill_once(words: int = 250, max_tokens: int = 4) -> float:
    """One uninterrupted NON-stream request on a long prompt, timing just the
    wait for the first token region (prefill), never disconnecting. The JIT-vs-
    real-kernel control: run it AFTER the disconnect load so kernels are warm;
    if a same-length prefill is still multi-second the cost is the kernel, not
    first-batch warmup. Returns wall seconds from submit to response."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": f"nonce {uuid.uuid4().hex} {'serendipity ' * words}"}
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    c = http.client.HTTPConnection(_host, PORT, timeout=180)
    t0 = time.time()
    c.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    r = c.getresponse()
    r.read()
    dt = time.time() - t0
    c.close()
    return dt


def timing_run(sse_reps: int, nonstream_reps: int, health_log: str = "") -> bool:
    """Root-cause data collection for the slow-tick OPEN, not a pass/fail gate:
    one independent-subprocess /health sampler (kept at health_log with
    '<epoch> <latency-s>' per line) spans (a) SSE disconnects,
    (b) non-stream disconnects, (c) an uninterrupted decode control, and
    (d) a warm, same-length uninterrupted PREFILL control. Logs the worst
    /health latency and every >=0.5 s / >=2 s sample's epoch per phase so fixkv
    can align them to servetiming.log. Release delay is printed as-is."""
    sampler = HealthSampler(keep_path=health_log)

    def phase_report(name: str, a: int, b: int) -> float:
        window = sampler.samples()[a:b]
        worst_v = max((lat for _, lat in window), default=0.0)
        over05 = [(round(ts, 3), round(lat, 3)) for ts, lat in window if lat >= 0.5]
        over2 = [(round(ts, 3), round(lat, 3)) for ts, lat in window if lat >= 2.0]
        log(f"TIMING {name}: /health worst={worst_v:.3f}s n={len(window)}")
        if over05:
            log(f"TIMING {name}: >=0.5s samples (epoch,lat) {over05}")
        if over2:
            log(f"TIMING {name}: >=2.0s samples (epoch,lat) {over2}")
        return worst_v

    try:
        m0 = sampler.mark()
        frame_plan = [1] + [45] * (sse_reps - 1)
        rel_sse = []
        for i, frames in enumerate(frame_plan):
            dt = sse_once(frames)
            rel_sse.append(None if dt is None else round(dt, 3))
            log(f"timing SSE rep {i} (frame {frames}): release {dt}")
            time.sleep(0.4)
        m1 = sampler.mark()
        rel_ns = []
        for i in range(nonstream_reps):
            dt = nonstream_once(dwell_s=3.0)
            rel_ns.append(None if dt is None else round(dt, 3))
            log(f"timing NONSTREAM rep {i}: release {dt}")
            time.sleep(0.4)
        m2 = sampler.mark()
        lines = normal_gen_once()
        m3 = sampler.mark()
        # warm, same-length, uninterrupted NON-stream prefill (control d)
        tpf = time.time()
        pf_s = long_prefill_once(words=250)
        log(f"timing warm long-prefill submit epoch={tpf:.3f} wait={pf_s:.3f}s")
        m4 = sampler.mark()

        log(f"TIMING SSE releases={rel_sse}")
        log(f"TIMING NONSTREAM releases={rel_ns}")
        phase_report("SSE", m0, m1)
        phase_report("NONSTREAM", m1, m2)
        wc = phase_report("DECODE-control", m2, m3)
        log(f"TIMING decode control: {lines} SSE lines, /health worst={wc:.3f}s")
        wp = phase_report("WARM-PREFILL-control", m3, m4)
        log(f"TIMING warm same-length prefill: wait={pf_s:.3f}s /health worst={wp:.3f}s")
        log(
            "TIMING read: a warm prefill still >=2s => real kernel time; "
            "<0.5s => first-batch/JIT warmup only."
        )
        return True
    finally:
        sampler.stop()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--timing",
        action="store_true",
        help="slow-tick data run: SSE+nonstream disconnects plus an "
        "uninterrupted control, independent /health sampler",
    )
    ap.add_argument("--sse-reps", type=int, default=10)
    ap.add_argument("--nonstream-reps", type=int, default=3)
    ap.add_argument(
        "--health-log",
        default=os.path.expanduser("~/health_timing.log"),
        help="kept path for '<epoch> <latency>' /health samples",
    )
    args = ap.parse_args()

    s = stats()
    log(
        f"start: running={s['running']} waiting={s['waiting']} slots={s['slots_total']} "
        f"decode_graph={s['decode_graph']} blocks={s['blocks_used']}/{s['blocks_total']}"
    )
    if args.timing:
        return 0 if timing_run(args.sse_reps, args.nonstream_reps, args.health_log) else 1

    a = short_sanity("before")
    b = disconnect_check(reps=args.sse_reps, nonstream_reps=args.nonstream_reps)
    c = overload()
    d = short_sanity("after")
    overall = a and b and c and d
    log(
        f"OVERALL {'PASS' if overall else 'FAIL'} "
        f"(sanity_before={a} disconnect_cancel={b} overload={c} sanity_after={d})"
    )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
