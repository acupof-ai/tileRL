"""V100 #796 repro/regression harness: does a prompt-end closed sparse prefix get
adopted by a short-advance follower?

#796 (device-only, V100 W2048 / 4 slot / sparse-k128 / 1 GiB RAM / 8 GiB SSD):
natural-leave `publish_dropped` grew index entries but never landed shared blobs,
so a same-head follower took a full-prefill miss (hits +0, warm_adoptions 0,
kv_cold_shared_pages 0) despite publishing entries. The accepted (b) fix adds a
prompt-end remedial closure: once a publisher REQUEST ENDS, its private cold
pages are rehomed under their content keys, so a follower that enters a SHORT
decode advance afterwards adopts immediately -- no long decode is needed.

This harness is the device gate fixkv's PR must pass. It drives a running 27B
serve and asserts, in ONE run:
  * a publisher of the N-word head closes at request end: shared pages/bytes
    leave 0 and sparse_prefix_entries >= 1 (sparse_prefix_* is the real sparse
    index, #798 -- dense prefix_* is NoPrefixStore noise under a sparse build);
  * a follower with the SAME head plus a short suffix then adopts:
    sparse_prefix_hits >= 1, sparse_prefix_warm_adoptions >= 1, and the serve
    reports a non-zero adopted prefix (sparse_matched in the log);
  * POSITIVE CONTROL for the five close keys: request-end ticks show the
    remedial transfer (pub_cold_transfer / pub_share_hold / pub_frame_d2h /
    pub_bounds_d2h / pub_draft_clone) and/or ssd_mmap -- the old symptom was all
    five absent and ssd_mmap 0 on every close tick;
  * tokens equal a temperature-0 single-request oracle;
  * close/request-end ticks remain zero PUBLISH on a cancelled/failed row.

The serve identity (sha) and the live pgrep line are printed on the same row as
the result so a number is attributable to the process that produced it.

Dry-run (CI, no GPU, no network): resolves the geometry, assembles the
collectors/parsers, and prints the resolved parameters and the acceptance gates.
    TILERL_TARGET=cpu python3 scripts/repro_adopt_796.py --dry-run
Device:
    python3 scripts/repro_adopt_796.py --url http://127.0.0.1:8000 \\
        --words 16000 --suffix-words 64 --follower-tokens 48 --out /work/d796
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field

#: one sparse page; the same constant the engine's BLOCK_TOKENS uses.
BLOCK_TOKENS = 16
#: the five step-timing keys that prove a request-end remedial closure MOVED
#: bytes. #796's signature was all five 0 on every close tick.
CLOSE_TRANSFER_KEYS = (
    "pub_cold_transfer", "pub_share_hold", "pub_frame_d2h",
    "pub_bounds_d2h", "pub_draft_clone",
)
#: health counters that must move for a successful publish+adopt.
SPARSE_KEYS = (
    "sparse_prefix_published", "sparse_prefix_hits", "sparse_prefix_evictions",
    "sparse_prefix_entries", "sparse_prefix_warm_adoptions",
    "kv_cold_shared_bytes", "kv_cold_shared_pages", "kv_cold_shared_ssd_bytes",
)
_TICK = re.compile(r"\[step-timing\] tick (\d+) total=(-?\d+)ms(?: dec=(\d+) pre=(\d+))? (.*)")
_KVMS = re.compile(r"(\w+)=(-?\d+)(?:ms)?")


@dataclass
class Geometry:
    words: int
    tokens_per_word: float
    suffix_words: int
    follower_tokens: int

    @property
    def head_tokens(self) -> int:
        return int(round(self.words * self.tokens_per_word))

    @property
    def head_pages(self) -> int:
        return self.head_tokens // BLOCK_TOKENS

    @property
    def suffix_tokens(self) -> int:
        return int(round(self.suffix_words * self.tokens_per_word))

    def describe(self) -> dict:
        return {
            "words": self.words,
            "tokens_per_word": self.tokens_per_word,
            "head_tokens_est": self.head_tokens,
            "head_pages_est": self.head_pages,
            "suffix_words": self.suffix_words,
            "suffix_tokens_est": self.suffix_tokens,
            "follower_gen_tokens": self.follower_tokens,
        }


def parse_close_tick(line: str) -> dict | None:
    """One [step-timing] line -> {key: ms} for the byte-moving segments, plus
    ssd_mmap. None when the line is not a step-timing tick. The parser accepts
    both the old ``key=Nms`` and the bare ``key=N`` spellings."""
    m = _TICK.search(line)
    if not m:
        return None
    segs = {k: int(v) for k, v in _KVMS.findall(m.group(5))}
    return {"n": int(m.group(1)),
            "dec": int(m.group(3)) if m.group(3) is not None else 0,
            "pre": int(m.group(4)) if m.group(4) is not None else 0,
            **segs}


def close_transfer_present(parsed_ticks: list[dict]) -> bool:
    """Positive control: at least one tick paid one of the five transfer
    segments or an SSD mmap. Their total absence on EVERY tick is #796's
    silent-zero signature."""
    return any(t.get(k, 0) > 0 for t in parsed_ticks if t for k in
               (*CLOSE_TRANSFER_KEYS, "ssd_mmap"))


def close_window_moved_zero(parsed_ticks: list[dict]) -> bool:
    """#785 red line for a CANCELLED/FAILED row: its request-end ticks move NO
    publish bytes -- every five-key segment and ssd_mmap are 0. Returns True
    only when every supplied tick is clean (an empty window is not accepted:
    no ticks means the cancellation was never observed, not that it was clean)."""
    ticks = [t for t in parsed_ticks if t]
    if not ticks:
        return False
    return all(t.get(k, 0) == 0 for t in ticks for k in
               (*CLOSE_TRANSFER_KEYS, "ssd_mmap"))


def cancel_during_stream(base_url: str, head: str, settle_s: float = 3.0):
    """Open a STREAMING chat on the long head and sever the socket while it is
    generating, which is the reader-disconnect / abort geometry the serve maps
    to engine.cancel. Returns once the connection is torn down; the caller then
    gives the serve a short settle window and reads the cancel tick from the log.
    Non-streaming chat cannot be interrupted (it returns only on completion), so
    the stream is what exercises the cancel release path."""
    import http.client
    hostport = base_url.replace("http://", "").replace("https://", "")
    host, _, port = hostport.partition(":")
    conn = http.client.HTTPConnection(host, int(port or 8000), timeout=30)
    body = json.dumps({
        "model": "qwen38-27b",
        "messages": [{"role": "user", "content": head}],
        "max_tokens": 4096, "temperature": 0.0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    # Drain enough to prove the request is admitted and generating, then sever.
    resp.read(64)
    time.sleep(settle_s)
    with contextlib.suppress(OSError):
        conn.sock.shutdown(socket.SHUT_RDWR)
    conn.close()


def cold_fill_state(stats: dict) -> dict:
    """The four cold-tier keys (GiB) + residency, the state a tok/s number must
    be read with; identical shape to the H20/A6 collector."""
    g = 2 ** 30
    return {
        "priv_gib": round(stats.get("kv_cold_bytes", 0) / g, 3),
        "priv_ssd_gib": round(stats.get("kv_cold_ssd_bytes", 0) / g, 3),
        "shared_gib": round(stats.get("kv_cold_shared_bytes", 0) / g, 3),
        "shared_ssd_gib": round(stats.get("kv_cold_shared_ssd_bytes", 0) / g, 3),
        "shared_pages": stats.get("kv_cold_shared_pages", 0),
    }


def _http(url: str, body: dict | None = None, timeout: float = 1800.0):
    if body is None:
        with urllib.request.urlopen(url, timeout=10.0) as r:
            return json.loads(r.read())
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stats_of(health: dict) -> dict:
    return health.get("stats") or health  # tolerate flat or nested


def build_head(words: int) -> str:
    # A deterministic, token-rich head; the exact tokenizer ratio is reported,
    # not assumed (16000 words measured ~32028 tokens = ~2.00 tok/word on qwen38).
    return ("The sparse prefix ledger records each completed page once and "
            "reuses it under a content key. ") * ((words // 14) + 1)


def serve_identity() -> dict:
    """sha and the live serve pgrep on one record, so a result names the binary."""
    out = {"sha": os.environ.get("TILERL_SERVE_SHA", ""), "pgrep": ""}
    try:
        p = subprocess.run(["pgrep", "-af", "tilerl.cli serve"], capture_output=True,
                           text=True, timeout=5)
        out["pgrep"] = " | ".join(p.stdout.strip().splitlines()[:2])
    except Exception as exc:  # noqa: BLE001 - identity is best-effort metadata
        out["pgrep"] = f"<pgrep unavailable: {type(exc).__name__}>"
    return out


@dataclass
class Acceptance:
    close_transfer_seen: bool = False
    shared_pages_after_publisher: int = 0
    sparse_entries_after_publisher: int = 0
    follower_hits_delta: int = 0
    follower_adoptions_delta: int = 0
    follower_sparse_matched: int = 0
    tokens_equal_oracle: bool | None = None
    close_zero_on_cancel: bool | None = None
    detail: dict = field(default_factory=dict)

    def ok(self) -> bool:
        return (self.close_transfer_seen
                and self.shared_pages_after_publisher >= 1
                and self.sparse_entries_after_publisher >= 1
                and self.follower_hits_delta >= 1
                and self.follower_adoptions_delta >= 1
                and self.follower_sparse_matched > 0
                and self.tokens_equal_oracle is True
                and self.close_zero_on_cancel is True)


GATES_DOC = [
    "close tick: >=1 of {pub_cold_transfer,pub_share_hold,pub_frame_d2h,"
    "pub_bounds_d2h,pub_draft_clone,ssd_mmap} > 0 (positive control)",
    "after publisher request-end: kv_cold_shared_pages >= 1 and "
    "sparse_prefix_entries >= 1",
    "short-advance follower: sparse_prefix_hits >= 1 and "
    "sparse_prefix_warm_adoptions >= 1, log sparse_matched > 0",
    "follower tokens == temperature-0 single-request oracle",
    "cancel/failed row: request-end ticks move no publish bytes",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--words", type=int, default=16000, help="publisher head words")
    ap.add_argument("--tokens-per-word", type=float, default=2.002)
    ap.add_argument("--suffix-words", type=int, default=64,
                    help="short follower-only suffix (short advance, not long decode)")
    ap.add_argument("--follower-tokens", type=int, default=48)
    ap.add_argument("--out", default="/work/d796")
    ap.add_argument("--serve-log", default=os.environ.get("SERVE_LOG", ""))
    ap.add_argument("--poll-s", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve geometry, assemble collectors, print gates; no network")
    args = ap.parse_args()

    geo = Geometry(args.words, args.tokens_per_word,
                   args.suffix_words, args.follower_tokens)

    if args.dry_run:
        plan = {
            "mode": "dry-run",
            "geometry": geo.describe(),
            "endpoints": {"health": f"{args.url}/health",
                          "chat": f"{args.url}/v1/chat/completions"},
            "collected_sparse_keys": list(SPARSE_KEYS),
            "close_transfer_keys": list(CLOSE_TRANSFER_KEYS),
            "acceptance_gates": GATES_DOC,
            "out_dir": args.out,
            "serve_identity": serve_identity(),
        }
        print(json.dumps(plan, indent=2))
        # Assemble the parser on a synthetic tick to prove the collector wiring,
        # including the #796 all-zero close tick the positive control rejects.
        zero = "[step-timing] tick 7 total=90ms dec=1 pre=0 model=80ms sample=4ms path=eager sparse=1 pub_cold_transfer=0ms pub_share_hold=0ms pub_frame_d2h=0ms pub_bounds_d2h=0ms pub_draft_clone=0ms ssd_mmap=0ms"
        moved = zero.replace("pub_cold_transfer=0ms", "pub_cold_transfer=12ms")
        t0, t1 = parse_close_tick(zero), parse_close_tick(moved)
        assert t0 and t1 and t0["pub_cold_transfer"] == 0
        assert close_transfer_present([t1]) and not close_transfer_present([t0])
        print("dry-run OK: geometry resolved; parser + close positive-control assembled")
        return 0

    return run_device(args, geo)


def run_device(args, geo) -> int:  # pragma: no cover - device path
    os.makedirs(args.out, exist_ok=True)
    head = build_head(args.words)[: geo.head_tokens * 4]  # char budget, tokenizer trims
    suffix = ("Now continue with a distinct short continuation. "
              * ((args.suffix_words // 9) + 1))[: geo.suffix_tokens * 4]
    acc = Acceptance()

    # Background /health poller: the publisher's closure lands at request END, so
    # sampling only the two endpoints can miss the transition. One CSV row per
    # poll gives the shared-byte fill curve and proves it leaves 0 only after end.
    poll_rows: list[dict] = []
    import threading
    polling = {"on": True}

    def poll_loop():
        while polling["on"]:
            try:
                s = health_stats()
                row = {"t": round(time.time() - t_start, 3),
                       **{k: s.get(k) for k in SPARSE_KEYS}}
                poll_rows.append(row)
            except Exception:  # noqa: BLE001 - a missed poll must not kill the run
                pass
            time.sleep(args.poll_s)

    t_start = time.time()
    worker = threading.Thread(target=poll_loop, daemon=True)
    worker.start()

    def health_stats():
        return stats_of(_http(f"{args.url}/health"))

    def chat(content: str, max_tokens: int):
        return _http(f"{args.url}/v1/chat/completions", {
            "model": "qwen38-27b",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.0,
            # sparse follower must not silently take thinking defaults
            "chat_template_kwargs": {"enable_thinking": False},
        })

    # ---- publisher: head only; request END triggers the remedial closure (b).
    pre = health_stats()
    t0 = time.perf_counter()
    chat(head, 8)
    pub_wall = time.perf_counter() - t0
    post = health_stats()
    acc.shared_pages_after_publisher = post.get("kv_cold_shared_pages", 0)
    acc.sparse_entries_after_publisher = post.get("sparse_prefix_entries", 0)

    # ---- follower: same head + short suffix, short advance, adopts immediately.
    f0 = health_stats()
    tout = time.perf_counter()
    fout = chat(head + suffix, args.follower_tokens)
    fol_wall = time.perf_counter() - tout
    f1 = health_stats()
    acc.follower_hits_delta = f1.get("sparse_prefix_hits", 0) - f0.get("sparse_prefix_hits", 0)
    acc.follower_adoptions_delta = (
        f1.get("sparse_prefix_warm_adoptions", 0)
        - f0.get("sparse_prefix_warm_adoptions", 0))

    # ---- temperature-0 single-request oracle (full-prefill reference)
    oracle = chat(head + suffix, args.follower_tokens)
    ftxt = (fout.get("choices") or [{}])[0].get("message", {}).get("content", "")
    otxt = (oracle.get("choices") or [{}])[0].get("message", {}).get("content", "")
    acc.tokens_equal_oracle = ftxt.strip() == otxt.strip()

    # ---- CANCEL red line (#785): a reader-aborted row moves no publish bytes.
    # Record the log offset, sever a streaming head mid-generation, then read
    # only the ticks the serve wrote during the cancel window and demand every
    # close key + ssd_mmap be 0. A normal request-end closure elsewhere is
    # allowed to move bytes; THIS window must not.
    cancel_ticks: list[dict] = []
    log_off = 0
    if args.serve_log:
        log_off = os.path.getsize(args.serve_log)
    try:
        cancel_during_stream(args.url, head)
    except Exception as exc:  # noqa: BLE001 - reported as a failed gate, not a crash
        acc.detail["cancel_error"] = f"{type(exc).__name__}: {exc}"
    # give the serve the disconnect-detection + cancel release tick
    deadline = time.time() + 15
    if args.serve_log:
        while time.time() < deadline:
            time.sleep(0.5)
            with open(args.serve_log, "rb") as fh:
                fh.seek(log_off)
                window = fh.read().decode(errors="replace")
            cancel_ticks = [p for p in
                           (parse_close_tick(line) for line in window.splitlines()) if p]
            if cancel_ticks and "cancel" in window.lower():
                break
    acc.close_zero_on_cancel = close_window_moved_zero(cancel_ticks)
    acc.detail["cancel_tick_count"] = len(cancel_ticks)

    # ---- log evidence: close-tick positive control + sparse_matched
    polling["on"] = False
    worker.join(timeout=2)
    if args.serve_log and os.path.exists(args.serve_log):
        with open(args.serve_log, errors="replace") as fh:
            log_text = fh.read()
        ticks = [p for p in (parse_close_tick(line) for line in log_text.splitlines()) if p]
        acc.close_transfer_seen = close_transfer_present(ticks)
        m = re.findall(r"sparse_matched[=: ]+(\d+)", log_text)
        acc.follower_sparse_matched = max((int(x) for x in m), default=0)
    else:
        acc.close_transfer_seen = post.get("kv_cold_shared_pages", 0) >= 1

    acc.detail = {
        "publisher_wall_s": round(pub_wall, 2),
        "follower_wall_s": round(fol_wall, 2),
        "cold_before": cold_fill_state(pre),
        "cold_after_publisher": cold_fill_state(post),
        "cold_after_follower": cold_fill_state(f1),
        "sparse_before": {k: pre.get(k) for k in SPARSE_KEYS},
        "sparse_after": {k: f1.get(k) for k in SPARSE_KEYS},
        "follower_text_head": ftxt[:80],
    }
    identity = serve_identity()
    record = {
        "metric": "d796_adoption", "verdict": "PASS" if acc.ok() else "FAIL",
        "geometry": geo.describe(), "serve": identity,
        "gates": {
            "close_transfer_seen": acc.close_transfer_seen,
            "shared_pages_after_publisher": acc.shared_pages_after_publisher,
            "sparse_entries_after_publisher": acc.sparse_entries_after_publisher,
            "follower_hits_delta": acc.follower_hits_delta,
            "follower_adoptions_delta": acc.follower_adoptions_delta,
            "follower_sparse_matched": acc.follower_sparse_matched,
            "tokens_equal_oracle": acc.tokens_equal_oracle,
            "close_zero_on_cancel": acc.close_zero_on_cancel,
        },
        **acc.detail,
    }
    out_json = os.path.join(args.out, "d796_adopt.json")
    with open(out_json, "w") as fh:
        json.dump(record, fh, indent=2)
    if poll_rows:
        import csv
        # sha/pgrep ride on every poll row, not just the result JSON, so a CSV
        # row read on its own still names the binary that produced it.
        for row in poll_rows:
            row["serve_sha"] = identity["sha"]
            row["serve_pgrep"] = identity["pgrep"]
        csv_path = os.path.join(args.out, "d796_health_poll.csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(poll_rows[0]))
            w.writeheader()
            w.writerows(poll_rows)
    print(json.dumps(record["gates"], indent=2))
    print(f"verdict={record['verdict']} artifact={out_json}")
    return 0 if acc.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
