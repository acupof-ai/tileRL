"""N conversations interleaved: the workload a snapshot tier is supposed to win.

A single conversation re-reads only its newest prefix, so the LRU entry a demotion picks
is never asked for again — measured on the V100, 43 demotions and 0 promotions, with the
wall clock 1.51x worse than no tier at all. That makes the tier pure overhead there.

Interleaving A1 B1 A2 B2 ... changes it: while B is being served, A's entries age to the
LRU end and get demoted, and A's next turn asks for one back. If the tier does not win
here it does not win anywhere, so this is the arm that decides whether it ships.

`--sessions` exists because the tier's condition is `sessions > snapshot budget` and the
published verdict swept only the budget, holding sessions at 2. A one-axis sweep finds a
threshold and a threshold reads like a law; the other axis reversed it, 0/63 -> 24/0 hits.
Default stays 2 so the published arm reproduces.

`--sys-tokens` adds the shape a real Claude Code session has: one large system prefix --
tool defs plus instructions -- resent identically on every turn of every session. That
prefix is the same tokens for all of them, so it is the store's best case and the arm the
serve path actually cares about; the fillers below still diverge after it, so a session's
own longer entry stays its own. 0 reproduces the published arm.

  python scripts/bench_chat_interleaved.py --turns 4 --grow 40                # published arm
  python scripts/bench_chat_interleaved.py --turns 4 --grow 40 --sessions 12  # the other axis
  python scripts/bench_chat_interleaved.py --turns 4 --grow 40 --sessions 12 \
      --sys-tokens 30000 --ttft --server-log /work/serve.log                  # agent shape
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchrec  # noqa: E402

_TOPICS = (
    "Explain in detail how a paged key-value cache serves a transformer decode step, "
    "including how block tables map logical positions to physical pages. ",
    "Describe how a gated delta network keeps recurrent state across a chunked prefill, "
    "and what the conv window carries between chunks. ",
)


def _fillers(n: int) -> list[str]:
    """`n` fillers whose FIRST tokens differ, so each conversation gets its own prefix hash.

    Prefixes are hashed from token 0, so beyond the two distinct topics below, `n` copies of
    one topic would collide into a single entry and the hit rate would measure the fixture
    rather than the tier. Past that point each filler is index-prefixed.

    `n <= 2` returns the topics UNCHANGED, byte for byte, so the published two-session arm
    reproduces -- an index prefix there would move every prompt length and every prefix hash,
    and the arm on record could not be re-run. The asymmetry costs little across arms: the
    prefix is 16 characters and appears once, while the filler after it is repeated
    `grow * (turn + 1)` times.
    """
    if n <= len(_TOPICS):
        return list(_TOPICS[:n])
    return [f"Session {i} of {n}. {_TOPICS[i % len(_TOPICS)]}" for i in range(n)]


def _system(target_tokens: int) -> str:
    """An agent session's shared head: tool definitions and standing instructions.

    Shape, not lorem -- what a Claude Code turn actually resends unchanged: the same tool
    schemas and rules on every turn of every session.

    1.556 tokens per word is MEASURED on this unit with the 27B's own tokenizer (72 words,
    112 tokens), not the 1.3 the filler-length scripts assume: identifiers and punctuation
    split far harder than prose, and 1.3 here would overshoot the request by 16%. Whole units
    are repeated rather than words sliced, so the count is 112 per rep and lands at -0.84% of
    30000. The caller still checks the ACHIEVED count from `usage`, since a fixture that asked
    for 30k and got 8k is not the workload it claims to be.
    """
    unit = (
        "Tool: read_file(path: string, offset: integer, limit: integer) -- returns the file "
        "with line numbers. Tool: edit_file(path: string, old: string, new: string) -- exact "
        "string replacement, fails when `old` is not unique. Tool: run(command: string, "
        "timeout_ms: integer) -- runs in the session shell, working directory persists. "
        "Rule: read a file before editing it. Rule: prefer the dedicated tool over a shell "
        "equivalent. Rule: report what the command printed, never what it should have. "
    )
    return unit * max(1, round(target_tokens / 112))


def _label(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"S{i}"


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post_stream(url: str, body: dict, timeout: float) -> tuple[dict, float]:
    """(response-shaped dict, seconds to the first token).

    Non-streaming cannot yield TTFT at all: the route awaits the whole completion before it
    builds a reply (server.py:298) and the JSON carries no per-phase timing, so wall clock is
    the only number available there. A tier hit lands in PREFILL, and at a 30k prefix the
    decode dominates, so wall clock dilutes the effect this bench exists to measure. Hence
    the same script streams when asked, rather than a second script existing.
    """
    # include_usage or a streamed reply carries no prompt_tokens (server.py:492) and the
    # achieved system-prefix check below would compare against 0 and pass vacuously.
    req = urllib.request.Request(
        url, data=json.dumps({**body, "stream": True,
                              "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    text, ttft, usage, t0 = [], None, {}, time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                # Without this the frame parses as a turn with no content and the row records
                # a fast zero-token reply -- a failure that reads as the best wall clock here.
                raise SystemExit(f"server error mid-stream: {ev['error']}")
            usage = ev.get("usage") or usage
            delta = (ev.get("choices") or [{}])[0].get("delta", {})
            # Reasoning counts: this checkpoint opens <think> in the prompt, so the first
            # token out is reasoning_content, and timing only `content` would time the
            # thinking block instead of prefill. First token of EITHER kind is the TTFT.
            piece = delta.get("content")
            if piece or delta.get("reasoning_content"):
                ttft = ttft if ttft is not None else time.perf_counter() - t0
            if piece:
                text.append(piece)
    return ({"choices": [{"message": {"content": "".join(text)}}], "usage": usage},
            -1.0 if ttft is None else ttft)


def _compiles(path: str) -> int:
    """`begins to compile` lines in the server's own log, or -1 when it was not given.

    An EMPTY file returns -1, not 0. A `python3` (no `-u`) server redirected to a file
    block-buffers stdout and the arm's `kill $SRV` is a SIGTERM, so nothing is ever
    flushed: measured on the pod, a process that had already printed the marker left
    0 bytes after 3 s and 0 after SIGTERM, while the same process under `python3 -u`
    left 38 bytes. Every cell of the 2026-09-08 DRAM grid reported `compiles: clean`
    against a 0-byte log -- a green verdict that could not have gone red.
    """
    if not path:
        return -1
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            n = sum("begins to compile" in line for line in f)
    except OSError:
        return -1
    return n if n else (-1 if os.path.getsize(path) == 0 else 0)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10.0) as r:
        return json.loads(r.read())["stats"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--sessions", type=int, default=2,
                    help="interleaved conversations; 2 reproduces the published arm")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--grow", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--sys-tokens", type=int, default=0,
                    help="shared system prefix per session, an agent's shape; 0 reproduces "
                         "the published arm")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--ttft", action="store_true",
                    help="stream, and time the first token: a prefix hit shows up in prefill, "
                         "which wall clock dilutes at a 30k prefix")
    ap.add_argument("--server-log", default="",
                    help="the server's own stdout, for the compiles count; without it a "
                         "compile inside a measured turn is invisible and reads as tier cost")
    benchrec.add_record_args(ap, client_side=True)
    args = ap.parse_args()
    if args.sessions < 1:
        ap.error("--sessions must be >= 1")

    fillers = _fillers(args.sessions)
    # Not an assert: `-O` strips those, and this guard's whole value is firing in someone
    # else's later edit. Retargeted for `--sys-tokens`: the shared system head is now WANTED
    # (it is the entry every session hits), so what must still differ is the body after it --
    # two fillers agreeing at their own token 0 make one entry serve both bodies, and the hit
    # rate goes back to measuring the fixture.
    heads = {s[:20] for s in fillers}
    if len(heads) != args.sessions:
        raise SystemExit(
            f"{args.sessions} sessions produced {len(heads)} distinct filler heads: past the "
            "shared system prefix the conversations would still share entries, and the hit "
            "rate would measure the fixture, not the tier"
        )
    # One system message, one object per conversation: the chat route renders every message's
    # role as its own ChatML turn (prompt.py:38-39), so a `system` role really does reach the
    # token stream, first and identically for all sessions -- shared tokens, not a client-side
    # fiction. Verified: `<|im_start|>system` in the render, 29749 of the 30848 prompt tokens.
    system = _system(args.sys_tokens) if args.sys_tokens else ""
    convs: list[list[dict]] = [[{"role": "system", "content": system}] if system else []
                               for _ in fillers]
    rows = []
    # Checked against the server's own `usage`, not against `_system`'s arithmetic: the second
    # would agree with the first by construction and could not catch a rendering that dropped
    # the system turn.
    sys_seen = 0
    n_skipped = 0
    for turn in range(args.turns):
        for c, filler in enumerate(fillers):
            convs[c].append({"role": "user", "content": filler * args.grow * (turn + 1)})
            before = _get(f"{args.url}/health")
            c0 = _compiles(args.server_log)
            t0 = time.perf_counter()
            body = {"model": "qwen38-27b", "messages": convs[c],
                    "max_tokens": args.max_tokens, "temperature": 0.0}
            url = f"{args.url}/v1/chat/completions"
            out, ttft = (_post_stream(url, body, args.timeout) if args.ttft
                         else (_post(url, body, args.timeout), -1.0))
            wall = time.perf_counter() - t0
            after = _get(f"{args.url}/health")
            compiles = max(0, _compiles(args.server_log) - c0) if c0 >= 0 else -1
            convs[c].append(
                {"role": "assistant", "content": out["choices"][0]["message"]["content"]}
            )
            d = {
                k: after.get(k, 0) - before.get(k, 0)
                # a publisher retiring its own entry counts as superseded, not eviction,
                # so an eviction delta alone cannot say whether pressure eased or moved.
                for k in ("prefix_hits", "prefix_hit_tokens", "prefix_published",
                          "prefix_evictions", "prefix_superseded",
                          "dram_demotions", "dram_promotions")
            }
            n = out.get("usage", {}).get("prompt_tokens", 0)
            # Peak, not delta: a pool-bound cell is the TIER's case rather than a confound to
            # engineer away, so it has to be readable in the row instead of inferred from a
            # wall clock. The tier-off arm is the one that can exhaust it.
            pool = {"pool_used_blocks": after.get("pool_used_blocks", 0),
                    "blocks_total": after.get("blocks_total", 0)}
            # Resident entry count per row, not just at the end: the V100 read turn-0 hits on
            # every other conversation, and a final count cannot say whether the store was
            # holding the shared head at the moment a given session looked it up.
            resident = {"prefix_entries": after.get("prefix_entries", 0),
                        "prefix_entries_capacity": after.get("prefix_entries_capacity", 0)}
            if args.sys_tokens and not sys_seen:
                sys_seen = n
                if n < args.sys_tokens * 0.9:
                    raise SystemExit(
                        f"--sys-tokens {args.sys_tokens} produced a {n}-token first prompt: "
                        "the shared head did not reach the token stream (a dropped system "
                        "turn looks exactly like this) and this is not the agent workload"
                    )
                print(f"shared system prefix: asked {args.sys_tokens}, first prompt {n} tokens "
                      f"({n / args.sys_tokens:.2f}x)", flush=True)
            rows.append({"turn": turn, "conv": _label(c), "prompt_tokens": n,
                         "wall_s": round(wall, 2), "ttft_s": round(ttft, 2),
                         "compiles": compiles, **pool, **resident, **d})
            # compiles is this turn's own delta; -1 means unknown (no --server-log).
            # A turn that compiled, or whose compile status is unknown, is not a record:
            # warm.compiles=0 is an assertion, and absent is unmeasured, not 0.
            if compiles == 0:
                rec = {
                    "metric": "chat_turn_wall_s", "value": round(wall, 3), "unit": "s",
                    "shape": {"turn": turn, "prompt_tokens": n,
                              "sessions": args.sessions, "conv": _label(c)},
                    "warm": {"state": "warm", "compiles": 0},
                    "n": 1, "spread": 0.0, **benchrec.record_common(args),
                }
                rec["floor"] = benchrec.measured_best_floor(rec, lower_is_better=True)
                print(f"  record {benchrec.append(rec)} appended", flush=True)
            else:
                n_skipped += 1
            pct = 100.0 * pool["pool_used_blocks"] / max(1, pool["blocks_total"])
            # depth, not just hits: the count says a match happened, this says how much of the
            # prompt it spared. 512 of 30826 reports a hit and re-prefills 98% (2026-09-08).
            depth = 100.0 * d["prefix_hit_tokens"] / max(1, n) if d["prefix_hits"] else 0.0
            print(
                f"turn {turn} conv {_label(c)}  prompt={n:6d}  wall={wall:8.2f}s  "
                f"ttft={ttft:7.2f}s  compiles={compiles:2d}  pool={pct:5.1f}%  "
                f"ent={resident['prefix_entries']}/{resident['prefix_entries_capacity']}  "
                f"hits={d['prefix_hits']}  depth={depth:5.1f}%  "
                f"demote={d['dram_demotions']}  "
                f"promote={d['dram_promotions']}  evict={d['prefix_evictions']}  "
                f"super={d['prefix_superseded']}",
                flush=True,
            )

    st = _get(f"{args.url}/health")
    total = round(sum(r["wall_s"] for r in rows), 2)
    # Per session, not a mean: a mean hides both a fixture collision (one session takes
    # every hit) and the case where the tier pays off for one conversation only.
    per_session = {
        _label(c): {
            "prefix_hits": sum(r["prefix_hits"] for r in rows if r["conv"] == _label(c)),
            "dram_promotions": sum(r["dram_promotions"] for r in rows if r["conv"] == _label(c)),
            "dram_demotions": sum(r["dram_demotions"] for r in rows if r["conv"] == _label(c)),
        }
        for c in range(args.sessions)
    }
    for label, v in per_session.items():
        print(f"session {label}: hits={v['prefix_hits']} promote={v['dram_promotions']} "
              f"demote={v['dram_demotions']}", flush=True)
    # This script attaches to a server it did not start, so a compile is only visible when the
    # operator points --server-log at that server's stdout; unknown is reported as unknown
    # rather than as clean, since a JIT inside a measured turn is charged to the tier. An empty
    # log reads unknown too -- see `_compiles`: a server without `-u` flushes nothing, so
    # "clean" would be a verdict with no negative branch.
    dirty = [(r["turn"], r["conv"], r["compiles"]) for r in rows if r["compiles"] > 0]
    known = all(r["compiles"] >= 0 for r in rows)
    verdict = "unknown (no --server-log, or it is empty -- run serve under python3 -u)" \
        if not known else dirty or "clean"
    print(f"compiles: {verdict}", flush=True)
    if n_skipped:
        print(f"records skipped: {n_skipped} turns compiled or had unknown compile status "
              f"(pass --server-log under python3 -u to make them records)", flush=True)
    peak = max((r["pool_used_blocks"] for r in rows), default=0)
    tot = max((r["blocks_total"] for r in rows), default=0)
    print(f"pool peak: {peak}/{tot} blocks ({100.0 * peak / max(1, tot):.1f}%)", flush=True)
    # Hit and miss TTFT as two measured buckets, plus the depth that explains them. Without
    # these the hit rate and the wall clock are the only two numbers, and a rate that rises
    # while the clock doubles has to be solved for depth instead of read (2026-09-08, #271).
    hit_rows = [r for r in rows if r["prefix_hits"]]
    miss_rows = [r for r in rows if not r["prefix_hits"]]
    depth = {"hit_turns": len(hit_rows), "miss_turns": len(miss_rows),
             "mean_hit_depth_pct": round(
                 100.0 * sum(r["prefix_hit_tokens"] for r in hit_rows)
                 / max(1, sum(r["prompt_tokens"] for r in hit_rows)), 1),
             "mean_hit_ttft_s": round(
                 sum(r["ttft_s"] for r in hit_rows) / max(1, len(hit_rows)), 2),
             "mean_miss_ttft_s": round(
                 sum(r["ttft_s"] for r in miss_rows) / max(1, len(miss_rows)), 2)}
    print(f"hit depth: {depth['mean_hit_depth_pct']}% of prompt over {depth['hit_turns']} hit "
          f"turns; ttft hit {depth['mean_hit_ttft_s']}s vs miss {depth['mean_miss_ttft_s']}s "
          f"over {depth['miss_turns']} miss turns", flush=True)
    print(json.dumps({"sessions": args.sessions, "turns": args.turns, "rows": rows,
                      "per_session": per_session, "total_wall_s": total,
                      "turns_with_compiles": dirty, "compiles_known": known,
                      "pool_peak_blocks": peak, "blocks_total": tot,
                      "hit_depth": depth,
                      "final_stats": st}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
