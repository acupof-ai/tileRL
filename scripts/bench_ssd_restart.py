"""Does a restart fault the prefix off disk, or is the second run just warmer?

The SSD tier's whole claim is that after a restart HBM is empty so every returning
conversation's first turn reaches back to disk. Measuring it as "start, serve, restart,
serve" does not test that claim: the tilelang JIT cache is shared across starts
(`TILELANG_CACHE_DIR=/work/tilelang_cache`), the page cache holds the weights, and both
make the SECOND start faster whatever the tier does.

So four server starts. The first is a throwaway whose only job is to fill the shared JIT
cache at this prompt's shape buckets, because a compile inside a timed window is worth
more than the tier is: measured, the cold arm paid 6 compiles and the arms after it paid
0, which alone made an EMPTY-tier control 3.956x faster than cold. Then three arms:

    cold     empty spill dir            -> the number to beat
    faulted  the dir cold just filled   -> the tier's number
    control  a DIFFERENT empty dir      -> must land back at `cold`

`control` must land within noise of `cold` (the check is 0.85-1.15x): it is the same arm
order with an EMPTY tier, so anything it gains is start order rather than the tier.

Two numbers come out, for two scenarios, both real:

  restart          the faulted arm's wall clock. A process restart empties HBM and leaves
                   the HOST page cache alone, so the fault-in reads from memory. This is
                   the common case and the one asked about.
  reboot/evicted   `composed_tier_s` -- the entry's measured standalone disk read plus the
                   prefill of the tokens the hit did not cover.

Getting there took the bytes/bandwidth division, which is what caught three runs that all
read like disk numbers and were not: 320.6 MiB at a measured 182.6 MiB/s is 1.756 s
against a 1.690 s arm, so the arm never touched the device. `_evict_cache` fsyncs and
fadvises each spill file between the arms and moves the probe from 4477.8 MiB/s to ~509,
no further -- DONTNEED only drops pages nothing else references. So the disk number is
composed from a standalone read rather than chased with more eviction.

  scripts/pod_run.sh ssdrestart 6 -- /work/tl013/bin/python -u \
      scripts/bench_ssd_restart.py --tokens 3000

The card comes from pod_run.sh, not from a flag here: this script had a --card
that only set CUDA_VISIBLE_DEVICES, which OVERRODE the launcher's pin and sent a
run asked for card 6 onto card 0, where another team held 28 GB.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

#: Turn 2's extra text. Short on purpose: it must extend the prompt past turn 1 (so the
#: stored entry is a strict prefix and therefore servable) without adding enough tokens to
#: move the wall clock.
_FOLLOWUP = "Now also explain what happens on a cache miss."

_FILLER = (
    "Explain in detail how a paged key-value cache serves a transformer decode step, "
    "including how block tables map logical positions to physical pages. "
)


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _stats(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10.0) as r:
        return json.loads(r.read())["stats"]


def _wait_up(port: int, proc: subprocess.Popen, deadline_s: float) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before serving")
        try:
            _stats(port)
            return
        except (urllib.error.URLError, OSError, KeyError):
            time.sleep(1.0)
    raise TimeoutError(f"server not up within {deadline_s}s")


def _prompt(target_tokens: int) -> str:
    # ~1.3 tokens per word for this filler; deliberately NOT block-aligned, since a
    # ragged length is the case the publish fix exists for.
    words = max(1, int(target_tokens / 1.3))
    text = (_FILLER * (words // len(_FILLER.split()) + 2)).split()
    return " ".join(text[:words]) + " Summarize the mechanism in one sentence."


def _serve(args, spill: str, log: str):
    cmd = [
        args.python, "-u", "-m", "tilerl.cli", "serve",
        "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port),
        "--max-batch", "1", "--max-ctx", str(args.max_ctx), "--slots", str(args.slots),
    ]
    if spill:
        cmd += ["--ssd-path", spill]
    env = dict(os.environ, TILELANG_CACHE_DIR="/work/tilelang_cache")
    # CUDA_VISIBLE_DEVICES is NOT set here: pod_run.sh already pins the card, and setting
    # it again overrode that -- the first run of this script asked for card 6 through the
    # launcher and landed on card 0, which another team was holding with 28 GB.
    with open(log, "wb") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                cwd=args.repo)


def _compiles(log: str) -> int:
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            return sum("begins to compile" in line for line in f)
    except OSError:
        return -1


def _entry_bytes(spill: str) -> int:
    """Bytes a single fault-in reads: the servable .kv (the longest) plus its .st."""
    d = os.path.join(spill, "tilerl_kvtier")
    if not os.path.isdir(d):
        return 0
    kvs = sorted((os.path.getsize(os.path.join(d, f)), f)
                 for f in os.listdir(d) if f.endswith(".kv"))
    if not kvs:
        return 0
    size, name = kvs[-1]
    st = os.path.join(d, name[:-3] + ".st")
    return size + (os.path.getsize(st) if os.path.exists(st) else 0)


def _matched_tokens(spill: str, turn2_ids: int) -> int:
    """Tokens the longest SERVABLE entry covers -- read from the entries, not inferred.

    Previously this divided the largest .kv by the smallest and multiplied by a 512-token
    unit, which was calibrated when six entries formed a size ladder. With two entries it
    reported 512 for a hit that covered 2720, and the ceiling built on it said the arm
    saved 2.8x more than the tokens it matched could explain -- an instrument artifact
    that reads exactly like a bench measuring its own page cache.

    Servable means a strict prefix of turn 2: `_match_prefix` treats a full-length match
    as a miss, and the decode entry (prompt + generated reply) is not a prefix at all.
    """
    import glob

    import torch
    d = os.path.join(spill, "tilerl_kvtier")
    best = 0
    for f in glob.glob(os.path.join(d, "*.kv")):
        try:
            n = len(torch.load(f, map_location="cpu")["tokens"])
        except Exception:  # noqa: BLE001, PERF203 - written in place, may be mid-save
            continue
        if n < turn2_ids:
            best = max(best, n)
    return best


def _prefix_check(spill: str, args, prompt: str, reply: str) -> dict:
    """Is ANY spilled entry a prefix of what turn 2 tokenizes to?

    Reads the ids the tier stored and the ids the server would build for turn 2, and
    reports the first index where each diverges. Without this a 0-hit run cannot be told
    apart from an engine bug -- which is exactly the two days this bench already cost.

    Every entry, not the largest: the largest is the DECODE publish (prompt plus the
    generated reply), which `blocks_to_text` strips from replayed history by design and
    which therefore always diverges. Reading only that one reported DIVERGES over a tier
    holding a perfectly good prompt-only entry (measured, card 1).
    """
    import glob
    d = os.path.join(spill, "tilerl_kvtier")
    kvs = sorted(glob.glob(os.path.join(d, "*.kv")), key=os.path.getsize)
    if not kvs:
        return {"prefix_check": "no spill file"}
    try:
        import sys
        sys.path.insert(0, os.path.join(args.repo, "src"))
        sys.path.insert(0, os.path.join(args.repo, "packages/tilerl-kernels/src"))
        import torch

        from tilerl.cli import _qwen38_tokenizer
        from tilerl.server import ChatMessage, _render_chat
        from tilerl.tokenizer import get_tokenizer
        # the CLI's own resolver: a bare hub id 401s, and the server used this one
        tk = _qwen38_tokenizer() if args.model == "qwen38-27b" else get_tokenizer(None)
        turn2 = tk.encode(_render_chat([
            ChatMessage(role="user", content=prompt),
            ChatMessage(role="assistant", content=reply),
            ChatMessage(role="user", content=_FOLLOWUP),
        ]))
        # Skip an entry mid-write: the tier saves .kv in place with no temp-and-rename,
        # so a file the writer thread has not finished raises inside torch.load. Skipping
        # one is right -- an unfinished entry cannot serve a lookup either -- but aborting
        # the whole probe on it turns a readable tier into "unavailable" (measured).
        entries, unreadable = [], []
        for f in kvs:
            try:
                entries.append(list(torch.load(f, map_location="cpu")["tokens"]))
            except Exception:  # noqa: BLE001, PERF203 - still being written
                unreadable.append(os.path.basename(f))
    except Exception as e:  # noqa: BLE001 - a probe; the arms still run
        return {"prefix_check": f"unavailable: {type(e).__name__}: {e}"}

    def diff(stored):
        n = min(len(stored), len(turn2))
        at = next((i for i in range(n) if stored[i] != turn2[i]), None)
        ok = at is None and len(turn2) > len(stored)  # `>`: a full-length match is a miss
        return {"stored_ids": len(stored), "prefix": ok, "first_diff": at}

    per = [diff(e) for e in entries]
    servable = [p["stored_ids"] for p in per if p["prefix"]]
    return {"prefix_check": "MATCH" if servable else "DIVERGES",
            "turn2_ids": len(turn2), "servable": sorted(servable), "entries": per,
            "unreadable": unreadable}


def _evict_cache(spill: str) -> str:
    """Drop the spill files from the host page cache, per file, with POSIX_FADV_DONTNEED.

    Without this the bench cannot see the disk at all: the cold arm WRITES the spill and
    the faulted arm reads it seconds later, so write-through puts every byte in page
    cache by construction. Measured 2026-09-05 -- the faulted arm came in at 1.168 s
    while reading its own 309.6 MiB off this device takes 1556 ms at 198.9 MiB/s, i.e.
    the arm was faster than its own disk traffic and therefore never did it.

    fadvise on the individual files, NOT `/proc/sys/vm/drop_caches`: this host is shared
    and dropping the whole cache evicts other teams' weights. (I did that once before
    reasoning about it.)
    """
    d = os.path.join(spill, "tilerl_kvtier")
    if not os.path.isdir(d):
        return "no spill dir"
    n = 0
    for name in sorted(os.listdir(d)):
        if not name.endswith((".kv", ".st")):
            continue
        # fsync FIRST. POSIX_FADV_DONTNEED silently skips a dirty page, and these files
        # were written seconds ago by the tier's flush daemon, so most of them are dirty.
        # Measured 2026-09-05: without the fsync the evict removed only ~33% of the bytes
        # (arm delta 0.564 s against a full cold read's 1.684 s) and the arm came in at
        # 1.732 s, FASTER than the 1.756 s its own bytes take off this device -- the tell
        # that the eviction was partial.
        fd = os.open(os.path.join(d, name), os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        finally:
            os.close(fd)
    # Verify rather than trust: read 4 MiB of the servable entry and time it. Cached
    # pages come back at GiB/s, disk at ~180 MiB/s, so this separates them by 20x.
    probe = _probe_mib_s(d)
    return f"evicted {n} files, probe {probe:.0f} MiB/s"


def _probe_mib_s(d: str) -> float:
    """Read bandwidth of the first 4 MiB of the largest spill file, right now."""
    kvs = sorted((os.path.getsize(os.path.join(d, f)), f)
                 for f in os.listdir(d) if f.endswith(".kv"))
    if not kvs:
        return 0.0
    t0 = time.monotonic()
    with open(os.path.join(d, kvs[-1][1]), "rb") as fh:
        n = len(fh.read(1 << 22))
    el = time.monotonic() - t0
    return (n / 2**20) / el if el else 0.0


def _arm(args, name: str, spill: str, prompt, reply: str = "") -> dict:
    """One server start, one request, the counters either side of it.

    ``prompt`` is a string for a single-turn request; ``reply`` makes it a real second
    turn -- user, assistant, user -- which is what the tier stores. The engine publishes
    `req.tokens[:materialized]` during DECODE, so the entry on disk is prompt PLUS the
    reply it generated; a turn 2 that omits the reply is not a prefix of it and cannot
    hit. That is what made every arm read 0 SSD hits (errors/2026-09-07).
    """
    log = f"/work/ssd_restart_{name}.log"
    msgs = [{"role": "user", "content": prompt}]
    if reply:
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": _FOLLOWUP}]
    proc = _serve(args, spill, log)
    try:
        _wait_up(args.port, proc, args.boot_s)
        before = _stats(args.port)
        t0 = time.monotonic()
        r = _post(f"http://127.0.0.1:{args.port}/v1/messages",
                  {"model": args.model, "max_tokens": args.gen,
                   "messages": msgs}, args.req_s)
        wall = time.monotonic() - t0
        after = _stats(args.port)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    d = lambda k: int(after.get(k, 0)) - int(before.get(k, 0))  # noqa: E731
    return {
        "arm": name,
        "wall_s": round(wall, 3),
        "compiles": _compiles(log),
        "prompt_tokens": int(r["usage"]["input_tokens"]),
        "output_tokens": int(r["usage"]["output_tokens"]),
        "ms_per_prompt_token": round(1000 * wall / max(1, r["usage"]["input_tokens"]), 3),
        # The reply, so a later arm can send a REAL turn 2 (user, assistant, user).
        # thinking blocks too: at max_tokens=8 the whole reply can land inside one and
        # a text-only read comes back empty (measured).
        "reply": "".join(b.get("text") or b.get("thinking") or ""
                         for b in r.get("content", [])),
        "reply_blocks": [b.get("type") for b in r.get("content", [])],
        "ssd_hits": d("ssd_hits"),
        "ssd_faults": d("ssd_faults"),
        "ssd_entries": int(after.get("ssd_entries", 0)),
        "ssd_recovered": int(after.get("ssd_recovered", 0)),
        "ssd_offered": d("ssd_offered"),
        "ssd_refusals": d("ssd_refusals"),
        "prefix_hits": d("prefix_hits"),
        "prefix_published": d("prefix_published"),
        # `evictions` is what replace-on-publish aims at, `superseded` proves it ran
        "prefix_evictions": d("prefix_evictions"),
        "prefix_superseded": d("prefix_superseded"),
        "prefix_entries": int(after.get("prefix_entries", 0)),
        # tick_loads refutes the off-tick claim: a fault served from a torch.load on the
        # calling thread is synchronous, whatever the wall clock says.
        "ssd_prefetches": d("ssd_prefetches"),
        "ssd_fetches_ready": d("ssd_fetches_ready"),
        "ssd_fetch_drops": d("ssd_fetch_drops"),
        "ssd_tick_loads": d("ssd_tick_loads"),
        # 0 hits with waits > 0 means a row was admitted before its fetch landed
        "ssd_fetch_waits": d("ssd_fetch_waits"),
        # B for this arm, over both planes: the reader thread fetches the .st with the .kv
        "fetch_ms": d("ssd_fetch_ms"),
        "fetch_mib_s": round(d("ssd_fetch_bytes") / 2**20 / (d("ssd_fetch_ms") / 1000), 1)
                       if d("ssd_fetch_ms") > 0 else None,
        "prefill_rate": after.get("prefill_rate"),
        "break_even_tokens": after.get("prefix_break_even_tokens"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--python", default="/work/tl013/bin/python")
    ap.add_argument("--repo", default=os.environ.get("REMOTE_DIR"),
                    required="REMOTE_DIR" not in os.environ,
                    help="server cwd, one tree per session. No fallback: a wrong tree produces a number, not an error")
    ap.add_argument("--spill", default="/work/ssd_tier_bench")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--tokens", type=int, default=3000, help="target prompt length")
    ap.add_argument("--gen", type=int, default=8, help="tokens to generate; keep small so "
                    "the wall clock is prefill. NOTE: a decode-boundary publish needs "
                    "gen >= BLOCK_TOKENS, so the default measures the fetch and says "
                    "nothing about decode-side churn -- pass --gen 256 for that")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--boot-s", type=float, default=900.0)
    ap.add_argument("--req-s", type=float, default=1800.0)
    ap.add_argument("--device-mib-s", type=float, default=182.6,
                    help="measured read bandwidth of the spill device; the verdict's "
                         "bytes/bandwidth check uses it. Default is this pod's, from "
                         "scripts/bench_ssd_bandwidth.py one_entry (182.6 MiB/s cold, "
                         "4477.8 warm -- 24x apart, which is why the check works)")
    args = ap.parse_args()

    # A decode-boundary publish needs a chain end landing on a 16-multiple, so below
    # gen 16 there are ZERO of them and the prefix_evictions/superseded columns are
    # structurally 0 -- a green row that measured nothing. Printed, not raised: the
    # default is deliberate (the wall clock stays prefill) and the fetch arms are valid.
    if args.gen < 16:
        print(f"# NOTE gen={args.gen} < 16: no decode-boundary publishes are possible, so "
              f"prefix_evictions and prefix_superseded below are 0 by construction, not by "
              f"measurement. Use --gen 256 to price decode-side churn.")

    prompt = _prompt(args.tokens)
    main_dir, ctrl_dir = args.spill, args.spill + "_control"
    for d in (main_dir, ctrl_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    # A throwaway start whose only job is to fill the shared TileLang JIT cache at the
    # target prompt's shape buckets. Measured 2026-09-05: without it the cold arm paid 6
    # compiles inside its timed window and the two arms after it paid 0, which by itself
    # made an EMPTY-tier control 3.956x faster than cold. The compiles are per prefill
    # shape bucket, so a short warm-up request does not reach them -- it has to be this
    # prompt. Its spill dir is discarded so it leaves the disk tier untouched.
    warm_dir = args.spill + "_warmup"
    shutil.rmtree(warm_dir, ignore_errors=True)
    os.makedirs(warm_dir, exist_ok=True)
    _arm(args, "jitwarm", warm_dir, prompt)
    shutil.rmtree(warm_dir, ignore_errors=True)

    # Turn 2 is user, ASSISTANT, user -- a real second turn, not the prompt with more text
    # appended. The engine publishes `req.tokens[:materialized]` during decode, so the entry
    # on disk is the prompt plus the reply it generated; a turn 2 that omits the reply
    # diverges from it at the first generated token and cannot hit at any length. Measured:
    # every arm 0 SSD hits with 1 entry recovered (errors/2026-09-07).
    rows = [_arm(args, "cold", main_dir, prompt)]
    print(json.dumps(rows[-1]), flush=True)
    reply = rows[0]["reply"]
    if not reply:
        raise SystemExit("cold arm returned no reply text; turn 2 cannot be built from it")
    # The spill was just WRITTEN, so it is in page cache. Evict it, or the faulted arm
    # measures memory and reports it as disk.
    print(json.dumps({"evict": _evict_cache(main_dir)}), flush=True)
    # Before the ratio: do turn 2's ids actually START with the spilled entry's ids?
    # The entry is engine token ids (prompt + generated); turn 2 is that reply rendered
    # back through the chat template, which re-tokenizes. A thinking block re-renders
    # with think markup and a boundary token can merge with the followup's first token.
    # A mismatch here means the BENCH cannot hit, not that the engine misses.
    print(json.dumps(_prefix_check(main_dir, args, prompt, reply)), flush=True)
    rows.append(_arm(args, "faulted", main_dir, prompt, reply=reply))
    print(json.dumps(rows[-1]), flush=True)
    rows.append(_arm(args, "control", ctrl_dir, prompt, reply=reply))
    print(json.dumps(rows[-1]), flush=True)

    # Below the break-even, on purpose. Without it the bench cannot tell "the threshold
    # works" from "fetching is always better": every other arm is above n*, so a build
    # that ignored the threshold entirely would produce the same three rows. This arm's
    # pass condition is that it does NOT prefetch.
    short_dir = args.spill + "_short"
    shutil.rmtree(short_dir, ignore_errors=True)
    os.makedirs(short_dir, exist_ok=True)
    n_star = rows[1].get("break_even_tokens") or 0
    short_tokens = max(16, n_star // 2)   # 16 = BLOCK_TOKENS; this script drives a server
                                          # over HTTP and does not import the package
    if 0 < n_star < (1 << 31):
        short = _prompt(short_tokens)
        rows.append(_arm(args, "below_break_even", short_dir, short))
        print(json.dumps(rows[-1]), flush=True)
        rows.append(_arm(args, "below_break_even_2nd", short_dir, short + " " + _FOLLOWUP))
        print(json.dumps(rows[-1]), flush=True)
    else:
        # Say it out loud. `n_star == 0` fails `0 < n_star` and used to skip in silence,
        # so a run that never tested the threshold read exactly like one that passed it.
        print(json.dumps({"below_break_even": "SKIPPED", "n_star": n_star, "why":
                          "no prompt can sit below a break-even of 0 (an unmeasured tier "
                          "answers 0 so the first fetch can calibrate B); the threshold "
                          "went untested this run"}), flush=True)

    cold, faulted, control = rows[:3]  # the break-even arms append past these three
    # The ceiling: a hit can save at most the prefill of the tokens it actually covered,
    # at the cold arm's own per-token rate. `matched` is the longest entry that is a
    # strict prefix of turn 2 -- `_match_prefix` treats a full-length hit as a miss, so
    # the whole-prompt entry cannot be the one that served.
    matched = _matched_tokens(main_dir, faulted["prompt_tokens"])
    ceiling_s = matched * cold["ms_per_prompt_token"] / 1000
    saved = control["wall_s"] - faulted["wall_s"]  # vs the empty-tier arm, not vs cold:
    # cold also pays first-start costs the other two do not, so `cold - faulted` credits
    # the tier with start order. `control` is the same arm position with an empty tier.
    # The check that caught this bench measuring its own page cache: divide the bytes a
    # fault-in must read by the device's measured bandwidth. If the whole arm is faster
    # than that read, the read did not come from the device.
    entry_mib, dev_mib_s = _entry_bytes(main_dir) / 2**20, args.device_mib_s
    read_s = entry_mib / dev_mib_s if dev_mib_s else 0.0
    # Two numbers, two scenarios, both measured -- not one number and one confound.
    #
    # A process restart (the case ckl asked about) empties HBM and leaves the HOST page
    # cache alone, so the fault-in legitimately reads from memory: that is the faulted
    # arm's wall clock, `speedup_faulted_over_control`. A host reboot, or a spill old
    # enough to have been evicted, pays the disk: that is `composed_tier_s`, the measured
    # standalone read plus the prefill of the tokens the hit did not cover.
    #
    # This distinction is why the arm's ratio is reported rather than discarded. What it
    # is NOT is a disk number, and three runs of it read like one until the bytes were
    # divided by the device's measured bandwidth (320.6 MiB / 182.6 MiB/s = 1.756 s
    # against a 1.690 s arm). fsync+fadvise per file moved the probe from 4477.8 MiB/s to
    # ~509 and no further, since DONTNEED only drops pages nothing else references.
    tail_tokens = max(0, faulted["prompt_tokens"] - matched)
    tail_s = tail_tokens * cold["ms_per_prompt_token"] / 1000
    composed_s = read_s + tail_s
    verdict = {
        "matched_tokens": matched,
        "entry_mib": round(entry_mib, 1),
        "device_mib_s": dev_mib_s,
        "implied_read_s": round(read_s, 3),
        "tail_tokens": tail_tokens,
        "tail_prefill_s": round(tail_s, 3),
        "composed_tier_s": round(composed_s, 3),
        "cold_prefill_s": cold["wall_s"],
        "composed_speedup": round(cold["wall_s"] / composed_s, 3) if composed_s else None,
        # Bandwidth at which reading the entry costs exactly what prefilling it saves.
        "break_even_mib_s": round(entry_mib / (cold["wall_s"] - tail_s), 1)
        if cold["wall_s"] > tail_s else None,
        "ceiling_s": round(ceiling_s, 3),
        "saved_s": round(saved, 3),
        "saved_over_ceiling": round(saved / ceiling_s, 3) if ceiling_s else None,
        # The win is against the CONTROL, not against cold: control is the same arm order
        # with an empty tier, so it carries whatever start-order effect remains.
        "speedup_faulted_over_control": round(control["wall_s"] / faulted["wall_s"], 3),
        "speedup_faulted_over_cold": round(cold["wall_s"] / faulted["wall_s"], 3),
        "control_over_cold": round(control["wall_s"] / cold["wall_s"], 3),
        "faulted_recovered_entries": faulted["ssd_recovered"],
        "faulted_ssd_hits": faulted["ssd_hits"],
        "control_ssd_hits": control["ssd_hits"],
        # Row 50 PR B. tick_loads is what says the fetch was asynchronous: a fault served
        # by a torch.load on the calling thread is the old synchronous path and would show
        # the same wall clock on this arm, since nothing else is running.
        "faulted_prefetches": faulted["ssd_prefetches"],
        "faulted_fetches_ready": faulted["ssd_fetches_ready"],
        "faulted_tick_loads": faulted["ssd_tick_loads"],
        "faulted_fetch_waits": faulted["ssd_fetch_waits"],
        "faulted_fetch_drops": faulted["ssd_fetch_drops"],
        "prefill_rate": faulted.get("prefill_rate"),
        "break_even_tokens": faulted.get("break_even_tokens"),
    }
    below = [r for r in rows if r["arm"].startswith("below_break_even")]
    if below:
        verdict["below_break_even_prefetches"] = sum(r["ssd_prefetches"] for r in below)
        verdict["below_break_even_tokens"] = below[0]["prompt_tokens"]
        if verdict["below_break_even_prefetches"]:
            verdict["INVALID"] = (
                f"a {below[0]['prompt_tokens']}-token prompt prefetched with a break-even "
                f"of {verdict['break_even_tokens']}: the threshold is not gating anything, "
                "so the other arms measure 'fetching is always on', not 'fetching wins'"
            )
    elif verdict["break_even_tokens"] in (None, 1 << 31):
        verdict["below_break_even"] = (
            "skipped: no finite break-even was reported, so no prompt can be placed "
            "below it -- the threshold went untested this run"
        )
    # The assertions that decide whether the number means anything.
    if any(r["compiles"] for r in rows):
        verdict["INVALID"] = (
            "TileLang compiled inside a measured window ("
            + ", ".join(f"{r['arm']}={r['compiles']}" for r in rows)
            + "), so the arms differ by JIT and not by the tier"
        )
    elif faulted["ssd_hits"] < 1:
        verdict["INVALID"] = (
            f"the faulted arm took {faulted['ssd_hits']} SSD hits with "
            f"{faulted['ssd_recovered']} entries recovered, so whatever it measured was "
            "not the tier"
        )
    elif control["ssd_hits"] != 0:
        verdict["INVALID"] = (
            f"the control arm took {control['ssd_hits']} SSD hits from a directory that "
            "was created empty"
        )
    elif verdict["control_over_cold"] > 1.15 or verdict["control_over_cold"] < 0.85:
        verdict["INVALID"] = (
            f"the control ran at {verdict['control_over_cold']}x cold with an EMPTY tier, "
            "so arm order alone moves the wall clock and neither speedup is the tier's"
        )
    elif matched and saved > ceiling_s:
        verdict["INVALID"] = (
            f"saved {saved:.3f} s against a ceiling of {ceiling_s:.3f} s "
            f"({matched} matched tokens x cold's {cold['ms_per_prompt_token']} ms/tok) -- "
            "a hit cannot save more prefill than it covered, so something else moved"
        )
    elif read_s and faulted["wall_s"] < read_s:
        verdict["SCENARIOS"] = (
            f"restart (host cache warm, the common case): "
            f"{verdict['speedup_faulted_over_control']}x, {faulted['wall_s']:.3f} s vs "
            f"{control['wall_s']:.3f} s -- a process restart empties HBM but not the host "
            f"page cache, so the fault-in reads from memory. "
            f"host reboot / evicted cache: {verdict['composed_speedup']}x, "
            f"{composed_s:.3f} s composed from a measured {read_s:.3f} s disk read plus "
            f"{tail_s:.3f} s of tail prefill. Both are real; they answer different questions."
        )
    print(json.dumps(verdict, indent=2), flush=True)
    # Exit nonzero on INVALID. Printing it and returning 0 makes a bench that measured
    # nothing indistinguishable from one that passed, to a launcher that reads rc.
    if "INVALID" in verdict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
