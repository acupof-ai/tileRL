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


def _matched_tokens(spill: str) -> int:
    """Tokens covered by the entry that can actually serve, from the spill file sizes.

    Every .kv is a whole number of publish units, so the sizes give the coverage ladder
    directly, and the LONGEST entry is the one that serves: the measured request is turn 1
    plus a follow-up, so turn 1's prompt-complete publish is a strict prefix of it. (When
    the two requests are identical the longest entry is a full-length match, which
    `_match_prefix` treats as a miss -- that is a bench artifact, and the fix is the
    follow-up rather than reading the second-longest here.)
    """
    d = os.path.join(spill, "tilerl_kvtier")
    sizes = sorted(
        os.path.getsize(os.path.join(d, f))
        for f in os.listdir(d) if f.endswith(".kv")
    ) if os.path.isdir(d) else []
    if not sizes:
        return 0
    return round(sizes[-1] / sizes[0]) * _UNIT_TOKENS


#: Tokens per publish unit, derived from the size ladder rather than assumed: the smallest
#: spill was 33557933 B and the largest 179316717 B = 5.343x it, which lands on the whole
#: 2729-token prompt (171 blocks x 16 = 2736 slots) only at 512 tokens per unit.
_UNIT_TOKENS = 512


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


def _arm(args, name: str, spill: str, prompt: str) -> dict:
    log = f"/work/ssd_restart_{name}.log"
    proc = _serve(args, spill, log)
    try:
        _wait_up(args.port, proc, args.boot_s)
        before = _stats(args.port)
        t0 = time.monotonic()
        r = _post(f"http://127.0.0.1:{args.port}/v1/messages",
                  {"model": args.model, "max_tokens": args.gen,
                   "messages": [{"role": "user", "content": prompt}]}, args.req_s)
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
        "ssd_hits": d("ssd_hits"),
        "ssd_faults": d("ssd_faults"),
        "ssd_entries": int(after.get("ssd_entries", 0)),
        "ssd_recovered": int(after.get("ssd_recovered", 0)),
        "ssd_offered": d("ssd_offered"),
        "ssd_refusals": d("ssd_refusals"),
        "prefix_hits": d("prefix_hits"),
        "prefix_published": d("prefix_published"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--python", default="/work/tl013/bin/python")
    ap.add_argument("--repo", default="/work/tilerl")
    ap.add_argument("--spill", default="/work/ssd_tier_bench")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--tokens", type=int, default=3000, help="target prompt length")
    ap.add_argument("--gen", type=int, default=8, help="tokens to generate; keep small so "
                    "the wall clock is prefill")
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

    # Turn 2 is turn 1 plus more text -- that is what a chat client sends, and it is what
    # makes the tier's LAST publish the entry that serves. Re-sending the IDENTICAL prompt
    # instead makes the longest stored entry a full-length match, which `_match_prefix`
    # treats as a miss, so the served entry would be the second-longest and the bench would
    # disagree with production about which publish matters.
    turn2 = prompt + " " + _FOLLOWUP
    rows = [_arm(args, "cold", main_dir, prompt)]
    print(json.dumps(rows[-1]), flush=True)
    # The spill was just WRITTEN, so it is in page cache. Evict it, or the faulted arm
    # measures memory and reports it as disk.
    print(json.dumps({"evict": _evict_cache(main_dir)}), flush=True)
    rows.append(_arm(args, "faulted", main_dir, turn2))
    print(json.dumps(rows[-1]), flush=True)
    rows.append(_arm(args, "control", ctrl_dir, turn2))
    print(json.dumps(rows[-1]), flush=True)

    cold, faulted, control = rows
    # The ceiling: a hit can save at most the prefill of the tokens it actually covered,
    # at the cold arm's own per-token rate. `matched` is read off the largest SERVABLE
    # entry, i.e. the second-largest spill -- `_match_prefix` treats a full-length hit as
    # a miss, so the whole-prompt entry cannot be the one that served.
    matched = _matched_tokens(main_dir)
    ceiling_s = matched * cold["ms_per_prompt_token"] / 1000
    saved = cold["wall_s"] - faulted["wall_s"]
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
    }
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
    elif matched and saved > ceiling_s:
        verdict["INVALID"] = (
            f"saved {saved:.3f} s against a ceiling of {ceiling_s:.3f} s "
            f"({matched} matched tokens x cold's {cold['ms_per_prompt_token']} ms/tok) -- "
            "a hit cannot save more prefill than it covered, so something else moved"
        )
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
