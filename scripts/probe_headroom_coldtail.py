#!/usr/bin/env python3
"""One-command per-arm load for the --device-headroom-mib V100 cold-tail test
(#702; runbook: docs/experience/errors/2026-09-17-cold-tier-full-finalize-
relocation-long-tail.md).

The operator boots ONE arm's serve (TILERL_DEVICE_HEADROOM_MIB=0|384|768,
TILERL_DRAFT_ATTN_WINDOW_TOKENS=<the baseline W, 2048>, TILERL_STEP_TIMING=1,
TILERL_STEP_TIMING_SLOW_MS=0) and then runs this against it. It does NOT
restart the server — restarting belongs to ops, one boot per arm, so two flows
never race. It drives the full load against the already-up arm and ends with a
single machine-readable line:

    ARM_DONE <headroom_mib> <median_tok_s> reps=<good>/<N> W=[<engaged>]  (exit 0)

For each of --warm-reps (default 3) independent repetitions it REFILLS the cold
tier then runs one warm, so the reported tok/s is a median with a spread rather
than one noisy shot:
  1. fill  — N independent (non-prefix-sharing) >=32k sparse prompts force the
     shared cold pool to the full-tier state (the W-sweep filler shape); assert
     cold occupancy actually crossed --min-cold-gb before measuring;
  2. warm  — one primed-warm 32k prompt streamed; time first-token -> last-token
     (decode-only tok/s, prefill excluded), the baseline's quantity;
  3. read  — /health build fields + spec_drafted/spec_accepted delta (accept
     rate) + nvidia-smi physical free; parse THIS rep's server log window
     (byte offset at the warm POST, dec>0 ticks only) and split type-1 finalize
     batches (offers_pages>0) from type-2 hollow forwards (no offers, 1.1-1.4 s,
     inner model well under total);
  4. cross-check the serve's actually-engaged draft READ window from its
     `[draft-window] W=` self-proof line against --expect-window; a mismatch or
     a missing engage (when W>0 was armed) fails the rep.

Whether the window is armed is a BUILD-TIME constant, and the engine logs each
(batch, first-page) shape's `[draft-window] W=` line only once for the whole
boot — fill's 32k decodes already print the same-shape line, so a post-warm
offset search finds nothing and false-reports a mismatch. The engage check is
therefore existence evidence read from the WHOLE log (offset 0); the warm byte
offset bounds only the decode-tick distribution.

Fail-closed: rc3 cold tier under --min-cold-gb; rc13 no/too-few decode ticks,
window mismatch, or fewer than --min-good-reps good reps — none prints ARM_DONE.

Writes one JSON summary per arm and prints ARM_DONE. SLOW_MS must be 0 so every
tick is logged — p50/p90 and the >300 ms fraction need the full distribution.
Pure stdlib, no torch; runs from the laptop against the pod URL. The baseline
(2.69/4.25/6.56/7.86 tok/s) was measured at W=2048, so every arm arms
TILERL_DRAFT_ATTN_WINDOW_TOKENS=2048 and passes --expect-window 2048; the warm
quantity is first-token→last-token decode tok/s over a 32k context.

Run: python3 scripts/probe_headroom_coldtail.py --self-check
     python3 scripts/probe_headroom_coldtail.py arm --url http://127.0.0.1:8000 \
         --headroom 768 --log /root/servehybridsse.log --out /root/arm768.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import uuid

MODEL = "qwen38-27b"
# One 32k slot's hot ceiling (n_groups*k + WINDOW + chunk = 4*128+8+33); blocks.
HOT_BLOCKS_PER_SLOT = 553

_WORDS = [
    "paged",
    "cache",
    "scheduler",
    "latency",
    "block",
    "allocation",
    "prefix",
    "sharing",
    "cold",
    "tier",
    "migration",
    "attention",
    "kernel",
    "quantized",
    "weight",
    "draft",
    "head",
    "graph",
    "capture",
    "ledger",
    "capacity",
    "planning",
    "eviction",
    "promote",
    "demote",
    "resident",
    "window",
    "selection",
    "decode",
]


def prompt_32k(n_tokens: int, seed: int) -> str:
    """Independent filler: a uuid lead per seed so the N prompts share no prefix
    (shared pages would not fill the pool), then a deterministic word stream."""
    out = [f"Reference {uuid.UUID(int=(seed or 1) % (1 << 128)).hex}. "]
    i = 0
    while True:
        for w in _WORDS:
            out.append(w.upper() if (i + seed) % 11 == 0 else w)
            i += 1
            if i >= n_tokens:
                return " ".join(out) + "."


def _post(url: str, body: dict, timeout: float, stream: bool) -> dict:
    body = {**body, "stream": stream}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    n = 0
    t0 = t_first = t_last = time.perf_counter()
    saw = False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            s = raw.decode(errors="replace").strip()
            if not s.startswith("data:"):
                continue
            s = s[5:].strip()
            if s == "[DONE]":
                break
            d = json.loads(s)["choices"][0]["delta"].get("content", "")
            if d:
                now = time.perf_counter()
                if not saw:
                    t_first = now
                    saw = True
                t_last = now
                n += 1
    return {"chunks": n, "wall_s": t_last - t0, "decode_s": (t_last - t_first) if saw else 0.0}


def _body(prompt: str, gen: int) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": gen,
        "enable_thinking": False,
    }


def health(url: str) -> dict:
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        return json.loads(r.read())["stats"]


def wait_ready(url: str, trials: int) -> None:
    for _ in range(trials):
        try:
            health(url)
            return
        except OSError:
            time.sleep(2)
    raise SystemExit("server not reachable on /health within ready wait")


def _log_size(path: str) -> int:
    """Current byte length of the server log; the parse window starts here so
    fill-phase ticks (already flushed) are excluded. A missing/empty log is 0."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def nvidia_free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


# --- log parsing -------------------------------------------------------------

_TICK = re.compile(r"\[step-timing\] tick (\d+) total=(-?\d+)ms(?: dec=(\d+) pre=(\d+))? (.*)")
_KV = re.compile(r"(\w+)=(-?\d+)ms")
_OFFERS = re.compile(r"offers_pages=(\d+)")
_FREE = re.compile(r" free=(-?\d+)MiB")
_WLINE = re.compile(r"\[draft-window\] W=(\d+)")


def parse_tick(line: str) -> dict | None:
    m = _TICK.search(line)
    if not m:
        return None
    n, total = int(m.group(1)), int(m.group(2))
    # Older logs predate the dec/pre tag; treat untagged as decode-less so they
    # are excluded rather than silently counted (fail-safe toward prefill).
    dec = int(m.group(3)) if m.group(3) is not None else 0
    pre = int(m.group(4)) if m.group(4) is not None else 0
    rest = m.group(5)
    segs = {k: int(v) for k, v in _KV.findall(rest)}
    mo = _OFFERS.search(rest)
    offers = int(mo.group(1)) if mo else 0
    mf = _FREE.search(rest)
    hollow = offers == 0 and total >= 1100 and segs.get("model", 0) <= int(0.6 * total)
    return {
        "n": n,
        "total_ms": total,
        "dec": dec,
        "pre": pre,
        "is_decode": dec > 0,
        "offers_pages": offers,
        "finalize_ms": segs.get("sparse_finalize", 0),
        "model_ms": segs.get("model", 0),
        "free_mib": int(mf.group(1)) if mf else None,
        "type1": dec > 0 and offers > 0,
        "type2": dec > 0 and hollow,
    }


def observed_draft_window(log_path: str, byte_offset: int = 0) -> int | None:
    """The draft READ window the serve engaged, from its self-proof
    `[draft-window] W=<tokens>` line, read at/after ``byte_offset``.

    The arm passes byte_offset=0: whether a window is armed is a build-time
    constant and each (batch, first-page) shape logs the line once per boot, so
    the evidence may sit in the fill phase long before the warm offset. Use the
    warm byte offset ONLY to bound the decode-tick distribution, never this."""
    seen = None
    with open(log_path, errors="replace") as fh:
        fh.seek(byte_offset)
        for line in fh:
            m = _WLINE.search(line)
            if m:
                seen = int(m.group(1))  # build constant; every shape logs the same W
    return seen


def pct(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def fpct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def parse_log(log_path: str, byte_offset: int = 0, byte_end: int | None = None) -> dict:
    """Parse ticks in ``[byte_offset, byte_end)`` and keep DECODE ticks only. The
    fill phase's chunked prefills and the warm request's own prefill ticks (pre>0)
    are dropped: the baseline distribution is warm DECODE ticks alone. With
    SLOW_MS=0 every tick is logged, so this is the real decode distribution, not a
    >threshold tail.

    ``byte_end`` is the second boundary of a rep's warm window. ``byte_offset``
    alone is a warm START, so a reader that runs to EOF takes in the NEXT rep's fill
    -- whose short-context decode ticks pass the standard steady set. The caller
    passes the log size taken at THIS point in the rep, before that fill exists, so
    the window is exact; ``None`` means read to EOF and report where it ended.
    """
    out = []
    with open(log_path, errors="replace") as fh:
        fh.seek(byte_offset)
        while True:
            pos = fh.tell()
            if byte_end is not None and pos >= byte_end:
                break
            line = fh.readline()
            if not line:
                break
            out.append(line)
        end = fh.tell() if byte_end is None else byte_end
    raw = out
    all_ticks = [t for line in raw if (t := parse_tick(line))]
    ticks = [t for t in all_ticks if t["is_decode"]]
    totals = [t["total_ms"] for t in ticks]
    t1 = [t for t in ticks if t["type1"]]
    t2 = [t for t in ticks if t["type2"]]
    fin = [t["finalize_ms"] for t in t1 if t["finalize_ms"] > 0]
    pages = [t["offers_pages"] for t in t1]
    return {
        "log_byte_offset": byte_offset,
        "log_byte_end": end,
        "ticks_in_window": len(all_ticks),
        "decode_ticks": len(ticks),
        "dropped_prefill_ticks": len(all_ticks) - len(ticks),
        "p50_ms": pct(totals, 0.50),
        "p90_ms": pct(totals, 0.90),
        "max_ms": max(totals, default=0),
        "frac_over_300": round(len([x for x in totals if x > 300]) / len(totals), 3)
        if totals
        else 0.0,
        "type1_finalize_ticks": len(t1),
        "type1_finalize_ms_median": pct(fin, 0.5),
        "type1_finalize_ms_max": max(fin, default=0),
        "type1_offers_pages_median": pct(pages, 0.5),
        "type1_offers_pages_max": max(pages, default=0),
        "type2_hollow_ticks": len(t2),
        "steady_free_mib": ticks[-1]["free_mib"] if ticks else None,
    }


# --- the one-command arm -----------------------------------------------------


def _cold_tier_gb(h: dict) -> tuple[float, float, float, float, float]:
    # Cold occupancy across BOTH tiers: private/shared pages in pinned host RAM
    # plus their SSD spill (each has its own /health key). Excluding SSD was
    # correct for the old RAM-only 8GiB tier; with --kv-cold-bytes 1GiB +
    # --cold-ssd-bytes most cold bytes live under kv_cold_*_ssd_bytes and a
    # RAM-only gate can never pass (the #731 private-vs-shared under-count
    # applies per tier, so read all four keys, not a merged name).
    priv = h.get("kv_cold_bytes", 0) / 2**30
    shared = h.get("kv_cold_shared_bytes", 0) / 2**30
    priv_ssd = h.get("kv_cold_ssd_bytes", 0) / 2**30
    shared_ssd = h.get("kv_cold_shared_ssd_bytes", 0) / 2**30
    return priv, shared, priv_ssd, shared_ssd, priv + shared + priv_ssd + shared_ssd


def _fill_cold_tier(a, rep: int) -> tuple[dict, list[int]]:
    """Drive fill_n independent 32k sparse prompts; return (health, prompt tokens)."""
    rows = []
    for i in range(a.fill_n):
        out = _post(
            a.url + "/v1/chat/completions",
            _body(prompt_32k(a.prompt_tokens, rep * 1000 + i + 1), a.fill_gen),
            a.timeout,
            stream=False,
        )
        pt = out.get("usage", {}).get("prompt_tokens", 0)
        rows.append(pt)
        if pt < a.prompt_tokens * 0.9:
            print(
                f"WARN rep{rep} fill {i}: only {pt} prompt tokens (want ~{a.prompt_tokens})",
                flush=True,
            )
    return health(a.url), rows


def _one_warm(a, rep: int) -> dict | None:
    """One independent refill + primed-warm 32k, return the rep record or None on a
    fail-closed condition (caller counts it; None never yields a clean ARM_DONE)."""
    hfill, pt_rows = _fill_cold_tier(a, rep)
    cold_priv_gb, cold_shared_gb, cold_priv_ssd_gb, cold_shared_ssd_gb, cold_gb = _cold_tier_gb(
        hfill
    )
    if cold_gb < a.min_cold_gb:
        print(
            f"REP{rep}-FILL-INSUFFICIENT cold_total={cold_gb:.2f}GiB "
            f"(ram_priv={cold_priv_gb:.2f} ram_shared={cold_shared_gb:.2f} "
            f"ssd_priv={cold_priv_ssd_gb:.2f} ssd_shared={cold_shared_ssd_gb:.2f}) "
            f"< {a.min_cold_gb}GiB",
            flush=True,
        )
        return None
    # hfill is the post-fill /health snapshot: it both carries the cold totals
    # and is the spec-counter baseline (fill drafts its own gen tokens), so the
    # warm accept rate is isolated to the warm request without a second poll.
    log_offset = _log_size(a.log)
    s = _post(
        a.url + "/v1/chat/completions",
        _body(prompt_32k(a.prompt_tokens, rep * 1000), a.warm_gen),
        a.timeout,
        stream=True,
    )
    h_spec_after = health(a.url)
    # Second boundary of this rep's warm window. Taken here, before the NEXT rep's
    # fill starts, so the size at this instant is exactly the end of this rep's warm
    # decode -- no later phase can have written past it. Without an end, a reader
    # starting at log_offset runs into the next rep's fill, whose short-context
    # decode ticks pass the standard steady set.
    log_end = _log_size(a.log)
    if s["chunks"] < 2 or s["decode_s"] <= 0:
        print(f"REP{rep}-NO-DECODE chunks={s['chunks']} decode_s={s['decode_s']}", flush=True)
        return None
    ticks = parse_log(a.log, log_offset, log_end)
    if ticks["decode_ticks"] < a.min_decode_ticks:
        print(
            f"REP{rep}-TOO-FEW-DECODE-TICKS decode={ticks['decode_ticks']} "
            f"window={ticks['ticks_in_window']} dropped_prefill="
            f"{ticks['dropped_prefill_ticks']} < --min-decode-ticks "
            f"{a.min_decode_ticks}",
            flush=True,
        )
        return None
    # Build-time constant: read the engage line from the WHOLE log (offset 0),
    # since fill's same-shape decode already printed it and the engine logs each
    # shape once per boot. The warm offset would find nothing and false-fail.
    w_obs = observed_draft_window(a.log, 0)
    if a.expect_window > 0 and w_obs != a.expect_window:
        print(
            f"REP{rep}-WINDOW-MISMATCH armed W={a.expect_window} "
            f"serve engaged W={w_obs}; do not compare against a W=2048 baseline",
            flush=True,
        )
        return None
    if a.expect_window == 0 and w_obs not in (None, 0):
        print(f"REP{rep}-WINDOW-MISMATCH armed W=0 but serve engaged W={w_obs}", flush=True)
        return None
    drafted_d = h_spec_after.get("spec_drafted", 0) - hfill.get("spec_drafted", 0)
    accepted_d = h_spec_after.get("spec_accepted", 0) - hfill.get("spec_accepted", 0)
    accept_rate = round(accepted_d / drafted_d, 4) if drafted_d > 0 else None
    return {
        "rep": rep,
        "decode_tok_s": round((s["chunks"] - 1) / s["decode_s"], 3),
        "content_chunks": s["chunks"],
        "first_to_last_s": round(s["decode_s"], 3),
        "wall_s": round(s["wall_s"], 3),
        "engaged_draft_window_w": w_obs if a.expect_window > 0 else 0,
        "cold_total_gb": round(cold_gb, 3),
        "cold_private_gb": round(cold_priv_gb, 3),
        "cold_shared_gb": round(cold_shared_gb, 3),
        "cold_private_ssd_gb": round(cold_priv_ssd_gb, 3),
        "cold_shared_ssd_gb": round(cold_shared_ssd_gb, 3),
        "fill_prompt_tokens": pt_rows,
        "spec_drafted_delta": drafted_d,
        "spec_accepted_delta": accepted_d,
        "spec_accept_rate": accept_rate,
        "ticks": ticks,
        "device_free_mib": h_spec_after.get("device_free_bytes", 0) >> 20,
    }


def cmd_arm(a) -> int:
    wait_ready(a.url, a.ready_trials)
    h0 = health(a.url)
    cold0 = _cold_tier_gb(h0)
    cold0_total = cold0[-1]
    reps = []
    for r in range(a.warm_reps):
        rec = _one_warm(a, r)
        if rec is None:
            # Fail fast: a later rep cannot refill what this rep's gate proved
            # absent, and every extra fill grows the SSD spill high-water file.
            break
        reps.append(rec)
    # Fail closed: too few good reps must never print ARM_DONE or a clean median.
    if len(reps) < a.min_good_reps:
        print(
            f"NO-GOOD-REPS good={len(reps)}/{a.warm_reps} "
            f"< --min-good-reps {a.min_good_reps}; refusing ARM_DONE",
            flush=True,
        )
        return 13
    hw = health(a.url)
    blocks_total = hw.get("blocks_total", 0)
    toks = [r["decode_tok_s"] for r in reps]
    rates = [r["spec_accept_rate"] for r in reps if r["spec_accept_rate"] is not None]
    windows = sorted({r["engaged_draft_window_w"] for r in reps})
    summary = {
        "arm_headroom_mib": a.headroom,
        "expect_draft_window_w": a.expect_window,
        "engaged_draft_window_w_values": windows,
        "warm_reps": a.warm_reps,
        "good_reps": len(reps),
        "warm32k": {
            "decode_tok_s_median": fpct(toks, 0.5),
            "decode_tok_s_min": min(toks),
            "decode_tok_s_max": max(toks),
            "decode_tok_s_spread": round(max(toks) - min(toks), 3),
            "per_rep_tok_s": toks,
        },
        "spec_accept_rate_median": fpct(rates, 0.5) if rates else None,
        "spec_accept_rate_per_rep": rates,
        "fill": {
            "n_per_rep": a.fill_n,
            "cold_total_gb_first": reps[0]["cold_total_gb"],
            "cold_total_gb_last": reps[-1]["cold_total_gb"],
            "cold_private_gb_last": reps[-1]["cold_private_gb"],
            "cold_shared_gb_last": reps[-1]["cold_shared_gb"],
            "cold_private_ssd_gb_last": reps[-1]["cold_private_ssd_gb"],
            "cold_shared_ssd_gb_last": reps[-1]["cold_shared_ssd_gb"],
            "cold_total_before_gb": round(cold0_total, 3),
        },
        "health": {
            "sparse_headroom_bytes": hw.get("sparse_headroom_bytes", 0),
            "sparse_headroom_dropped_blocks": hw.get("sparse_headroom_dropped_blocks", 0),
            "blocks_total": blocks_total,
            "resident_slots_floor": blocks_total // HOT_BLOCKS_PER_SLOT,
            "slots_total_state_pool": hw.get("slots_total", 0),
            "process_free_mib": hw.get("device_free_bytes", 0) >> 20,
            "physical_free_mib_nvidia": nvidia_free_mib(),
        },
        "reps": reps,
    }
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    med = summary["warm32k"]["decode_tok_s_median"]
    print(f"ARM_DONE {a.headroom} {med} reps={len(reps)}/{a.warm_reps} W={windows}", flush=True)
    return 0


def cmd_compare(a) -> int:
    """Lay the three arm summaries next to the historical full/empty bands.
    Run after all three ARM_DONE; it decides nothing, it places the numbers."""
    arms = {}
    for spec in a.arms:
        name, path = spec.split("=", 1)
        with open(path) as fh:
            arms[name] = json.load(fh)
    base = {
        "full_tier_tok_s": [2.69, 4.25, 6.56],
        "empty_tier_tok_s": 7.86,
        "full_finalize_ms": [629, 667],
        "empty_finalize_ms": [81, 186],
        "full_frac_over_300": [0.40, 0.58],
        "empty_frac_over_300": 0.10,
    }
    rows = {}
    for name, s in sorted(arms.items()):
        h, w3 = s["health"], s["warm32k"]
        rep_ticks = [r["ticks"] for r in s.get("reps", [])]
        # Pool the per-rep tick distributions (n varies per rep) so the reported
        # p50/p90/finalize reflect every measured warm decode tick, not rep 1.
        all_p50 = [t["p50_ms"] for t in rep_ticks]
        all_p90 = [t["p90_ms"] for t in rep_ticks]
        all_max = [t["max_ms"] for t in rep_ticks]
        all_frac = [t["frac_over_300"] for t in rep_ticks]
        all_fin_med = [t["type1_finalize_ms_median"] for t in rep_ticks]
        all_fin_max = [t["type1_finalize_ms_max"] for t in rep_ticks]
        all_off_med = [t["type1_offers_pages_median"] for t in rep_ticks]
        all_off_max = [t["type1_offers_pages_max"] for t in rep_ticks]
        t1n = sum(t["type1_finalize_ticks"] for t in rep_ticks)
        t2n = sum(t["type2_hollow_ticks"] for t in rep_ticks)
        rows[name] = {
            "headroom_mib": s["arm_headroom_mib"],
            "draft_window_w": s.get("expect_draft_window_w"),
            "warm32k_tok_s_median/min/max": [
                w3["decode_tok_s_median"],
                w3["decode_tok_s_min"],
                w3["decode_tok_s_max"],
            ],
            "tok_s_spread": w3["decode_tok_s_spread"],
            "per_rep_tok_s": w3["per_rep_tok_s"],
            "good_reps": s.get("good_reps"),
            "spec_accept_rate_median": s.get("spec_accept_rate_median"),
            "p50/p90/max_ms_median_across_reps": [
                pct(all_p50, 0.5),
                pct(all_p90, 0.5),
                max(all_max),
            ],
            "frac_over_300_range": [min(all_frac), max(all_frac)] if all_frac else [],
            "finalize_med/max_ms": [pct(all_fin_med, 0.5), max(all_fin_max)],
            "offers_med/max_pages": [pct(all_off_med, 0.5), max(all_off_max)],
            "type1_ticks_total": t1n,
            "type2_hollow_ticks_total": t2n,
            "blocks_total": h["blocks_total"],
            "resident_slots_floor": h["resident_slots_floor"],
            "slots_total": h["slots_total_state_pool"],
            "dropped_blocks": h["sparse_headroom_dropped_blocks"],
            "process_free_mib": h["process_free_mib"],
            "physical_free_mib": h["physical_free_mib_nvidia"],
            "cold_total_gb_last": s["fill"].get("cold_total_gb_last"),
        }
    print(json.dumps({"baseline": base, "arms": rows}, indent=1))
    return 0


def _self_check(ns=None) -> int:
    l1 = (
        "[step-timing] tick 7 total=1259ms dec=1 pre=0 plan=1ms stats=2ms "
        "forward=1250ms sparse_select=3ms model=330ms sparse_finalize=629ms "
        "sample=1ms [offers_pages=132] free=210MiB reserved=30000MiB path=eager "
        "sparse=1 fwd_host=1250ms fwd_gpu=400ms why=sync_wait"
    )
    t = parse_tick(l1)
    assert t["is_decode"] and t["type1"] and not t["type2"]
    assert t["finalize_ms"] == 629 and t["offers_pages"] == 132 and t["free_mib"] == 210
    # A finalize batch on a PREFILL tick is not a type-1 decode tick.
    lp = l1.replace("dec=1 pre=0", "dec=0 pre=1")
    tp = parse_tick(lp)
    assert tp["is_decode"] is False and not tp["type1"]
    l2 = (
        "[step-timing] tick 8 total=1303ms dec=1 pre=0 plan=1ms forward=1300ms "
        "model=350ms sample=1ms  free=206MiB reserved=30000MiB path=eager "
        "sparse=1 fwd_host=1300ms fwd_gpu=420ms why=sync_wait"
    )
    t = parse_tick(l2)
    assert t["is_decode"] and t["type2"] and not t["type1"] and t["model_ms"] == 350
    t = parse_tick("[step-timing] tick 9 total=180ms dec=1 pre=0 forward=176ms model=170ms why=cpu")
    assert t and t["is_decode"] and not t["type1"] and not t["type2"]
    # An untagged legacy line decodes as non-decode (fail-safe toward exclusion).
    t = parse_tick("[step-timing] tick 10 total=200ms forward=199ms model=199ms")
    assert t and t["is_decode"] is False
    assert pct([100, 200, 300, 400], 0.5) == 200
    assert pct(list(range(100, 1100, 100)), 0.9) == 900  # 10 pts, 9th nearest-rank
    assert fpct([5.0, 9.0, 1.0, 3.0], 0.5) == 3.0
    assert fpct([2.0, 4.0], 0.0) == 2.0 and fpct([2.0, 4.0], 1.0) == 4.0
    assert len(prompt_32k(32000, 1).split()) > 30000
    assert prompt_32k(100, 1) != prompt_32k(100, 2)  # seeds -> independent

    # Window: fill prefill ticks BEFORE the offset and the warm request's own
    # prefill tick after it must both be dropped; only dec>0 ticks count.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write("[step-timing] tick 1 total=900ms dec=0 pre=1 model=900ms\n")  # fill
        fh.write("[step-timing] tick 2 total=800ms dec=0 pre=1 model=800ms\n")  # fill
        offset = fh.tell()
        fh.write("[step-timing] tick 3 total=700ms dec=0 pre=1 model=700ms\n")  # warm prefill
        fh.write("[step-timing] tick 4 total=170ms dec=1 pre=0 model=165ms\n")
        fh.write("[step-timing] tick 5 total=1300ms dec=1 pre=0 model=350ms\n")  # hollow
        logf = fh.name
    stats = parse_log(logf, offset)
    assert stats["ticks_in_window"] == 3
    assert stats["decode_ticks"] == 2, stats
    assert stats["dropped_prefill_ticks"] == 1
    assert stats["type2_hollow_ticks"] == 1
    assert stats["p50_ms"] == 170 and stats["max_ms"] == 1300
    os.unlink(logf)
    # All-prefill window: zero decode ticks -> the arm must fail closed, and the
    # fraction must not read as a clean 0.0 off an empty set.
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write("[step-timing] tick 1 total=900ms dec=0 pre=1 model=900ms\n")
        logf0 = fh.name
    s0 = parse_log(logf0, 0)
    assert s0["decode_ticks"] == 0 and s0["frac_over_300"] == 0.0
    os.unlink(logf0)

    # The window engage line is a build-time constant the engine prints once per
    # (batch, first-page) shape. Fill's 32k decode prints it BEFORE the warm
    # offset; the arm reads the WHOLE log (offset 0), so an engage line that only
    # exists in the fill phase is still recognized as armed. Searching from the
    # warm offset would return None and false-fail every rep (the #728 bug).
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write("[draft-window] W=2048 sq=[2] first=[115] windowed_seq_len=[2048]\n")
        wm_off = fh.tell()
        fh.write("[step-timing] tick 900 total=180ms dec=1 pre=0 model=170ms\n")
        logfw = fh.name
    assert observed_draft_window(logfw, 0) == 2048  # whole-log: armed
    assert observed_draft_window(logfw, wm_off) is None  # warm-offset: deduped
    assert observed_draft_window(logfw, 10_000) is None  # past EOF: nothing
    os.unlink(logfw)
    # An off (W=0) run writes no engage line anywhere -> None.
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write("[step-timing] tick 1 total=180ms dec=1 pre=0 model=170ms\n")
        logfo = fh.name
    assert observed_draft_window(logfo, 0) is None
    os.unlink(logfo)

    # Dispatch arity: arm/compare run as fn(namespace). A regressed ns.fn() call
    # only raises TypeError on the server arm (selfcheck never dispatches), so pin
    # the one-arg call here without a server.
    class _NS:
        def fn(self, ns):
            self.got = ns
            return 0

    nschk = _NS()
    assert _dispatch(nschk) == 0 and nschk.got is nschk
    # Cold fullness spans RAM and SSD, private and shared: a 1GiB-RAM/8GiB-SSD
    # tier reads RAM near its cap with most bytes on SSD; gating on RAM alone,
    # or dropping one SSD key, false-fails the full-tier check.
    gb = 2**30
    p, sh, ps, ss, tot = _cold_tier_gb(
        {
            "kv_cold_bytes": int(0.5 * gb),
            "kv_cold_shared_bytes": int(0.5 * gb),
            "kv_cold_ssd_bytes": int(1.0 * gb),
            "kv_cold_shared_ssd_bytes": int(5.5 * gb),
        }
    )
    assert (p, sh, ps, ss, tot) == (0.5, 0.5, 1.0, 5.5, 7.5), (p, sh, ps, ss, tot)
    _p, _sh, _ps, _ss, tot_no_ssd = _cold_tier_gb(
        {"kv_cold_bytes": int(0.5 * gb), "kv_cold_shared_bytes": int(0.5 * gb)}
    )
    assert tot_no_ssd == 1.0 and tot_no_ssd < 7, tot_no_ssd

    # Reclaim sampler: a series that grows then truncates must report the true
    # peak->last drop in GiB (2**30), and a plateau must NOT claim a shrink. The
    # negative control guards against the #742 "8.6->8.1 GiB" unit/operand error.
    gib = 2**30
    grow_shrink = _reclaim_summary(
        _reclaim_rows(iter([0, 4 * gib, 8 * gib, 5 * gib]).__next__, 4, 0)
    )
    assert grow_shrink["peak_apparent_gib"] == 8.0, grow_shrink
    assert grow_shrink["last_apparent_gib"] == 5.0, grow_shrink
    assert grow_shrink["reclaimed_off_peak_gib"] == 3.0, grow_shrink
    assert grow_shrink["shrank_after_peak"] is True, grow_shrink
    plateau = _reclaim_summary(_reclaim_rows(iter([8 * gib, 8 * gib]).__next__, 2, 0))
    assert plateau["reclaimed_off_peak_gib"] == 0.0 and plateau["shrank_after_peak"] is False
    empty = _reclaim_summary(_reclaim_rows(iter([0, 0]).__next__, 2, 0))
    assert empty["first_nonzero_gib"] == 0.0 and empty["shrank_after_peak"] is False

    print("probe_headroom_coldtail self-check ok")
    return 0


def _reclaim_rows(size_provider, n_samples: int, interval_s: float):
    """Pure sampler core: call size_provider() -> apparent bytes (or 0 if the
    spill file does not exist yet) n_samples times, timestamped. Returns a list
    of (iso_ts, apparent_bytes). The caller owns sleeping and the real fs call,
    so the self-check drives it deterministically with no disk/time dependency.

    One operand (apparent file size), one unit (GiB at 2**30 downstream) — the
    #742/#744 unit errors came from mixing apparent vs logical_ssd and decimal GB
    with binary GiB, so a reclaim claim is built only from this single series.
    """
    import datetime as _dt

    rows = []
    for _ in range(n_samples):
        rows.append((_dt.datetime.now().strftime("%H:%M:%S.%f")[:-3], int(size_provider())))
        if len(rows) < n_samples:
            time.sleep(interval_s)
    return rows


def _reclaim_summary(rows):
    """Peak -> last apparent bytes from a row series; reclaimed is the drop off
    the peak (trailing-extent truncation). Returns the GiB figures the doc cites.
    """
    vals = [b for _, b in rows]
    nz = [b for b in vals if b > 0]
    gib = lambda b: round(b / 2**30, 3)  # noqa: E731
    return {
        "samples": len(vals),
        "first_nonzero_gib": gib(nz[0]) if nz else 0.0,
        "peak_apparent_gib": gib(max(vals)),
        "last_apparent_gib": gib(vals[-1]),
        "reclaimed_off_peak_gib": gib(max(vals) - vals[-1]),
        "shrank_after_peak": bool(nz) and vals[-1] < max(vals),
    }


def cmd_reclaim_sample(a) -> int:
    """Sample the shared-prefix spill apparent size across a request release to
    capture #740 trailing-extent reclaim. Passive: drives no requests itself —
    run it alongside an arm/follower whose publish refs release mid-window, keep
    the spill until sampling ends, then stop the serve to delete. Timestamped
    rows + a GiB summary go to --out. Spans release only if --duration-s covers
    it; the printed shrink is honest 0 if the window ends on the plateau."""
    import os

    path = a.spill_path
    rows = _reclaim_rows(
        lambda: os.path.getsize(path) if os.path.exists(path) else 0,
        max(1, a.samples),
        max(0.0, a.interval_s),
    )
    summary = _reclaim_summary(rows)
    summary["spill_path"] = path
    with open(a.out, "w") as fh:
        json.dump(
            {"summary": summary, "rows": [{"t": t, "apparent_bytes": b} for t, b in rows]},
            fh,
            indent=1,
        )
    print("reclaim-sample", json.dumps(summary))
    return 0


def _dispatch(ns) -> int:
    # Every subcommand takes the parsed namespace; the bare-run default _self_check
    # accepts it optionally. Passing ns here is the whole function -- calling ns.fn()
    # raised TypeError only on the server arm (selfcheck never dispatches), so the
    # assert in _self_check pins the one-arg call shape without a server.
    return getattr(ns, "fn", _self_check)(ns)


def main() -> int:
    if "--self-check" in sys.argv[1:]:
        return _self_check()
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd")
    # Bare invocation runs the self-check: the repo-wide
    # test_main_selfchecks gate runs every hermetic script with NO args and
    # requires rc 0, so a missing subcommand must default to the self-check
    # rather than argparse-rc-2.
    a = s.add_parser("selfcheck")
    a.set_defaults(fn=_self_check)
    a = s.add_parser("arm")
    a.add_argument("--url", default="http://127.0.0.1:8000")
    a.add_argument(
        "--headroom",
        type=int,
        required=True,
        help="this arm's TILERL_DEVICE_HEADROOM_MIB (0/384/768)",
    )
    a.add_argument("--log", required=True, help="this arm's server stderr/stdout log")
    a.add_argument("--out", required=True, help="arm summary JSON path")
    a.add_argument("--fill-n", type=int, default=5)
    a.add_argument("--prompt-tokens", type=int, default=32000)
    a.add_argument("--fill-gen", type=int, default=8)
    a.add_argument("--warm-gen", type=int, default=32)
    a.add_argument(
        "--warm-reps",
        type=int,
        default=3,
        help="independent refill+warm repetitions per arm; reports the "
        "median and spread of warm-32k tok/s (not one shot)",
    )
    a.add_argument(
        "--min-good-reps",
        type=int,
        default=2,
        help="fail closed (rc13) unless at least this many reps succeed; "
        "a 3-rep arm must have >=2 good reps",
    )
    a.add_argument(
        "--expect-window",
        type=int,
        default=0,
        help="the TILERL_DRAFT_ATTN_WINDOW_TOKENS the serve was armed "
        "with (2048 for the baseline-comparable arms); cross-checked "
        "against the whole-log serve [draft-window] line, mismatch rc13",
    )
    a.add_argument("--min-cold-gb", type=float, default=7.0)
    a.add_argument(
        "--min-decode-ticks",
        type=int,
        default=5,
        help="fail closed unless the warm window logs at least this "
        "many decode ticks (baseline n was 5-12)",
    )
    a.add_argument("--timeout", type=float, default=7200.0)
    a.add_argument("--ready-trials", type=int, default=600)
    a.set_defaults(fn=cmd_arm)
    c = s.add_parser("compare")
    c.add_argument(
        "--arms",
        nargs="+",
        required=True,
        help="name=path per arm, e.g. control=/r/0.json target=/r/768.json",
    )
    c.set_defaults(fn=cmd_compare)
    r = s.add_parser("reclaim-sample")
    r.add_argument(
        "--spill-path", required=True, help="path to the shared .prefix.bin spill to sample"
    )
    r.add_argument("--out", required=True, help="timestamped rows + GiB summary JSON")
    r.add_argument("--samples", type=int, default=120)
    r.add_argument(
        "--interval-s",
        type=float,
        default=20.0,
        help="set so the window spans the publish refs' release",
    )
    r.set_defaults(fn=cmd_reclaim_sample)
    ns = ap.parse_args()
    return _dispatch(ns)


if __name__ == "__main__":
    # __main__ guard carries an assert so the scripts closure audit's set7
    # recognizes this hermetic self-check; bare invocation must exit 0.
    if {"--self-check", "selfcheck"} & set(sys.argv[1:]) or len(sys.argv) == 1:
        rc = _self_check()
        assert rc == 0
        raise SystemExit(rc)
    raise SystemExit(main())
