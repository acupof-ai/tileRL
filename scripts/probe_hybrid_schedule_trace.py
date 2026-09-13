"""Hybrid scheduler trace: is the 6:1 the scheduler or the measurement?

Two phases, to separate JIT from steady state and capacity from fairness:

1. WARMUP: one short dense and one short sparse request are drained first so
   both graph/JIT paths are hot; timed measurements start after.
2. TIMED FILL: one long sparse prefill starts; <= (slots-1) short dense
   requests are submitted STAGGERED mid-prefill, each guaranteed a free slot,
   so there is no capacity queue by construction. The question is then exactly:
   does a short request with a free slot, arriving during a long fill, get a
   normal TTFT and decode speed?

Per tick it records mode, synced wall ms (with tick index, so a first-tick JIT
outlier is distinguishable from a steady-state stall), planned/running/waiting
rows by mode, free slots/blocks, and why the waiting head did not admit.
Per short: submit tick, queue_ms (submit->admit), wait2dense, TTFT, decode.

--dump-out saves the long greedy tokens for a 96-vs-192 exactness check.

Run (V100):
  TILERL_TARGET=cuda /usr/bin/python3 scripts/probe_hybrid_schedule_trace.py \
      --source <ckpt> --draft <mtp> [--cap 96] [--dump-out /tmp/out96.json]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--cap", type=int, default=192)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--long-tokens", type=int, default=32768)
    # Shorts arrive STAGGERED MID-PREFILL, never more than the free slots (one
    # slot is held by the long request). This isolates "does a short request with
    # a free slot get normal latency during a long fill" from capacity queueing.
    ap.add_argument("--n-shorts", type=int, default=3,
                    help="must be <= slots-1 so every short has a free slot")
    ap.add_argument("--short-stagger-ticks", type=int, default=12,
                    help="ticks between mid-prefill short submissions")
    ap.add_argument("--warmup-new-tokens", type=int, default=2048)
    ap.add_argument("--short-new-tokens", type=int, default=64)
    ap.add_argument("--short-max-new", type=int, default=40)
    ap.add_argument("--dump-out", default="")
    args = ap.parse_args()
    assert args.n_shorts <= args.slots - 1, (
        f"n-shorts {args.n_shorts} exceeds free slots {args.slots - 1}; the "
        "point is no capacity queueing")

    import torch
    from tilerl_kernels.backend import get_backend

    from tilerl import cli
    from tilerl.cli import _build_model
    from tilerl.engine import SamplingParams, build_engine

    cli._QWEN38_SOURCE = args.source
    be = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    from tilerl.spec import load_draft

    e = build_engine(
        cfg, model, be, num_slots=args.slots, max_batch=args.slots,
        max_total_tokens=args.ctx, sparse_k=128, scorer="bounds",
        kv_cold_bytes=1 << 33, sparse_min_tokens=8192,
        sparse_prefill_tokens=args.cap,
        decode_graph=True, draft=load_draft(model, args.draft), spec_depth=1)

    def drain(e, rids, limit=4000):
        for _ in range(limit):
            e.step()
            p = e.poll()
            if all(r in p for r in rids):
                return
        raise RuntimeError("warmup drain timed out")

    print("warmup: one short dense and one short sparse request to pay JIT/capture "
          "in BOTH modes before timing...", flush=True)
    rng = np.random.default_rng(3)
    warm_ids = rng.integers(3, 300, args.warmup_new_tokens).astype(np.int64)
    # dense warmup (under N) and sparse warmup (over N) drain sequentially
    wd = e.submit(warm_ids[:64], SamplingParams(temperature=0.0, max_new_tokens=8, seed=900))
    drain(e, [wd])
    ws = e.submit(warm_ids, SamplingParams(temperature=0.0, max_new_tokens=4, seed=901))
    drain(e, [ws])
    print("warmup done; starting the timed long sparse fill", flush=True)

    long_rid = e.submit(
        rng.integers(3, 300, args.long_tokens).astype(np.int64),
        SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))

    # Stagger the short requests mid-prefill, one every --short-stagger-ticks;
    # n_shorts <= slots-1 so each always has a free slot (no capacity queue).
    short_rids = []
    next_submit_at_tick = 4  # let the long fill be clearly underway
    submit_tick = {}

    rids = [long_rid]
    submit_ts = {long_rid: time.perf_counter()}
    admit_ts: dict[int, float] = {}
    first_ts: dict[int, float] = {}
    finish_ts: dict[int, float] = {}
    out_tokens: dict[int, list] = {long_rid: []}
    ticks = []

    # Wrap the two seams the real step calls; keep step itself in charge of
    # sample/finalize/finish so requests actually complete.
    orig_plan = e._build_plan
    orig_fwd = e._run_forward

    admit_fail = {"slots": 0, "blocks": 0, "reclaim": 0}

    def plan_wrap():
        decodes, prefills, chunks = orig_plan()
        rows = decodes + prefills
        # WHY did the waiting queue head not admit? mirror _admit's gates.
        if e._waiting:
            w = e._waiting[0]
            if e._states.free_slots < 1:
                admit_fail["slots"] += 1
            else:
                total = (len(w.tokens) + 15) // 16
                need = (0 if w.sparse_on else total)
                if not w.sparse_on and e._sparse is not None:
                    need += e._sparse_hot_headroom()
                if e._kv.free_blocks < need:
                    if (e._kv.free_blocks + e._prefix.reclaimable_blocks()) < need:
                        admit_fail["reclaim"] += 1
                    else:
                        admit_fail["blocks"] += 1
        meta = {
            "tick": len(ticks),
            "mode": ("sparse" if rows and rows[0].sparse_on else
                     "dense" if rows else "idle"),
            "plan_d": sum(1 for r in rows if not r.sparse_on),
            "plan_s": len(rows) - sum(1 for r in rows if not r.sparse_on),
            "run_d": sum(1 for r in e._running if not r.sparse_on),
            "run_s": sum(1 for r in e._running if r.sparse_on),
            "wait_d": sum(1 for r in e._waiting if not r.sparse_on),
            "wait_s": sum(1 for r in e._waiting if r.sparse_on),
            "free_slots": e._states.free_slots,
            "free_blocks": int(e._kv.free_blocks),
        }
        ticks.append(meta)
        return decodes, prefills, chunks

    def fwd_wrap(decodes, prefills, chunks):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rv = orig_fwd(decodes, prefills, chunks)
        torch.cuda.synchronize()
        ticks[-1]["ms"] = (time.perf_counter() - t0) * 1000
        # what did THIS tick's dense rows actually do?
        ticks[-1]["d_dec"] = len(decodes)
        ticks[-1]["d_pf"] = len(prefills)
        ticks[-1]["d_nout"] = sum(len(r.output) for r in decodes)
        return rv

    e._build_plan = plan_wrap
    e._run_forward = fwd_wrap

    # peek() (not poll) after each tick: it returns the growing token list without
    # consuming it, so first-token time is the first tick the list is non-empty and
    # finish time is when it reaches max_new -- poll-once collapsed all 40 tokens
    # to the finish tick and made decode_ms 0.
    target = {long_rid: 2}
    first_dense_ts: dict[int, float] = {}
    tick_idx = 0
    submitted = 0
    failed: dict[int, tuple[str, str]] = {}
    phase_seen: dict[int, set] = {}

    for _ in range(20000):
        # stagger a short request mid-prefill on its schedule tick, if free slot
        if (submitted < args.n_shorts
                and tick_idx == next_submit_at_tick + submitted * args.short_stagger_ticks):
            ids = rng.integers(3, 300, args.short_new_tokens).astype(np.int64)
            rid = e.submit(ids, SamplingParams(
                temperature=0.0, max_new_tokens=args.short_max_new, seed=100 + submitted))
            short_rids.append(rid)
            rids.append(rid)
            submit_ts[rid] = time.perf_counter()
            submit_tick[rid] = tick_idx
            out_tokens[rid] = []
            target[rid] = args.short_max_new
            submitted += 1
        e.step()
        tick_idx += 1
        now = time.perf_counter()
        for r in e._running:
            if r.req_id not in admit_ts and r.state_slot is not None:
                admit_ts[r.req_id] = now
            if not r.sparse_on and r.req_id not in first_dense_ts:
                first_dense_ts[r.req_id] = now
            phase_seen.setdefault(r.req_id, set()).add(r.phase)
        # A per-request failure removes it from running (run_d -> 0) and peek
        # never returns it; surface it instead of reporting a silent UNFINISHED.
        for fr, (reason, msg) in list(e._failed.items()):
            failed.setdefault(fr, (reason, msg))
        for rid in rids:
            cur = e.peek(rid)
            if cur is None:
                continue
            if rid not in first_ts and cur:
                first_ts[rid] = now
            out_tokens[rid] = list(cur)
            if len(cur) >= target[rid]:
                finish_ts.setdefault(rid, now)
        if failed:
            break
        if submitted >= args.n_shorts and long_rid in finish_ts and all(
                r in finish_ts for r in short_rids):
            break
    e.poll()

    e.shutdown()

    print(f"\n== short requests (cap={args.cap}, n={len(short_rids)}) ==")
    print(f"{'rid':>4s} {'queue_ms':>9s} {'wait2dense':>10s} {'ttft_ms':>9s} "
          f"{'decode_ms':>10s} {'toks':>4s} {'dec_tok/s':>9s}")
    rates = []
    for r in short_rids:
        if r in failed:
            reason, msg = failed[r]
            print(f"{r:4d} FAILED [{reason}] {msg[:80]}")
            continue
        if r not in finish_ts:
            ph = sorted(phase_seen.get(r, ()))
            in_wait = any(w.req_id == r for w in e._waiting)
            print(f"{r:4d} UNFINISHED phases_seen={ph} still_waiting={in_wait}")
            continue
        q = (admit_ts.get(r, submit_ts[r]) - submit_ts[r]) * 1000
        wd = (first_dense_ts.get(r, submit_ts[r]) - submit_ts[r]) * 1000
        ttft = (first_ts[r] - submit_ts[r]) * 1000
        dec_ms = (finish_ts[r] - first_ts[r]) * 1000
        n = len(out_tokens[r])
        rate = (n - 1) / (dec_ms / 1000) if dec_ms > 0 else float("inf")
        rates.append(rate)
        print(f"{r:4d} {q:9.0f} {wd:10.0f} {ttft:9.0f} {dec_ms:10.0f} "
              f"{n:4d} {rate:9.1f}")
    if rates:
        s = sorted(rates)
        print(f"decode tok/s: min {s[0]:.1f}  median {s[len(s)//2]:.1f}  "
              f"max {s[-1]:.1f}  mean {sum(s)/len(s):.1f}")
    print(f"\nadmit head-of-line blocked (tick counts): slots_full={admit_fail['slots']} "
          f"blocks+headroom={admit_fail['blocks']} blocks+reclaim={admit_fail['reclaim']}")

    sp = [t for t in ticks if t["mode"] == "sparse"]
    de = [t for t in ticks if t["mode"] == "dense"]
    starved = sum(1 for t in sp if t["run_d"] > 0)
    no_dense = sum(1 for t in sp if t["run_d"] == 0)
    print(f"\n== ticks == sparse {len(sp)} ({sum(t['ms'] for t in sp):.0f} ms, "
          f"med {sorted(t['ms'] for t in sp)[len(sp)//2]:.0f})  "
          f"dense {len(de)} ({sum(t['ms'] for t in de):.0f} ms)")
    print(f"sparse ticks WITH runnable dense row (scheduler starvation): {starved}")
    print(f"sparse ticks with NO runnable dense row (queue/arrival gap):  {no_dense}")

    # Steady-state dense tick cost DURING the long fill, with a dense row runnable:
    # the JIT/capture question. Print index+ms so first-tick outliers are visible.
    fill_dense = [t for t in ticks if t["mode"] == "dense" and t["run_d"] > 0]
    if fill_dense:
        ms = sorted(t["ms"] for t in fill_dense)
        print(f"dense ticks during fill with dense runnable (n={len(ms)}): "
              f"min {ms[0]:.1f} med {ms[len(ms)//2]:.1f} max {ms[-1]:.1f} ms")
        # of those, how many actually carried a decode row vs only a prefill?
        dec = [t for t in fill_dense if t["d_dec"] > 0]
        pf = [t for t in fill_dense if t["d_pf"] > 0 and t["d_dec"] == 0]
        print(f"  carried decode rows: {len(dec)}; prefill-only: {len(pf)}; "
              f"sum nout seen on decode ticks: {sum(t['d_nout'] for t in dec)}")
        slow = [t for t in fill_dense if t["ms"] > 500]
        print(f"  dense ticks >500 ms (steady-state stall, not JIT): {len(slow)}")
        for t in slow[:10]:
            print(f"    tick {t['tick']}: {t['ms']:.0f} ms dec/pf {t['d_dec']}/{t['d_pf']} "
                  f"runD/runS {t['run_d']}/{t['run_s']} waitD {t['wait_d']}")

    gaps, cur, started = [], 0, False
    for t in ticks:
        if t["mode"] == "sparse":
            if started:
                gaps.append(cur)
            cur, started = 0, True
        elif started and t["run_d"] > 0:
            cur += 1
    if started:
        gaps.append(cur)
    if gaps:
        print(f"dense-runnable ticks per sparse gap (n={len(gaps)}): "
              f"min {min(gaps)} median {sorted(gaps)[len(gaps)//2]} max {max(gaps)}")

    print("\nfirst 20 ticks: mode ms planD/planS runD/runS waitD/waitS "
          "freeSlots/freeBlocks")
    for t in ticks[:20]:
        print(f"  {t['mode']:6s} {t.get('ms',0):7.0f}  {t['plan_d']}/{t['plan_s']}  "
              f"{t['run_d']}/{t['run_s']}  {t['wait_d']}/{t['wait_s']}  "
              f"{t['free_slots']}/{t['free_blocks']}")

    if args.dump_out:
        with open(args.dump_out, "w") as f:
            json.dump({"long_tokens": out_tokens[long_rid], "cap": args.cap}, f)
        print(f"\nwrote long greedy output to {args.dump_out}; diff the 192/96 files.")


if __name__ == "__main__":
    main()
