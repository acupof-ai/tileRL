"""Hybrid scheduler trace: is the 6:1 the scheduler or the measurement?

The wall-time scheduler SHOULD give ~28 dense ticks per sparse tick
(1023 ms / 37 ms). The V100 run showed 5.95:1 and ~9.7 tok/s for short
requests. A longer sparse tick earns MORE dense ticks under wall-time
accounting, so tick-duration spread cannot explain it. This wraps the real
step() and records, per tick, whether a dense row was even runnable when the
tick went to sparse:

  mode, synced wall ms,
  rows this tick ran (dense/sparse),
  admitted running rows by mode (dense/sparse),
  waiting rows by mode (dense/sparse)

Diagnosis:
- sparse tick with running_dense > 0  -> the scheduler starved dense.
- sparse tick with running_dense == 0 -> no dense row existed; a short-request
  mean that includes that wait is measuring queue/TTFT, not decode speed.

It splits every short request's wall time into queue (submit -> admit),
TTFT (submit -> first token), and decode (first -> last token) tok/s, and
prints dense-ticks-per-sparse-gap counted only while a dense row was runnable.

--dump-out saves the long request's greedy token ids for 96-vs-192 exactness;
run twice (--cap 192 and --cap 96) and diff the two files.

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
    ap.add_argument("--n-shorts", type=int, default=19)
    ap.add_argument("--short-new-tokens", type=int, default=64)
    ap.add_argument("--short-max-new", type=int, default=40)
    ap.add_argument("--dump-out", default="")
    args = ap.parse_args()

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

    rng = np.random.default_rng(3)
    long_rid = e.submit(
        rng.integers(3, 300, args.long_tokens).astype(np.int64),
        SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    short_rids = [
        e.submit(rng.integers(3, 300, args.short_new_tokens).astype(np.int64),
                 SamplingParams(temperature=0.0, max_new_tokens=args.short_max_new,
                                seed=i))
        for i in range(args.n_shorts)]

    rids = [long_rid, *short_rids]
    submit_ts = {r: time.perf_counter() for r in rids}
    admit_ts: dict[int, float] = {}
    first_ts: dict[int, float] = {}
    finish_ts: dict[int, float] = {}
    out_tokens: dict[int, list] = {r: [] for r in rids}
    ticks = []

    # Wrap the two seams the real step calls; keep step itself in charge of
    # sample/finalize/finish so requests actually complete.
    orig_plan = e._build_plan
    orig_fwd = e._run_forward

    def plan_wrap():
        decodes, prefills, chunks = orig_plan()
        rows = decodes + prefills
        meta = {
            "mode": ("sparse" if rows and rows[0].sparse_on else
                     "dense" if rows else "idle"),
            "plan_d": sum(1 for r in rows if not r.sparse_on),
            "plan_s": len(rows) - sum(1 for r in rows if not r.sparse_on),
            "run_d": sum(1 for r in e._running if not r.sparse_on),
            "run_s": sum(1 for r in e._running if r.sparse_on),
            "wait_d": sum(1 for r in e._waiting if not r.sparse_on),
            "wait_s": sum(1 for r in e._waiting if r.sparse_on),
        }
        ticks.append(meta)
        return decodes, prefills, chunks

    def fwd_wrap(decodes, prefills, chunks):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rv = orig_fwd(decodes, prefills, chunks)
        torch.cuda.synchronize()
        ticks[-1]["ms"] = (time.perf_counter() - t0) * 1000
        return rv

    e._build_plan = plan_wrap
    e._run_forward = fwd_wrap

    for _ in range(20000):
        e.step()
        now = time.perf_counter()
        for r in e._running:
            if r.req_id not in admit_ts and r.state_slot is not None:
                admit_ts[r.req_id] = now
        polled = e.poll()
        for rid, toks in polled.items():
            if rid not in first_ts:
                first_ts[rid] = now
            out_tokens[rid].extend(toks)
            finish_ts[rid] = now
        if all(r in finish_ts for r in rids):
            break

    e.shutdown()

    print(f"\n== short requests (cap={args.cap}, n={len(short_rids)}) ==")
    print(f"{'rid':>4s} {'queue_ms':>9s} {'ttft_ms':>9s} {'decode_ms':>10s} "
          f"{'toks':>4s} {'dec_tok/s':>9s}")
    rates = []
    for r in short_rids:
        if r not in finish_ts:
            print(f"{r:4d} UNFINISHED")
            continue
        q = (admit_ts.get(r, submit_ts[r]) - submit_ts[r]) * 1000
        ttft = (first_ts[r] - submit_ts[r]) * 1000
        dec_ms = (finish_ts[r] - first_ts[r]) * 1000
        n = len(out_tokens[r])
        rate = (n - 1) / (dec_ms / 1000) if dec_ms > 0 else float("inf")
        rates.append(rate)
        print(f"{r:4d} {q:9.0f} {ttft:9.0f} {dec_ms:10.0f} {n:4d} {rate:9.1f}")
    if rates:
        s = sorted(rates)
        print(f"decode tok/s: min {s[0]:.1f}  median {s[len(s)//2]:.1f}  "
              f"max {s[-1]:.1f}  mean {sum(s)/len(s):.1f}")

    sp = [t for t in ticks if t["mode"] == "sparse"]
    de = [t for t in ticks if t["mode"] == "dense"]
    starved = sum(1 for t in sp if t["run_d"] > 0)
    no_dense = sum(1 for t in sp if t["run_d"] == 0)
    print(f"\n== ticks == sparse {len(sp)} ({sum(t['ms'] for t in sp):.0f} ms, "
          f"med {sorted(t['ms'] for t in sp)[len(sp)//2]:.0f})  "
          f"dense {len(de)} ({sum(t['ms'] for t in de):.0f} ms)")
    print(f"sparse ticks WITH runnable dense row (scheduler starvation): {starved}")
    print(f"sparse ticks with NO runnable dense row (queue/arrival gap):  {no_dense}")

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

    print("\nfirst 20 ticks: mode ms planD/planS runD/runS waitD/waitS")
    for t in ticks[:20]:
        print(f"  {t['mode']:6s} {t.get('ms',0):7.0f}  {t['plan_d']}/{t['plan_s']}  "
              f"{t['run_d']}/{t['run_s']}  {t['wait_d']}/{t['wait_s']}")

    if args.dump_out:
        with open(args.dump_out, "w") as f:
            json.dump({"long_tokens": out_tokens[long_rid], "cap": args.cap}, f)
        print(f"\nwrote long greedy output to {args.dump_out}; diff the 192/96 files.")


if __name__ == "__main__":
    main()
