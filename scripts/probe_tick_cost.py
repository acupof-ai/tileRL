#!/usr/bin/env python3
"""What does a decode tick cost at k active rows? Measured, not extrapolated.

Two sessions fitted a two-term tick model and got incompatible coefficients:

    mine (19 GRPO steps, gen 6144, 1.19-4.81 active rows):  b = 8.05,  c = 4.451 ms
    25's (2 saturated points, gen 1024, k = 8 and 16):      b = 44.4,  c = 1.711 ms

`b` differs 5.5x and `c` 2.6x, in opposite directions, and both overpredict the densest
tick I measured (23.45 ms at 4.81 rows): mine says 29.46, 25's says 52.63. `b` has a
physical floor -- 21.896 GB of weights over 3.35 TB/s is 6.5 ms -- so 44.4 cannot be a
per-tick constant, but that does not make 8.05 right either.

Neither fit can settle it, for the same reason in two forms. Mine never observed more than
4.81 active rows, so `b/8 + c` -- the whole basis of the refill estimate -- is 1.7x outside
its range, and it explains 76% of its own variance (R2 0.759 on ms/tick; the 0.943 from
the wall-clock regression is weighted by `max` and so scored on the sparse long steps).
25's arm is saturated, so occupancy is identically k: two points on one line, and its two
"parameters" are that line's intercept and slope reparameterised, absorbing everything
that varies with k into `b`.

**Two arms, because one arm judges one model.** (27, 2026-09-08)

`flat` holds every row the same length, so occupancy is exactly k for the whole run and
`ms/tick = b + c*k` is exact by construction. That is 25's arm shape extended from two
points to five, so it judges 25's b=44.4 -- and it cannot judge mine, because my
coefficients came from data where the active count falls within a step and here it never
moves.

`stair` gives the k rows lengths gen, gen/2, gen/4, ..., so one run walks the active count
from k down to 1 with the depth of each survivor known. It reads `c` against active rows
directly over the full range instead of fitting over 1.19-4.81, and it is the only arm
where 8 rows of UNEQUAL context coexist -- the configuration refilling actually produces,
and the one my data never contains (its 8-row moments are all at step start, every context
short).

**Equal length in `flat` is enforced, not hoped for.** `stop_token_ids=()` is the point
here, the opposite of the tail probe where its absence was the bug: there the question was
when rows stop, here it is what a tick costs when none of them do. Lengths are asserted
equal afterwards regardless.

**Depth is held fixed across k in `flat`**, because `c` rises ~22% from depth 512 to 3072
(measured, R2 0.794 with a rows x depth interaction). A sweep that moved both would
reproduce the confound it exists to remove.

Per-tick timing is honest without an added sync: the decode path ends in `toks.tolist()`
(engine.py:1347), which blocks on the tick it just issued.

Compile gate (25, #318): a kernel compiled inside a timed region means the number includes
codegen, and the probe refuses rather than reporting it.

Run:
  scripts/pod_run.sh --wait ticks <card> -- python3 scripts/probe_tick_cost.py
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

import numpy as np  # noqa: E402

_WEIGHT_GB = 21.896
_HBM_TBS = 3.35
_MINE = (8.05, 4.451)
_ARM25 = (44.4, 1.711)


def _fit(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    p, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ p
    return p[0], p[1], 1 - (res**2).sum() / ((y - y.mean()) ** 2).sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--gen", type=int, default=1024, help="flat arm: every row runs this far")
    ap.add_argument("--stair-k", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=64, help="ticks dropped before timing")
    ap.add_argument("--out", default="/work/tick_cost.json")
    args = ap.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model, _qwen38_tokenizer
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore

    tok = _qwen38_tokenizer()
    prompt = tok.encode("Compute 17 * 23 step by step.")
    ks = [int(x) for x in args.ks.split(",")]
    kmax = max(ks + [args.stair_k])
    # stop_token_ids=() so no row ends early: equal length IS the measurement, and a row
    # that stops turns this into the tail probe it is meant to control for.
    base = SamplingParams(max_new_tokens=args.gen, temperature=1.0, seed=0)
    assert not base.stop_token_ids, "a stoppable row breaks the equal-length invariant"

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
    ctx = args.gen + len(prompt) + 64

    def engine_for(k):
        return build_engine(cfg, model, backend, num_slots=k, max_batch=k,
                            num_blocks=-(-ctx // BLOCK_TOKENS) * kmax + kmax,
                            max_total_tokens=max(ctx, 8192),
                            decode_graph=True, prefix_store=NoPrefixStore())

    print(f"arm flat: {len(ks)} readings at fixed occupancy, gen {args.gen}")
    print(f"{'k':>3} {'ticks':>6} {'ms/tick':>8} {'ms/token':>9} {'compiles':>8}")
    flat = []
    for k in ks:
        engine = engine_for(k)
        ids = [engine.submit(prompt, replace(base, seed=g)) for g in range(k)]
        for _ in range(args.warmup):
            engine.step()
        backend.synchronize()
        before = len(backend._kernels)
        # `while engine._running`, not `while engine.stats()["running"]`: stats() takes
        # the lock and builds a ~20-key dict including the prefix store's, once per tick,
        # inside the timed region. A fixed tick count is not usable either -- prefill
        # consumes an unknown number of ticks before the first decode.
        t0, ticks = time.perf_counter(), 0
        while engine._running:
            engine.step()
            ticks += 1
        backend.synchronize()
        dt = time.perf_counter() - t0
        lens = [len(engine._finished[i]) for i in ids]
        if len(set(lens)) != 1:
            raise SystemExit(
                f"k={k}: row lengths {sorted(set(lens))} are not all equal, so occupancy "
                "was not k for the whole run and this is a tail measurement")
        flat.append({"k": k, "ticks": ticks, "ms_per_tick": 1000 * dt / ticks,
                     "ms_per_token": 1000 * dt / (ticks * k),
                     "compiles": len(backend._kernels) - before, "len": lens[0]})
        print(f"{k:>3} {ticks:>6} {flat[-1]['ms_per_tick']:>8.3f} "
              f"{flat[-1]['ms_per_token']:>9.3f} {flat[-1]['compiles']:>8}")

    # The staircase: lengths gen, gen/2, gen/4, ... so the active count walks k -> 1 and
    # every intermediate tick has survivors of UNEQUAL depth -- refill's actual shape.
    # Halving, with no floor: a floor makes the shortest rows tie, and tied rows retire on
    # the same tick, so the active count skips values. At gen 1024 a floor of 32 gave
    # lengths ending 64,32,32,32 -- the walk went 4 -> 1 and never read 3 or 2 rows.
    K = args.stair_k
    lengths = [args.gen >> i for i in range(K)]
    if len(set(lengths)) != K:
        raise SystemExit(f"--gen {args.gen} halved {K} times is not {K} distinct lengths "
                         f"({lengths}); the active count would skip values. Need gen >= "
                         f"{1 << (K - 1)}.")
    print(f"\narm stair: k={K}, lengths {lengths}")
    engine = engine_for(K)
    for g, n in enumerate(lengths):
        engine.submit(prompt, replace(base, max_new_tokens=n, seed=100 + g))
    for _ in range(args.warmup):
        engine.step()
    backend.synchronize()
    before = len(backend._kernels)
    per_tick = []
    while engine.stats()["running"]:
        n = engine.stats()["running"]
        depth = np.mean([r.seq_len for r in engine._running])
        t = time.perf_counter()
        engine.step()
        per_tick.append((n, float(depth), 1000 * (time.perf_counter() - t)))
    stair_compiles = len(backend._kernels) - before

    print(f"{'rows':>5} {'ticks':>6} {'mean depth':>10} {'ms/tick':>8}")
    stair = []
    for n in sorted({p[0] for p in per_tick}, reverse=True):
        grp = [p for p in per_tick if p[0] == n]
        stair.append({"rows": n, "ticks": len(grp),
                      "depth": sum(p[1] for p in grp) / len(grp),
                      "ms_per_tick": sum(p[2] for p in grp) / len(grp)})
        print(f"{n:>5} {len(grp):>6} {stair[-1]['depth']:>10.0f} "
              f"{stair[-1]['ms_per_tick']:>8.3f}")

    bad = [r["k"] for r in flat if r["compiles"]] + (["stair"] if stair_compiles else [])
    if bad:
        raise SystemExit(f"kernels compiled inside the timed region at {bad}: those means "
                         "include codegen. Raise --warmup and rerun.")

    floor = 1000 * _WEIGHT_GB / (_HBM_TBS * 1000)
    print(f"\n{'arm':>6} {'b (ms)':>8} {'c (ms)':>8} {'R2':>7}  {'b / floor':>9}")
    out = {}
    for name, xs, ys in (("flat", [r["k"] for r in flat], [r["ms_per_tick"] for r in flat]),
                         ("stair", [r["rows"] for r in stair],
                          [r["ms_per_tick"] for r in stair])):
        b, c, r2 = _fit(np.array(xs, float), np.array(ys))
        out[name] = {"b_ms": b, "c_ms": c, "r2": r2}
        print(f"{name:>6} {b:>8.2f} {c:>8.3f} {r2:>7.4f}  {b / floor:>8.2f}x")
    print(f"       mine {_MINE[0]:>8.2f} {_MINE[1]:>8.3f}      --  {_MINE[0] / floor:>8.2f}x")
    print(f"       25's {_ARM25[0]:>8.2f} {_ARM25[1]:>8.3f}      --  {_ARM25[0] / floor:>8.2f}x")
    print(f"\nweight-stream floor {_WEIGHT_GB} GB / {_HBM_TBS} TB/s = {floor:.2f} ms/tick")
    if out["flat"]["b_ms"] < floor:
        raise SystemExit(f"flat b = {out['flat']['b_ms']:.2f} ms is below the {floor:.2f} ms "
                         "weight-stream floor, so the timed region is not reading the "
                         "weights once per tick")

    print(f"\n{'k':>3} {'flat':>8} {'stair':>8} {'mine':>8} {'arm25':>8}")
    smap = {r["rows"]: r["ms_per_tick"] for r in stair}
    for r in flat:
        k = r["k"]
        s = f"{smap[k]:8.2f}" if k in smap else f"{'--':>8}"
        print(f"{k:>3} {r['ms_per_tick']:8.2f} {s} {_MINE[0] + _MINE[1] * k:8.2f} "
              f"{_ARM25[0] + _ARM25[1] * k:8.2f}")
    print("\nflat and stair differing at the same k is the finding, not the noise: same "
          "occupancy, different context spread.")

    with open(args.out, "w") as f:
        json.dump({"gen": args.gen, "fits": out, "flat": flat, "stair": stair,
                   "stair_lengths": lengths}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    # The staircase's own arithmetic, before it costs a card: the active count must pass
    # through every value from K down to 1. A floored halving ties the short rows, they
    # retire on the same tick, and the walk skips values -- caught here, not on the card.
    _g, _K = 1024, 8
    _lens = [_g >> i for i in range(_K)]
    assert sorted({sum(1 for n in _lens if n > t) for t in range(max(_lens))}) == list(
        range(1, _K + 1)), _lens
    _floored = [max(_g >> i, 32) for i in range(_K)]
    assert sorted({sum(1 for n in _floored if n > t) for t in range(max(_floored))}) != list(
        range(1, _K + 1)), "the floored variant is the bug this check exists for"
    _b, _c, _r2 = _fit(np.array([1.0, 2, 4, 8]), np.array([12.5, 17.0, 26.0, 44.0]))
    assert abs(_b - 8.0) < 0.6 and abs(_c - 4.5) < 0.2, (_b, _c, _r2)
    raise SystemExit(main())
