#!/usr/bin/env python3
"""What is inside the rollout bucket? 63.156 s of an 85.617 s GRPO step, undecomposed.

`timings["rollout_secs"]` (train.py:444) is one `perf_counter` around `_drain`, which
ticks `engine.step()` until every rollout finishes. It is 73.8% of the step and nobody
has looked inside it. The last time a *name* did the reasoning here, `backward_secs`
turned out to be 19.7% forward.

**The bound, computed before the run so the result cannot be read into.** group 8 x
gen 1024 = 8192 tokens in 63.156 s is **130 tok/s aggregate**. The weights are 24.44 GB
(22.76 GiB resident, measured 2026-09-08 on the NVFP4 27B), and a decode tick streams
them ONCE for the whole batch -- that is what batching buys. At H20's ~4 TB/s nameplate
a forward is 6.11 ms, so the weight-stream ceiling is **1309 tok/s at B=8**, and the
measured rate is **0.099 of it**. Even at a realistic 3.35 TB/s achieved it is 0.119.

An earlier version of this docstring divided where it should have multiplied and put the
ceiling at 176 tok/s, making 130 look like 0.74 of the bound -- near-optimal. It is an
order of magnitude below. **The B=1 ceiling is 164 tok/s, which is what 176 was
approximating: a per-row figure compared against an aggregate measurement.**

So the outcomes are not "already optimal" versus "waste":

  decode dominates -> the decode itself runs at a tenth of its bandwidth ceiling, and
                      the question moves inside the kernel (occupancy, KV traffic, the
                      GDN state) rather than to spec decode
  decode is a part -> the rest is overhead outside the forward, and it is large

The ceiling is a ceiling, not a target: it ignores KV reads, the GDN state, sampling and
every launch. A measured rate ABOVE it would mean the weights are not re-read per tick
(a cache effect, or a wrong weight figure), not that the kernel beat physics.

Phases, taken from engine.step() itself. The call tree is NOT flat -- `_run_forward`
(engine.py:945) calls `_sample_commit` at :1041 and `_finish_prefills` at :1047 -- so
each timer records inclusive time and the leaves are derived:

  _build_plan          leaf
  _run_forward         parent; forward_excl = _run_forward - its children
    _sample_commit     parent; sample_excl = _sample_commit - sample_batch
      sample_batch     leaf, the backend call
    _finish_prefills   leaf

Reconstruction is asserted on the LEAVES, and the remainder must be non-negative.
A first version of this probe defined the remainder as `wall - sum(phases)` and then
asserted the phases summed to `wall` -- an identity that cannot fail. It printed
`unattributed = -27.2 s`, a negative remainder, and passed: the loudest possible
alarm turned into a data point by a gate that was checking its own definition.

**Wall clock on an async device attributes to the first blocking call, not to the work.**
Fixing the nesting is necessary and not sufficient. CUDA launches are asynchronous, and
there is NO synchronize, `.item()`, `.cpu()` or `.tolist()` anywhere on the decode path
between `self._model.forward` (engine.py:1025) and `toks.tolist()` (engine.py:1347) --
checked across engine.py, model.py, backend.py and reference.py. So `_run_forward`'s
timer stops when the last kernel is QUEUED, and `_sample_commit`'s timer absorbs the
whole forward draining at the first host read.

The first run's own numbers prove this without a second run: exclusive forward came out
at 3.07 ms/tick, and streaming 24.44 GB of weights at 4 TB/s takes 6.11 ms. **A forward
cannot finish in half the time it takes to read the weights it multiplies by.** Below a
hardware floor means the timer is not measuring that work.

So `--sync-forward` is the control: it makes the forward blocking and nothing else
changes. Two outcomes, and they cannot both be true:

  sampling really is 27 s -> forward stays ~3 ms, sample stays ~27 ms/tick
  the timer misattributed  -> forward becomes ~26-30 ms, sample collapses to ~microseconds

Run both arms. A breakdown from the un-synced arm alone is not evidence.

**A warm step is one with zero compiles, not one that is not the first.** This pooled
`rows[1:]` and called it warm; that is an assumption about where JIT lands, and 25
measured an arm on 2026-09-08 whose step 1 compiled 40 kernels -- the convention held by
luck there, and the summary would have looked identical if it had not. `backend._kernels`
is keyed on `(name, args, kw)` (backend.py:412), so a compile is exactly one new entry,
and the count is now a gate: any compile after step 0 refuses the run instead of pooling.

Which widths can compile mid-run, read off the two call sites rather than assumed: the
GEMV branch passes M as a factory argument (`self._kernel(gk, M)`, backend.py:696) so
every distinct M is its own entry, while the mma8 branch pads to `_MX`=8 and takes no M
(backend.py:779) so M=2..8 share one. With `_MGEMV`=3 a group=8 rollout draining to empty
therefore compiles only at M=3 and M=2 -- not at each of 7, 6, 5, 4. Step 0 runs that same
drain, so it normally exhausts them; the case that breaks is a step whose rows all finish
together, skipping a width that a later step reaches first.

ONE ARM PER PROCESS. The 2.6x rollout gap was build order inside one process -- two
engines built in one process, and the second one degraded. This script builds one.

Run (card 6, via pod_run so the claim and the reaping are handled):
  scripts/pod_run.sh --wait rollout 6 -- python3 scripts/probe_rollout_breakdown.py \
      --group 8 --gen 1024 --steps 3
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

import numpy as np  # noqa: E402

#: measured 2026-09-08 on the live card: 24436981888 bytes resident after load. Fallback
#: for the ceiling denominator; with --checkpoint the bytes come from
#: model.checkpoint_weight_faces (plan's weights row), which equals this on the 27B.
#: Bytes are the served DEVICE faces (incl. embed/norm), not the ~12.6 GiB on disk.
_WEIGHT_GIB_FALLBACK = 22.76
_WEIGHT_BYTES_MEASURED = 24_436_981_888
#: H20 HBM3 nameplate. A nameplate, so the ceiling it gives is optimistic by design.
_HBM_TBS = 4.0


def _weight_bytes(cfg, checkpoint: str | None) -> int:
    """Served weight bytes across every device face (incl. embed/norm), the ceiling's
    denominator. From checkpoint headers (the plan weights row) when a path is given,
    else the measured constant."""
    if not checkpoint:
        return int(_WEIGHT_GIB_FALLBACK * 1024**3)
    from tilerl.model import checkpoint_weight_faces
    from tilerl.precision import nbytes
    total = sum(nbytes(fmt, shape)
                for shape, fmt in checkpoint_weight_faces(cfg, checkpoint).values())
    assert total == _WEIGHT_BYTES_MEASURED, (
        f"served weight bytes {total} != measured {_WEIGHT_BYTES_MEASURED}; the checkpoint "
        "face map changed — re-derive the ceiling, do not reuse the constant")
    return total


def _ceiling_tok_s(batch: int, weight_bytes: int) -> float:
    """Aggregate tok/s if the only cost were streaming the weights once per TICK.

    Once per tick, not once per token: a decode tick reads the weight set one time and
    serves every row in the batch from it, which is the whole point of batching. Dividing
    by the batch instead of multiplying gives the B=1 figure and understates the ceiling
    8-fold at B=8.
    """
    gb = weight_bytes / 1e9
    return batch / (gb / (_HBM_TBS * 1000))


#: child -> parent, so inclusive timers can be turned into exclusive ones. Read off the
#: call sites, not guessed: engine.py:1041 and :1047 sit inside _run_forward's body, and
#: _sample_commit calls _sample_batch which calls the backend.
_PARENT = {
    "_sample_commit": "_run_forward",
    "_finish_prefills": "_run_forward",
    "sample_batch": "_sample_commit",
}


class _Phases:
    """Inclusive wall clock per phase, by wrapping the engine's and backend's methods.

    Wrapping, not sampling: a profiler's attribution would need its own validation, and
    the reconstruction assert cannot be written against sampled totals.
    """

    def __init__(self, engine):
        self.t = defaultdict(float)
        self.n = defaultdict(int)
        self._restore = []
        for name in ("_build_plan", "_run_forward", "_sample_commit", "_finish_prefills"):
            if hasattr(engine, name):
                self._patch(engine, name, name)
        # The sampler itself, one level down: 89.5% of the rollout landed in
        # _sample_commit and the question is what inside it.
        if hasattr(engine, "_backend") and hasattr(engine._backend, "sample_batch"):
            self._patch(engine._backend, "sample_batch", "sample_batch")

    def _patch(self, obj, attr, label):
        fn = getattr(obj, attr)
        self._restore.append((obj, attr, fn))

        def inner(*a, **kw):
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                self.t[label] += time.perf_counter() - t
                self.n[label] += 1
        setattr(obj, attr, inner)

    def exclusive(self) -> dict:
        """Inclusive timers minus each node's direct children."""
        out = dict(self.t)
        for child, parent in _PARENT.items():
            if child in self.t and parent in out:
                out[parent] -= self.t[child]
        return out

    def restore(self):
        for obj, attr, fn in self._restore:
            setattr(obj, attr, fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--sync-forward", action="store_true",
                    help="block after the model forward so its own timer holds its own "
                         "time; without this the async launch bills the forward to "
                         "whichever phase first reads a tensor on the host")
    ap.add_argument("--out", default="/work/rollout_breakdown.json")
    ap.add_argument("--checkpoint", default=None,
                    help="27B checkpoint dir: derive the weight denominator from its "
                         "served device faces (plan weights row); without it the measured "
                         "24.44 GB constant is used")
    args = ap.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
    from tilerl.train import _drain

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
    weight_bytes = _weight_bytes(cfg, args.checkpoint)
    weight_gib = weight_bytes / 1024**3
    ctx = args.gen + 512
    # The shipped training shape (cli.py:570-574), not a probe-specific one: a different
    # engine config is a different measurement, and this one is meant to explain a
    # number the shipped path produced.
    engine = build_engine(cfg, model, backend, num_slots=8, max_batch=8,
                          num_blocks=-(-ctx // BLOCK_TOKENS) * 8 + 8,
                          max_total_tokens=max(ctx, 8192),
                          decode_graph=True, prefix_store=NoPrefixStore())

    if args.sync_forward:
        import torch
        inner = engine._model.forward

        def _forward_sync(*a, **kw):
            out = inner(*a, **kw)
            # Not during capture: a synchronize inside a captured region raises
            # cudaErrorStreamCaptureInvalidated, the capture is abandoned, and the whole
            # run silently falls back to eager -- which changes the thing being measured
            # rather than only its attribution.
            if not torch.cuda.is_current_stream_capturing():
                torch.cuda.synchronize()
            return out
        engine._model.forward = _forward_sync

    prompt = np.random.default_rng(0).integers(1, 300, size=256).tolist()
    rows = []
    for step in range(args.steps):
        ph = _Phases(engine)
        k0 = len(backend._kernels)
        t0 = time.perf_counter()
        ids = [engine.submit(prompt, SamplingParams(max_new_tokens=args.gen,
                                                    seed=step * args.group + g))
               for g in range(args.group)]
        done = _drain(engine, ids, "probe rollout")
        wall = time.perf_counter() - t0
        ph.restore()

        ntok = sum(len(done[i]) for i in ids)
        acc = ph.exclusive()
        # `poll` and the loop's own bookkeeping are whatever the leaves do not explain.
        # Asserted non-negative below: a negative remainder means the timers double-count,
        # which is a broken instrument rather than a small phase.
        acc["unattributed"] = wall - sum(acc.values())
        assert acc["unattributed"] >= -1e-6, (
            f"negative remainder {acc['unattributed']:.3f} s: inclusive timers were not "
            f"made exclusive, so _PARENT is missing an edge. inclusive={dict(ph.t)}"
        )
        rows.append({"step": step, "wall_s": wall, "tokens": ntok, "tok_s": ntok / wall,
                     "phases": acc, "calls": dict(ph.n), "sync_forward": args.sync_forward,
                     "compiles": len(backend._kernels) - k0})
        print(f"step {step}: {wall:8.3f} s  {ntok:6d} tok  {ntok / wall:7.2f} tok/s")
        for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<18} {v:8.3f} s  {100 * v / wall:5.1f}%")

    # Step 0 carries the capture and the JIT; the pooled row drops it, as #109's arms did.
    # Dropping the first step is an ASSUMPTION that every compile lands there, not a check:
    # 25 measured a B=16 arm whose step 1 compiled 40 kernels, where the convention happened
    # to suffice by luck. The count below is what makes it a check.
    pooled = rows[1:] or rows
    warm_compiles = sum(r["compiles"] for r in pooled)
    if warm_compiles:
        raise SystemExit(
            f"{warm_compiles} kernels compiled after step 0, so the pooled rate includes "
            "codegen. Dropping the first step is not evidence that JIT is done -- only a "
            "zero count is."
        )
    w = sum(r["wall_s"] for r in pooled)
    tot = sum(r["tokens"] for r in pooled)
    ceiling = _ceiling_tok_s(args.group, weight_bytes)
    agg = tot / w
    print(f"\npooled over {len(pooled)} steps: {agg:.2f} tok/s aggregate")
    print(f"weight-stream ceiling at B={args.group}: {ceiling:.2f} tok/s"
          f"   measured/ceiling {agg / ceiling:.3f}")
    share = {k: sum(r["phases"].get(k, 0.0) for r in pooled) / w for k in pooled[0]["phases"]}
    for k, v in sorted(share.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<18} {100 * v:5.1f}%")

    # The reconstruction gate. `unattributed` is inside `share`, so this sums to 1 by
    # construction and is NOT the check -- the check is the non-negative assert per step
    # above, plus this bound on how much the leaves failed to explain.
    assert share["unattributed"] < 0.10, (
        f"{100 * share['unattributed']:.1f}% of the rollout is outside every timer; "
        "the decomposition does not describe this bucket"
    )

    # The floor check, which is what caught the async misattribution. A phase billed less
    # than the time to stream the weights it reads is not measuring that phase, whatever
    # the reconstruction says -- so this fires on the arm whose attribution is wrong
    # rather than letting a self-consistent table stand.
    ticks = sum(r["calls"].get("_run_forward", 0) for r in pooled)
    fwd_ms = 1000 * sum(r["phases"].get("_run_forward", 0.0) for r in pooled) / max(ticks, 1)
    floor_ms = weight_bytes / 1e9 / (_HBM_TBS * 1000) * 1000
    print(f"\nforward {fwd_ms:.2f} ms/tick vs weight-stream floor {floor_ms:.2f} ms/tick")
    if args.sync_forward:
        assert fwd_ms >= floor_ms, (
            f"forward billed {fwd_ms:.2f} ms/tick, below the {floor_ms:.2f} ms needed to "
            f"read {weight_gib:.2f} GiB at {_HBM_TBS} TB/s -- impossible, so the timer is "
            "still not measuring the forward even with the sync in place"
        )

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "aggregate_tok_s": agg, "ceiling_tok_s": ceiling,
                   "ratio_to_ceiling": agg / ceiling, "share": share,
                   "weight_bytes": weight_bytes, "weight_gib": weight_gib, "hbm_tbs": _HBM_TBS}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
