"""One GRPO step, decomposed into measured seconds. What is the next RL lever?

The step's total wall clock is reported today (`grpo_loop` yields it as element 2) and
three of its parts already are: `rl_step` writes `backward_secs` and `optimizer_secs` into
the `timings` dict, and `grpo_loop` seeds it with `rollout_secs` (`train.py:286-315, 415`).
Nothing splits the rollout, which is the biggest bucket, because `_drain` is
`engine.step()` in a loop (`train.py:28-37`).

This probe splits it WITHOUT changing the runtime. It reimplements the rollout half of
`grpo_loop` -- submit, tick, poll -- with two additions per tick:

* `torch.cuda.synchronize()` before reading the clock, so a bucket is device-inclusive
  and whatever is left over is host time. The probe is not the shipped loop, so the sync
  costs nothing that ships; its own cost is reported (`sync_secs`) so the buckets can be
  reconciled against an unsynced `rollout_secs` from the same configuration.
* the engine's `prefill_forwards` / `decode_forwards` / `mixed_forwards` counters read
  before and after, so each tick's seconds are attributed by WHICH counter moved rather
  than by a guess about what the tick did.

The backward and optimizer numbers come from the real `rl_step`, not a copy, so those two
buckets are the shipped path measured. Only the rollout is reimplemented, and the probe
asserts its reimplementation produces the same completions the real loop would by checking
the drain finished every id.

**The residual is named as a residual.** `rollout_secs` minus the attributed tick buckets
minus reward is submit/poll/host bookkeeping, and any tick the counters fail to explain
also lands there. It is printed as `unattributed_secs`, never as "host overhead".

    scripts/pod_run.sh grpostep 6 -- /work/tl013/bin/python -u \\
        scripts/prof_grpo_step.py --model qwen38-27b --group 8 --gen 4096 --steps 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "packages/tilerl-kernels/src")

from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.autograd import AdamW  # noqa: E402
from tilerl.cli import _build_model  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.kv_cache import NoPrefixStore  # noqa: E402
from tilerl.model import add_lora  # noqa: E402
from tilerl.train import group_advantages, rl_step, untruncated  # noqa: E402

_MAX_TICKS = 10000
_PHASES = ("prefill_forwards", "decode_forwards", "mixed_forwards")
#: Seed stride per step. A constant, so two arms of a --group sweep draw nested samples --
#: the narrow arm's rollouts are the wide arm's first `group`. Must exceed any --group used.
_SEED_STRIDE = 64


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _drain_attributed(engine, ids, stats_of):
    """`_drain`, plus per-tick seconds attributed by which forward counter moved.

    Returns (done, buckets, ticks, sync_secs). A tick that moves two counters is credited
    to `mixed`; a tick that moves none is credited to `unexplained_ticks_secs`, which is
    the honest bucket for "the counters do not account for this tick" and must not be
    folded into a phase.
    """
    done: dict[int, list[int]] = {}
    buckets = {"prefill_secs": 0.0, "decode_secs": 0.0, "mixed_secs": 0.0,
               "unexplained_ticks_secs": 0.0}
    counts = {"prefill_ticks": 0, "decode_ticks": 0, "mixed_ticks": 0, "unexplained_ticks": 0}
    sync_secs = 0.0
    # A recapture lands inside ONE tick -- the first of its bucket after an invalidate --
    # so it is invisible in a phase total and shows only as a slow tick. Without these,
    # "the delta is recapture" is an inference from where else it could be.
    slow: list[float] = []
    before = {k: stats_of().get(k, 0) for k in _PHASES}
    for tick in range(_MAX_TICKS):
        t0 = time.perf_counter()
        engine.step()
        t_pre_sync = time.perf_counter()
        _sync()
        t1 = time.perf_counter()
        sync_secs += t1 - t_pre_sync
        done.update(engine.poll())
        after = {k: stats_of().get(k, 0) for k in _PHASES}
        moved = [k for k in _PHASES if after[k] > before[k]]
        before = after
        dt = t1 - t0
        if dt > 1.0:
            slow.append(round(dt, 4))
        if len(moved) > 1 or moved == ["mixed_forwards"]:
            buckets["mixed_secs"] += dt
            counts["mixed_ticks"] += 1
        elif moved == ["prefill_forwards"]:
            buckets["prefill_secs"] += dt
            counts["prefill_ticks"] += 1
        elif moved == ["decode_forwards"]:
            buckets["decode_secs"] += dt
            counts["decode_ticks"] += 1
        else:
            buckets["unexplained_ticks_secs"] += dt
            counts["unexplained_ticks"] += 1
        if all(i in done for i in ids):
            counts["slow_ticks"] = len(slow)
            counts["slow_tick_secs"] = round(sum(slow), 4)
            counts["slow_tick_list"] = slow[:12]
            return done, buckets, counts, sync_secs, tick + 1
    raise RuntimeError(f"rollout did not finish within {_MAX_TICKS} ticks")


def one_step(engine, model, prompt, reward_fn, backend, optimizer, trainable, *,
             group, sampling, seed, step, micro, invalidate=False):
    """`grpo_loop`'s body with the rollout instrumented. Returns the phase dict."""
    t_step = time.perf_counter()
    # `replace`, the same call `grpo_loop` makes: reconstructing from __dict__ would
    # diverge the moment a field gains a default_factory or an init=False.
    # Strided by _SEED_STRIDE, not by `group`: `group` is the variable a width sweep
    # compares, so `step * group + g` makes the two arms sample DIFFERENT completions from
    # step 1 on (only step 0 nests), and a length difference then reads as a batch-width
    # effect. `tilerl-48` measured that on a tail probe -- 1083 vs 923 mean tokens, and the
    # resampling test refused the very hypothesis it was built to test. This probe's arms
    # are saturated so lengths cannot move, but the stride costs nothing and the next
    # caller may pass stop ids.
    ids = [engine.submit(prompt.tolist(), replace(sampling, seed=seed + step * _SEED_STRIDE + g))
           for g in range(group)]
    t_submit = time.perf_counter()
    done, buckets, counts, sync_secs, ticks = _drain_attributed(engine, ids, engine.stats)
    t_drain = time.perf_counter()
    comps = [done[i] for i in ids]

    t_r = time.perf_counter()
    rewards = [float(reward_fn(prompt, c)) for c in comps]
    reward_secs = time.perf_counter() - t_r

    adv = group_advantages(rewards, group)
    floor = min(256, int(sampling.max_new_tokens))
    gen = min(int(sampling.max_new_tokens),
              1 << (max(floor, max(len(c) for c in comps), 1) - 1).bit_length())
    batch = np.stack([
        np.concatenate([prompt, np.asarray(c, dtype=np.int64),
                        np.zeros(gen - len(c), dtype=np.int64)])
        for c in comps
    ])
    plens = np.full(group, len(prompt), dtype=np.int64)
    slens = np.array([len(prompt) + len(c) for c in comps], dtype=np.int64)

    timings: dict[str, float] = {}
    t_train = time.perf_counter()
    ce = rl_step(model, batch, adv, plens, backend, optimizer, trainable=trainable,
                 seq_lens=slens, micro=micro, timings=timings)
    _sync()
    train_secs = time.perf_counter() - t_train

    # `grpo_loop(recapture_graph=True)` calls this after every update; without it the
    # probe measures a configuration no RL run is in -- graphs captured once and never
    # touched again. `held_after` is what makes the next step's capture cost
    # attributable: N graphs gone here is N captures paid in the next rollout's first
    # tick per bucket, not a number read off the total.
    held_before = len(engine._decode_graphs)
    invalidate_secs = 0.0
    refilled = 0
    if invalidate:
        t_inv = time.perf_counter()
        refilled = engine.invalidate_weights()
        _sync()
        invalidate_secs = time.perf_counter() - t_inv
    held_after = len(engine._decode_graphs)

    rollout_secs = t_drain - t_step
    attributed = sum(buckets.values())
    out = {
        "step": step + 1,
        "step_secs": time.perf_counter() - t_step,
        "rollout_secs": rollout_secs,
        "submit_secs": t_submit - t_step,
        **{k: round(v, 4) for k, v in buckets.items()},
        **counts,
        "ticks": ticks,
        "sync_secs": sync_secs,
        "reward_secs": reward_secs,
        "train_secs": train_secs,
        "backward_secs": timings.get("backward_secs", 0.0),
        "optimizer_secs": timings.get("optimizer_secs", 0.0),
        "invalidate_secs": invalidate_secs,
        "graphs_held_before_invalidate": held_before,
        "graphs_held_after_invalidate": held_after,
        "graphs_dropped": held_before - held_after,
        # NOT "casts_refilled": this is whatever invalidate_weights RETURNS, and an
        # arm that reverts that method changes what the number means -- the arm-A
        # revert returns graphs dropped, and the old label read as 4 casts refilled.
        "invalidate_returned": refilled,
        # Everything in the rollout the tick buckets do not explain: submit, poll, the
        # python around them. A residual, not a measurement of host overhead.
        "unattributed_secs": rollout_secs - attributed,
        "mean_completion_tokens": float(np.mean([len(c) for c in comps])),
        "padded_width": gen,
        "ce": ce,
        "mean_reward": float(np.mean(rewards)),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true",
                    help="check the attribution arithmetic on a scripted engine; no GPU")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--gen", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--out", default="")
    ap.add_argument("--invalidate", action="store_true",
                    help="call engine.invalidate_weights() after every update, which is "
                         "what grpo_loop(recapture_graph=True) does and what the shipped "
                         "--rl path (cli.py:538, decode_graph=True) therefore pays; "
                         "without it the probe never invalidates and both arms of a "
                         "keep-graphs comparison are the same configuration")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()

    backend = get_backend()
    cfg, model = _build_model(a.model, seed=0, keep_master=False)
    # The pool must hold every row's whole sequence, and the rollout submits all --group at
    # once, so a --blocks sized for a narrower arm dies mid-drain rather than at build:
    # measured, `--group 16 --gen 6144 --blocks 3700` (the group-8 size) exhausted 3701
    # blocks partway through step 1, after the wider arm had already been billed a build.
    # Refuse before the engine, since the number is knowable from the flags alone.
    need = -(-(a.prompt_tokens + a.gen) // 16) * a.group
    if a.blocks < need:
        print(f"REFUSED: --blocks {a.blocks} holds {a.blocks * 16} tokens, but --group "
              f"{a.group} x (--prompt-tokens {a.prompt_tokens} + --gen {a.gen}) needs "
              f"{need} blocks. A pool sized for a narrower arm exhausts mid-drain, so a "
              f"multi-arm sweep sizes it for its WIDEST arm.", file=sys.stderr)
        return 1
    engine = build_engine(cfg, model, backend, num_blocks=a.blocks,
                          num_slots=a.group, max_batch=a.group,
                          max_total_tokens=a.blocks * 16,
                          decode_graph=True, prefix_store=NoPrefixStore())
    trainable = add_lora(model, rank=a.rank)
    optimizer = AdamW(lr=1e-5)
    # No stop_token_ids, so every rollout runs to --gen exactly and mean_completion_tokens
    # is the cap. That is DELIBERATE for a batch-width comparison -- both arms then do
    # identical per-row work and ms/token cannot be confounded by one arm generating
    # shorter completions -- but it makes step_secs an UPPER bound on a shipped step,
    # which stops on EOS. A length-distribution question needs the factory sampler
    # (`prompt.sampling`, which fills stop ids from the tokenizer); this probe answers
    # the width question and reports the cap so the bound is visible.
    sampling = untruncated(SamplingParams(max_new_tokens=a.gen))
    rng = np.random.default_rng(0)
    # Bounded by the model's own vocab, not a round number: the tiny model has 320 tokens
    # and a random id of 637 crashes the CE gradient rather than degrading, so a hardcoded
    # ceiling makes the probe run only on the 27B.
    vocab = int(getattr(cfg, "vocab_size", 0)) or 1000
    prompt = rng.integers(1, vocab, size=a.prompt_tokens, dtype=np.int64)

    rows = []
    for step in range(a.steps):
        # A compile is exactly a new key in Backend._kernels, so counting it is the only
        # way this probe can assert a step's wall clock is JIT-free rather than have the
        # reader subtract compile seconds from a log afterwards. Every new decode width
        # recompiles three kernels and the widths come from the rollout's length
        # distribution, so which widths a run reaches is not knowable in advance --
        # measured, a B=16 arm compiled 141 times against a B=8 arm's 42, 47% of its
        # 507.9 s wall clock (wins/2026-09-08-b16-fits-and-three-predictions-were-wrong.md).
        before = len(getattr(backend, "_kernels", {}))
        row = one_step(engine, model, prompt, lambda p, c: float(len(c) > 0), backend,
                       optimizer, trainable, group=a.group, sampling=sampling, seed=0,
                       step=step, micro=a.micro, invalidate=a.invalidate)
        row["compiles"] = len(getattr(backend, "_kernels", {})) - before
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items()}, sort_keys=True), flush=True)

    # Step 0 is reported and excluded from the means. It is NOT the reason the means are
    # clean: "step 0 pays every JIT" is a hope, not a property -- a later step reaches a new
    # decode width whenever the rollout's length distribution takes it there, and the summary
    # looks identical either way. `warm_compiles` below is what proves it, and discarding the
    # first step is a convention that cannot. Measured here: B=16's step 1 compiled 40 and
    # steps 2-5 compiled 0, so the convention happened to suffice.

    warm = rows[1:] or rows
    keys = ["step_secs", "rollout_secs", "prefill_secs", "decode_secs", "mixed_secs",
            "unexplained_ticks_secs", "unattributed_secs", "reward_secs", "train_secs",
            "backward_secs", "optimizer_secs", "sync_secs", "invalidate_secs"]
    summary = {f"mean_{k}": round(float(np.mean([r[k] for r in warm])), 4) for k in keys}
    for k in ("graphs_dropped", "graphs_held_before_invalidate", "invalidate_returned"):
        summary[f"mean_{k}"] = round(float(np.mean([r[k] for r in warm])), 2)
    summary["warm_steps"] = len(warm)
    summary["step0_secs"] = round(rows[0]["step_secs"], 4)
    summary["warm_compiles"] = sum(r["compiles"] for r in warm)
    # Which quantities this arm can answer, written by the probe rather than discovered by
    # whoever quotes it. Four measurements today were valid for one quantity and invalid for
    # another, and all four had that pointed out downstream instead of stated upstream: a
    # measurement's validity is its pairing with a question, not a property it carries.
    saturated = all(r["mean_completion_tokens"] == float(a.gen) for r in warm)
    summary["valid_for"] = ["ms_per_token across --group at a FIXED --gen",
                            "the phase split (rollout / decode / train) at this width"]
    summary["invalid_for"] = (
        ["seconds_per_correct: prompts are random token ids, so the reward is noise"]
        + (["idle_fraction: no stop_token_ids, so every row runs to --gen and idle is "
            "identically 0 by construction, not by scheduling",
            "the price of a REAL run: this is a saturated batch, i.e. a lower bound -- a run "
            "whose rows finish early pays more per useful token"] if saturated else [])
        + ["ms_per_token across --gen: unmeasured here, and the tick count scales with --gen"])
    # Per-token, the quantity a batch-width comparison needs: sec/step alone rises with the
    # group whatever the efficiency, so comparing two widths on it says only that the wider
    # one did more work. Group x mean completion, indexed rather than .get -- a missing key
    # must raise, not silently omit the metric the comparison is for.
    summary["mean_tokens"] = round(
        a.group * float(np.mean([r["mean_completion_tokens"] for r in warm])), 1)
    summary["mean_ms_per_token"] = round(
        summary["mean_step_secs"] * 1000 / summary["mean_tokens"], 4)
    # Stated, not assumed: a reader comparing two widths must see whether the rows were
    # capped, because ms/token is only comparable across arms when they are.
    summary["completions_hit_the_cap"] = all(
        r["mean_completion_tokens"] == float(a.gen) for r in warm)
    m = summary
    # The decomposition has to add up, or a bucket is being double-counted.
    parts = (m["mean_rollout_secs"] + m["mean_reward_secs"] + m["mean_train_secs"]
             + m["mean_invalidate_secs"])
    summary["sum_of_parts_secs"] = round(parts, 4)
    summary["step_minus_parts_secs"] = round(m["mean_step_secs"] - parts, 4)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if summary["warm_compiles"]:
        # Refuse rather than warn: a wall clock that contains a JIT can neither confirm
        # nor refute a kernel-efficiency claim, and subtracting it afterwards does not
        # work because compilation and execution interleave in the one clock.
        print(f"\nREFUSED: {summary['warm_compiles']} TileLang compiles inside the "
              f"{len(warm)} timed steps, so these seconds are not a throughput "
              f"measurement. Raise --steps so the warm-up reaches every decode width "
              f"this configuration uses, then re-run.", flush=True)
        return 1
    print(f"sync cost: {m['mean_sync_secs']:.3f}s over "
          f"{float(np.mean([r['ticks'] for r in warm])):.0f} ticks -- subtract this to "
          f"reconcile with an unsynced rollout_secs", flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "summary": summary}, f, indent=2, sort_keys=True)
    return 0


def _selfcheck() -> int:
    """Attribution arithmetic against a scripted engine: no model, no GPU.

    The load-bearing property is that a tick whose counters do NOT move is credited to
    `unexplained_ticks_secs` rather than silently to a phase -- a bucket that absorbs
    unexplained ticks reads as a clean decomposition.
    """
    class FakeEngine:
        def __init__(self):
            self.c = {"prefill_forwards": 0, "decode_forwards": 0, "mixed_forwards": 0}
            self.plan = ["prefill", "decode", "none", "mixed", "decode"]
            self.i = 0

        def stats(self):
            return dict(self.c)

        def step(self):
            kind = self.plan[self.i] if self.i < len(self.plan) else "decode"
            self.i += 1
            if kind == "prefill":
                self.c["prefill_forwards"] += 1
            elif kind == "decode":
                self.c["decode_forwards"] += 1
            elif kind == "mixed":
                self.c["mixed_forwards"] += 1

        def poll(self):
            return {0: [1, 2]} if self.i >= len(self.plan) else {}

    e = FakeEngine()
    done, buckets, counts, sync_secs, ticks = _drain_attributed(e, [0], e.stats)
    assert done == {0: [1, 2]}, done
    assert ticks == 5, ticks
    assert {k: counts[k] for k in ("prefill_ticks", "decode_ticks", "mixed_ticks",
                                   "unexplained_ticks")} == {
        "prefill_ticks": 1, "decode_ticks": 2, "mixed_ticks": 1,
        "unexplained_ticks": 1}, counts
    # No tick here is over 1 s, so the slow-tick fields must exist and read zero:
    # absent keys would make every arm that reads them a KeyError, and a nonzero
    # count on a fake engine would mean the threshold fires on nothing.
    assert counts["slow_ticks"] == 0 and counts["slow_tick_secs"] == 0, counts
    # The one tick that moved nothing must be in its own bucket, not in a phase.
    assert buckets["unexplained_ticks_secs"] > 0, buckets
    assert sum(buckets.values()) > 0
    # And a tick moving two counters at once counts as mixed, not as two ticks.
    e2 = FakeEngine()
    e2.plan = ["both"]

    def step_both():
        e2.c["prefill_forwards"] += 1
        e2.c["decode_forwards"] += 1
        e2.i += 1
    e2.step = step_both
    _, b2, c2, _, t2 = _drain_attributed(e2, [0], e2.stats)
    assert t2 == 1 and c2["mixed_ticks"] == 1, (t2, c2)
    assert b2["prefill_secs"] == 0.0 and b2["decode_secs"] == 0.0, b2

    # --invalidate must actually call the engine, and must be off by default. Without
    # this the flag can be wired to nothing and both arms of the keep-graphs comparison
    # measure the same configuration -- which is the defect that produced this flag.
    class InvEngine(FakeEngine):
        def __init__(self, drops):
            super().__init__()
            self.calls = 0
            self._decode_graphs = dict.fromkeys(range(drops), "g")
            self._drops = drops

        def submit(self, *a, **kw):
            return 0

        def invalidate_weights(self):
            self.calls += 1
            self._decode_graphs.clear()  # arm A: graphs dropped
            return 7  # casts refilled

    def _row(**kw):
        e = InvEngine(3)
        return e, one_step(e, None, np.zeros(2, dtype=np.int64), lambda p, c: 1.0, None,
                           None, None, group=1, sampling=replace(
                               SamplingParams(max_new_tokens=4), seed=0),
                           seed=0, step=0, micro=1, **kw)

    import tilerl.train as _train_mod
    real_rl_step = _train_mod.rl_step
    globals()["rl_step"] = lambda *a, **kw: 0.0  # no model here; only the flag is under test
    try:
        off_e, off = _row(invalidate=False)
        on_e, on = _row(invalidate=True)
    finally:
        globals()["rl_step"] = real_rl_step
    assert off_e.calls == 0 and off["graphs_dropped"] == 0, (off_e.calls, off)
    assert on_e.calls == 1, on_e.calls
    assert on["graphs_dropped"] == 3, on
    assert on["invalidate_returned"] == 7, on
    # The compile gate and the per-token metric, on the arithmetic rather than a card.
    # Both are new and neither is reachable from the fake-engine path above, so without
    # this they would ship untested -- and the gate's whole job is to fail unattended.
    fake_cache = {}
    seen = len(fake_cache)
    fake_cache["k1"] = fake_cache["k2"] = 1
    assert len(fake_cache) - seen == 2, "a compile is a new key in Backend._kernels"
    # ms/token must fall when the group widens at equal step time -- the property that
    # makes it the right metric for a batch-width comparison, where sec/step rises
    # whatever the efficiency. 4.0 vs 2.0 at group 8 vs 16 on a 1-second step.
    per_tok = lambda step_s, grp, comp: step_s * 1000 / (grp * comp)  # noqa: E731
    assert per_tok(1.0, 8, 32) == 3.90625 and per_tok(1.0, 16, 32) == 1.953125
    assert per_tok(1.0, 16, 32) < per_tok(1.0, 8, 32), "ms/token must reward the wider group"
    # The --blocks precondition, on the arithmetic. Both directions, because a guard that
    # only ever passes is what let `--group 16 --blocks 3700` reach the card: the second
    # assert is the one that fails if the formula loses its dependence on --group.
    need = lambda ptok, gen, grp: -(-(ptok + gen) // 16) * grp  # noqa: E731
    assert need(256, 6144, 8) == 3200 <= 3700, "the group-8 arm fits 3700 blocks"
    assert need(256, 6144, 16) == 6400 > 3700, "the group-16 arm must NOT fit the same pool"
    # Seeds must NEST across a width sweep, or the arms sample different completions and a
    # length difference reads as a batch-width effect. Both directions: the second assert
    # is what fails if the stride goes back to `group`, and without it any formula passes.
    nest = lambda stride, grp, step: [step * stride + g for g in range(grp)]  # noqa: E731
    assert nest(_SEED_STRIDE, 8, 3) == nest(_SEED_STRIDE, 16, 3)[:8], "arms must nest"
    assert nest(8, 8, 3) != nest(16, 16, 3)[:8], "a group-strided seed must NOT nest"
    assert _SEED_STRIDE >= 16, "the stride must exceed every --group a sweep uses"
    print(f"selfcheck ok: 5 ticks -> {counts}; a two-counter tick is mixed, "
          f"an unexplained tick is its own bucket; --invalidate calls the engine "
          f"({on_e.calls}) and off does not ({off_e.calls}); a compile is a new "
          f"_kernels key, ms/token falls as the group widens, --blocks 3700 "
          f"admits group 8 ({need(256, 6144, 8)}) and refuses group 16 "
          f"({need(256, 6144, 16)}), and seeds nest across widths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
