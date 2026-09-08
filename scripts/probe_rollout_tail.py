#!/usr/bin/env python3
"""How much of a GRPO rollout is slots idling because one row in the group runs long?

`grpo_loop` (train.py:435-443) submits one prompt's `group` rollouts together and drains
until the last finishes. Nothing else is queued behind them, so a row that stops early
leaves its slot empty for the rest of the step: the step costs `max(len)` ticks while the
work is `sum(len)` tokens.

    idle_fraction = 1 - sum(len) / (max(len) * group)

That is the ceiling on what tail-aware packing or partial rollout could return -- an upper
bound, not a forecast, because a scheduler that refills a slot pays for the refill and
because the freed capacity is only useful if there is work to put in it.

**The bound is only real if a finished row's slot actually idles**, so that was checked in
the code rather than assumed: `grpo_loop` builds `ids` from one prompt (train.py:439-442)
and hands exactly those to `_drain` (`:443`), which ticks `engine.step()` until every id
is done (`:35-44`) and submits nothing. No other request can occupy the slot.

**Real prompts, not synthetic.** Length distribution is the quantity under test, and a
fixed or random prompt set would fabricate it -- the same way a random-normal fixture put
top-p's nucleus at 162301/248320 when the real one was 43 (2026-09-08). Reads
`/work/p1_gsm8k_train.jsonl`.

**And the real SamplingParams, for the same reason.** The first run of this probe built
`SamplingParams(max_new_tokens=1024, seed=...)` directly. `stop_token_ids` defaults to
`()` (engine.py:149), so no EOS could end a row and all 160 rollouts ran to exactly 1024
tokens: idle 0.0% on every step, a number produced entirely by the probe's own arguments.
The shipped path builds params through `prompt.sampling(tok, thinking, ...)`
(prompt.py:56-62), which fills `stop_token_ids` from the tokenizer, and `grpo_loop` then
applies `untruncated()` (train.py:376). This does both, so a row can stop when the model
stops.

**Reported per step and pooled, with the spread.** One step's idle fraction is one draw
from a distribution over prompts; a single number would read as a property of the loop.

This measures only. It changes no scheduling.

Run:
  scripts/pod_run.sh --wait tail <card> -- python3 scripts/probe_rollout_tail.py \\
      --steps 20 --groups 8,16
"""
import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

_PROMPTS = "/work/p1_gsm8k_train.jsonl"


def _idle(lengths: list[int]) -> float:
    """Fraction of slot-ticks spent on a finished row, 0 when every row is equal."""
    return 1.0 - sum(lengths) / (max(lengths) * len(lengths))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--groups", default="8,16")
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--prompts", default=_PROMPTS)
    ap.add_argument("--thinking", action="store_true",
                    help="the model card's thinking-mode sampler; default is non-thinking, "
                         "which is what the P1 GRPO runs use")
    ap.add_argument("--out", default="/work/rollout_tail.json")
    args = ap.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model, _qwen38_tokenizer
    from tilerl.engine import build_engine
    from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
    from tilerl.prompt import sampling as build_sampling
    from tilerl.train import _drain, untruncated

    # The training path's own helper (cli.py:33), which reads the local checkpoint dir.
    # `get_tokenizer("qwen38-27b")` treats the name as a hub id and the pod has no network.
    tok = _qwen38_tokenizer()
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            if len(prompts) >= args.steps:
                break
            row = json.loads(line)
            text = row.get("question") or row.get("prompt") or row.get("text")
            if text:
                prompts.append(tok.encode(text))
    if len(prompts) < args.steps:
        raise SystemExit(f"{args.prompts}: {len(prompts)} usable prompts, need {args.steps}")
    print(f"{len(prompts)} real prompts, token lengths "
          f"{min(map(len, prompts))}..{max(map(len, prompts))}")

    # The rollout sampler as grpo_loop builds it, not a hand-rolled one.
    base = untruncated(build_sampling(tok, args.thinking, args.gen, seed=0))
    if not base.stop_token_ids:
        raise SystemExit(
            "stop_token_ids is empty, so no row can stop before the cap and every idle "
            "fraction would be 0 by construction -- which is what the first run measured"
        )
    print(f"stop_token_ids {base.stop_token_ids}  temperature {base.temperature} "
          f"top_p {base.top_p} top_k {base.top_k}")

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
    groups = [int(g) for g in args.groups.split(",")]
    ctx = args.gen + max(map(len, prompts)) + 64
    out: dict[str, list] = {}

    for group in groups:
        engine = build_engine(cfg, model, backend, num_slots=group, max_batch=group,
                              num_blocks=-(-ctx // BLOCK_TOKENS) * group + group,
                              max_total_tokens=max(ctx, 8192),
                              decode_graph=True, prefix_store=NoPrefixStore())
        rows = []
        print(f"\ngroup {group}")
        print(f"  {'step':>4} {'min':>5} {'med':>6} {'max':>5} {'sum':>6} "
              f"{'idle%':>6} {'wall s':>7}")
        for step, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            ids = [engine.submit(prompt, replace(base, seed=step * group + g))
                   for g in range(group)]
            done = _drain(engine, ids, "tail probe")
            wall = time.perf_counter() - t0
            lens = sorted(len(done[i]) for i in ids)
            idle = _idle(lens)
            rows.append({"step": step, "lengths": lens, "idle": idle, "wall_s": wall})
            print(f"  {step:>4} {lens[0]:>5} {statistics.median(lens):>6.0f} {lens[-1]:>5} "
                  f"{sum(lens):>6} {100 * idle:>5.1f}% {wall:>7.2f}")

        idles = [r["idle"] for r in rows]
        pooled = 1.0 - sum(sum(r["lengths"]) for r in rows) / sum(
            max(r["lengths"]) * group for r in rows)
        print(f"  pooled idle {100 * pooled:.1f}%   per-step median {100 * statistics.median(idles):.1f}%"
              f"   range {100 * min(idles):.1f}-{100 * max(idles):.1f}%")
        out[str(group)] = rows
        # A group whose rows all hit the cap has zero spread and zero idle -- a real
        # reading, but it means the cap truncated the distribution rather than that the
        # tail is absent, so say which one this is.
        capped = sum(1 for r in rows if r["lengths"][0] >= args.gen)
        if capped:
            print(f"  {capped}/{len(rows)} steps had every row at the {args.gen} cap: "
                  "their 0% idle is truncation, not balance")

    if len(groups) > 1:
        print("\ngroup   pooled idle")
        for g in groups:
            rs = out[str(g)]
            p = 1.0 - sum(sum(r["lengths"]) for r in rs) / sum(max(r["lengths"]) * g for r in rs)
            print(f"{g:>5}   {100 * p:>10.1f}%")

    with open(args.out, "w") as f:
        json.dump({"gen_cap": args.gen, "steps": args.steps, "by_group": out}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
