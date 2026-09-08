"""GSM8K rollout length distribution on the 27B: what cap does this task need?

The recorded 1029.1-token mean is MATH's before-arm, not GSM8K's, and a cap chosen from the
wrong task is what made the first ISO-RL arm read `tied 100%` at 128 tokens. So measure this
task's own distribution, with no training: submit N prompts, read the completion lengths and
whether each reached an answer.

Two quantities, because they answer different questions:
  * length distribution -> the cap the policy needs
  * fraction hitting the cap -> whether the cap chosen is already binding

Also reports reward variance across seeds at fixed weights, which is the error bar the
reward-vs-step curve needs and which a fixed seed reports as zero.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TILERL_TARGET", "cuda")

from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.cli import _build_model, _qwen38_tokenizer  # noqa: E402
from tilerl.engine import build_engine  # noqa: E402
from tilerl.eval import MATCHERS  # noqa: E402
from tilerl.kv_cache import NoPrefixStore  # noqa: E402
from tilerl.model import drop_quantized  # noqa: E402
from tilerl.prompt import render_chat, sampling  # noqa: E402
from tilerl.train import _drain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/work/p1_gsm8k_train.jsonl")
ap.add_argument("--prompts", type=int, default=16)
ap.add_argument("--group", type=int, default=4, help="samples per prompt, at temperature 1")
ap.add_argument("--cap", type=int, default=1024, help="max_new_tokens; report the cap-hit rate")
ap.add_argument("--matcher", default="number", choices=list(MATCHERS))
ap.add_argument("--blocks", type=int, default=2048)
ap.add_argument("--seeds", default="0,1,2", help="reward variance at FIXED weights")
# Same flag and same derivation as the RL path (cli.py:577, :1388): thinking is a function of
# the think cap, not an independent switch. A probe whose thinking mode differs from the arm's
# measures a different policy's lengths.
ap.add_argument("--max-think-tokens", type=int, default=0,
                help="cap on <think>; 0 = thinking off, matching `train --rl`'s default")
a = ap.parse_args()
thinking = a.max_think_tokens > 0

cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
drop_quantized(model)
backend = get_backend()
engine = build_engine(cfg, model, backend, num_blocks=a.blocks, num_slots=8,
                      decode_graph=False, prefix_store=NoPrefixStore())
tok = _qwen38_tokenizer()
match = MATCHERS[a.matcher]

# The run this replaces reported `mean 1024.0, at cap 100.0%` on a 1024 cap: a hand-rolled
# SamplingParams left `stop_token_ids` at its `()` default, so engine.py:1370 could never fire
# and the "length distribution" was the cap's. Nothing in the output distinguishes that from a
# policy that genuinely runs long, so the precondition is asserted rather than eyeballed.
_probe = sampling(tok, thinking, a.cap, temperature=1.0, seed=0)
assert _probe.stop_token_ids, "no stop ids: every rollout will run the full cap"

rows = [json.loads(ln) for ln in Path(a.data).read_text().splitlines() if ln.strip()]
rows = rows[:a.prompts]
print(f"{len(rows)} prompts x group {a.group}, cap {a.cap}, matcher {a.matcher}, "
      f"thinking {thinking}\n", flush=True)

seeds = [int(s) for s in a.seeds.split(",")]
per_seed = []
for seed in seeds:
    # One submit wave, one drain: the first version drained inside the prompt loop, so 8
    # prompts ran as 8 sequential waves of `group` against an 8-slot engine. A rollout probe
    # that serializes measures the engine's latency, not the policy's, and takes 8x the wall
    # clock for the same rows. `submit` queues past the slot count rather than raising
    # (engine.py:1697's warning), so a wave wider than the pool is correct, just batched.
    rids, gold_of = [], {}
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        # `render_chat` and `prompt.sampling`, not a bare encode and a hand-rolled
        # SamplingParams. Two things the hand-rolled version silently lost:
        #   * the chat template -- eval.py:146 renders every gsm8k row this way, and a base
        #     model fed a raw question is not the input the policy was trained on;
        #   * `stop_token_ids` -- engine.py:1370 stops on them and SamplingParams defaults to
        #     (), so every rollout ran the full cap whether or not it emitted <|im_end|>.
        #     That inflates the wall clock AND makes the length distribution the cap's rather
        #     than the policy's, which is the number this probe exists to produce.
        text = render_chat([("user", r["prompt"])], thinking)
        ids = tok.encode(text)
        for g in range(a.group):
            rid = engine.submit(list(ids), sampling(
                tok, thinking, a.cap, temperature=1.0,
                max_think_tokens=a.max_think_tokens or None,
                seed=seed * 10007 + i * a.group + g))
            rids.append(rid)
            gold_of[rid] = r["answer"]
    done = _drain(engine, rids, "length probe")
    wall = time.perf_counter() - t0
    lens, correct = [], 0
    for rid in rids:
        c = done[rid]
        lens.append(len(c))
        correct += int(match(tok.decode([int(t) for t in c]), gold_of[rid]))
    n = len(lens)
    arr = np.array(lens)
    acc = correct / n
    per_seed.append((seed, acc, arr))
    print(f"seed {seed}: accuracy {100*acc:5.2f}%  mean {arr.mean():7.1f}  "
          f"median {np.median(arr):7.1f}  p90 {np.percentile(arr, 90):7.1f}  "
          f"max {arr.max():5d}  at cap {100*(arr >= a.cap).mean():5.1f}%  "
          f"wall {wall:6.1f} s = {wall/n:5.2f} s/rollout", flush=True)

allarr = np.concatenate([x[2] for x in per_seed])
accs = np.array([x[1] for x in per_seed])
print(f"\npooled over {len(seeds)} seeds, n={len(allarr)}:")
print(f"  mean {allarr.mean():.1f}  median {np.median(allarr):.1f}  "
      f"p90 {np.percentile(allarr, 90):.1f}  p99 {np.percentile(allarr, 99):.1f}  "
      f"max {allarr.max()}")
print(f"  at cap: {100*(allarr >= a.cap).mean():.1f}%  "
      f"-> a cap of {a.cap} {'BINDS' if (allarr >= a.cap).mean() > 0.05 else 'does not bind'}")
print(f"\nreward across seeds at FIXED weights: {', '.join(f'{100*x:.2f}%' for x in accs)}")
print(f"  sd {100*accs.std(ddof=1):.2f} pt over {len(accs)} seeds, n={len(allarr)//len(seeds)} "
      f"rollouts each")
print("  This is the error bar the reward-vs-step curve needs. A fixed eval seed reports 0")
print("  and then every wiggle on the curve reads as signal (cli.py:570-572 uses args.seed).")
