#!/usr/bin/env python3
"""Per-problem pass@k distribution: how many of k samples each problem gets right.

One run answers three questions that were being answered separately, wrongly:

  base accuracy    total correct / (n*k)
  tied fraction    the share of problems at 0/k or k/k -- constant advantage in GRPO
  usable subset    the problems at 1/k..k-1/k, which are the only ones with gradient

The third is the one we have never had, and it is why this exists. "Pick a harder
task" was the wrong move all along: difficulty (base accuracy) is a proxy for what
actually decides whether GRPO learns, which is whether a group's rewards differ.
A problem at 3/8 carries gradient no matter how the dataset is labelled; one at 8/8
or 0/8 is a constant, and level 5 turned out to be EASIER than GSM8K once its cap
stopped truncating correct answers (91.0% vs 88.0%,
errors/2026-09-08-a-cap-reported-as-a-base.md).

Samples at temperature 1.0 with a distinct seed per (problem, sample). Greedy would
make every sample identical and the distribution a two-point mass at 0 and k by
construction -- the thing being measured is variance across samples, so removing the
variance removes the measurement. `--temperature` is settable because the rollout
temperature is what the tie fraction should be read at, not a canonical 1.0.

Writes one JSON row per problem, so the distribution is recoverable and the
aggregate is not the only artifact -- a fraction cannot be un-summed.

  python3 scripts/pass_at_k.py --data F --out O [--k 8] [--n 100] [--max-new-tokens 6144]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages",
                                "tilerl-kernels", "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl with {prompt, answer}")
    ap.add_argument("--out", required=True, help="jsonl, one row per problem")
    ap.add_argument("--k", type=int, default=8, help="samples per problem")
    ap.add_argument("--n", type=int, default=100, help="problems")
    ap.add_argument("--offset", type=int, default=0, help="skip this many problems (sharding)")
    ap.add_argument("--max-new-tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--batch", type=int, default=16,
                    help="engine width; 16 fills a wgmma M tile, 8 fills half of one")
    args = ap.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model, _qwen38_tokenizer
    from tilerl.engine import BLOCK_TOKENS, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.math_answer import boxed_match
    from tilerl.prompt import render_chat
    from tilerl.prompt import sampling as sampling_of
    from tilerl.tokenizer import get_tokenizer

    with open(args.data) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows = rows[args.offset:args.offset + args.n]
    print(f"pass_at_k: {len(rows)} problems x k={args.k} at cap {args.max_new_tokens}, "
          f"temperature {args.temperature}", flush=True)

    cfg, model = _build_model(args.model, seed=0)
    # The tiny path exists so this script's own logic is exercisable without a card.
    tok = _qwen38_tokenizer() if args.model == "qwen38-27b" else get_tokenizer(None)
    backend = get_backend()
    prompts = [render_chat([("user", r["prompt"])], False) for r in rows]
    ctx = max(max(len(tok.encode(p)) for p in prompts) + args.max_new_tokens + 64, 1024)
    engine = build_engine(cfg, model, backend, num_slots=args.batch, max_batch=args.batch,
                          num_blocks=-(-ctx // BLOCK_TOKENS) * args.batch + 8,
                          max_total_tokens=max(ctx, 8192), decode_graph=True,
                          prefix_store=NoPrefixStore())

    t0 = time.perf_counter()
    total_correct = total_tokens = distinct_1 = all_at_cap = 0
    hist = {j: 0 for j in range(args.k + 1)}
    with open(args.out, "w") as fh:
        for i, (row, prompt) in enumerate(zip(rows, prompts)):
            # All k in ONE engine batch, not k sequential calls: the samples are independent,
            # so serializing them would decode at width 1 and pay k times the weight stream --
            # the whole reason the engine is sized to `batch`. Submitted directly rather than
            # through generate_ids, which takes one SamplingParams for every prompt and so
            # cannot vary the seed per sample.
            ids_of = {}
            pid = tok.encode(prompt)
            for g in range(args.k):
                # Built through `prompt.sampling`, not SamplingParams(...) directly: the
                # constructor's stop_token_ids default is EMPTY, so a hand-built params
                # object lets nothing end a completion and every sample runs to the cap.
                # Measured: the first version did exactly that -- 64 of 64 samples at 6144,
                # against 3 of 32 on the same problems through the eval path -- so the
                # distribution was a property of the cap and the whole run was void.
                sp = replace(sampling_of(tok, None, args.max_new_tokens, seed=i * args.k + g),
                             temperature=args.temperature)
                ids_of[engine.submit(pid, sp)] = g
            done: dict[int, list[int]] = {}
            while len(done) < args.k:
                engine.step()
                for wid, comp in engine.poll().items():
                    done[ids_of[wid]] = comp
            ids = [done[g] for g in range(args.k)]
            hits = [bool(boxed_match(tok.decode(d), row["answer"])) for d in ids]
            c = sum(hits)
            hist[c] += 1
            total_correct += c
            total_tokens += sum(len(d) for d in ids)
            rec = {"i": args.offset + i, "correct": c, "k": args.k,
                   "tokens": [len(d) for d in ids], "answer": row["answer"],
                   "at_cap": sum(1 for d in ids if len(d) >= args.max_new_tokens),
                   # The measurement is variance ACROSS samples, so identical samples make
                   # every problem read 0/k or k/k and the tie fraction 100% by construction.
                   # A seed that failed to vary, or temperature reaching the sampler as 0,
                   # would look exactly like a genuinely decided problem. Recorded per row so
                   # a degenerate run is visible in the artifact rather than inferred from it.
                   "distinct": len({tuple(d) for d in ids})}
            # Not gated on at_cap: truncation does not make identical samples legitimate.
            # The first version excluded all-at-cap rows, which silenced this in exactly the
            # case the negative control produces -- temperature 0.0, every sample identical,
            # every one truncated -- so the guard reported nothing on the one input built to
            # trip it.
            if rec["distinct"] == 1:
                distinct_1 += 1
            if rec["at_cap"] == args.k:
                all_at_cap += 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()   # a long run must be readable while it runs, not only after
            print(f"  {args.offset + i:4d}: {c}/{args.k}  "
                  f"tokens {min(rec['tokens'])}-{max(rec['tokens'])}  "
                  f"at_cap {rec['at_cap']}  distinct {rec['distinct']}  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

    n = len(rows)
    tied = (hist[0] + hist[args.k]) / max(n, 1)
    print(f"\npass_at_k: n={n} k={args.k}")
    print(f"  base accuracy   {total_correct}/{n * args.k} = "
          f"{100 * total_correct / max(n * args.k, 1):.1f}%")
    print(f"  tied fraction   {hist[0]} at 0/{args.k} + {hist[args.k]} at {args.k}/{args.k} "
          f"= {100 * tied:.1f}%")
    print(f"  usable subset   {n - hist[0] - hist[args.k]} problems at "
          f"1..{args.k - 1}/{args.k} = {100 * (1 - tied):.1f}%")
    print(f"  histogram       {json.dumps(hist)}")
    print(f"  tokens          {total_tokens} in {time.perf_counter() - t0:.0f}s = "
          f"{total_tokens / max(time.perf_counter() - t0, 1e-9):.1f} tok/s")
    # A problem whose every sample hit the cap has not been scored -- it has been prevented
    # from being scored, so its 0/k is the cap's verdict and not the policy's. The first
    # version of this script set stop_token_ids to the SamplingParams default (empty), so
    # nothing could end a completion and this was 100%: the tie fraction it reported was a
    # property of the cap (errors/2026-09-08-a-cap-reported-as-a-base.md is the same defect
    # one level up). Refuses rather than warns above half, because a distribution measuring
    # the cap is not a weaker version of the measurement, it is a different one.
    if all_at_cap:
        pct = 100 * all_at_cap / max(n, 1)
        line = (f"  {'REFUSING' if pct > 50 else 'WARNING'}: {all_at_cap} of {n} problems had "
                f"ALL {args.k} samples hit the {args.max_new_tokens} cap. Those rows measure "
                f"the cap, not the problem -- a truncated completion is unscored, not wrong. "
                f"Raise --max-new-tokens, or check that stop_token_ids reached the sampler.")
        print(line)
        if pct > 50:
            return 1
    if distinct_1:
        print(f"  WARNING: {distinct_1} of {n} problems produced {args.k} IDENTICAL "
              f"samples. The tie fraction above is then a property of the "
              f"sampler, not of the problems -- check that --temperature reached the sampler "
              f"and that the per-sample seed varies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
