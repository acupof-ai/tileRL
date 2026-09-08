#!/usr/bin/env python3
"""Are the GSM8K rows that hit the 6144 cap solving, or looping?

Twelve of eighteen tail-probe steps had a row reach 5000+ tokens on grade-school
arithmetic whose reference answers run 100-150 tokens. If those tokens are a repetition
loop, then today's tail pricing (refill 1.53x, train padding 3.891x) is pricing a bug:
the fix for a row that loops is to stop it, not to schedule around it.

**Repetition is not the test.** A worked solution repeats by construction -- "Step 7:",
"Step 8:", "= 12", "Therefore" -- so a high n-gram repeat rate inside one window is the
normal signature of the format, not of degeneration (27/0a, 2026-09-08). Two things follow:

1. **The metric is cross-window recurrence, not within-window repetition.** Of the
   20-grams in the last WINDOW tokens, how many also occur in the WINDOW before it. A
   loop repeats a whole span, so this approaches 1; a template repeats short phrases
   with changing numbers, and a 20-gram spans past the changing part.
2. **The control is this run's own naturally-terminated rows**, not another dataset's
   normal. Absolute rates are meaningless here; only capped-vs-natural is.

The verdict threshold is fixed before the run: **capped >= 2x natural** on the
recurrence metric. 40% against a 35% control is not a finding.

gzip ratio of the same window runs alongside, as a second, independent view -- looped
text compresses far better. It is reported, not thresholded; two metrics that disagree is
information, and picking whichever one crosses a line afterwards is not.

**A positive control the instrument must pass**: a natural row's tail, artificially
duplicated, has to read near 1.0. Without it a metric that always returns ~0 would print
"no degeneration" for any input, which is exactly the shape of failure this session has
hit six times today.

**12 steps, not 4.** The first run drew 32 rows and found 1 capped, so it refused: a
capped-vs-natural ratio on one row is that row's property. The tail probe measured the
capped rate at 11/160 = 6.9%, which puts 32 rows at an expected 2.2 -- the refusal was the
design working, and the sample size was chosen without consulting the rate it needed to
resolve. 12 steps is 96 rows, expected 6.6 capped, and the probe still refuses below 3.

Run:
  scripts/pod_run.sh --wait degen <card> -- python3 scripts/probe_capped_degeneration.py
"""
import argparse
import gzip
import json
import os
import sys
from dataclasses import replace

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

_PROMPTS = "/work/p1_gsm8k_train.jsonl"


def recurrence(toks: list[int], window: int, n: int) -> float | None:
    """Fraction of the last `window` tokens' n-grams that also occur in the window
    before it. None when the row is shorter than two windows."""
    if len(toks) < 2 * window + n:
        return None
    prev = {tuple(toks[i:i + n]) for i in range(len(toks) - 2 * window,
                                                len(toks) - window - n + 1)}
    tail = [tuple(toks[i:i + n]) for i in range(len(toks) - window, len(toks) - n + 1)]
    return sum(g in prev for g in tail) / len(tail)


def gzip_ratio(text: str) -> float:
    raw = text.encode()
    return len(raw) / max(len(gzip.compress(raw, 6)), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--gen", type=int, default=6144)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--ngram", type=int, default=20)
    ap.add_argument("--ratio", type=float, default=2.0,
                    help="capped/natural recurrence at or above this is degeneration")
    ap.add_argument("--prompts", default=_PROMPTS)
    ap.add_argument("--out", default="/work/capped_degeneration.json")
    args = ap.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model, _qwen38_tokenizer
    from tilerl.engine import build_engine
    from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
    from tilerl.prompt import render_chat
    from tilerl.prompt import sampling as build_sampling
    from tilerl.train import _drain, untruncated

    tok = _qwen38_tokenizer()
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            if len(prompts) >= args.steps:
                break
            row = json.loads(line)
            text = row.get("question") or row.get("prompt") or row.get("text")
            if text:
                # Through render_chat, as cli.py:611 does. The tail probe fed the bare
                # question and the model continued the document to the cap: mean 1083
                # tokens against 322 measured through the template.
                prompts.append(tok.encode(render_chat([("user", text)], False)))
    if len(prompts) < args.steps:
        raise SystemExit(f"{args.prompts}: {len(prompts)} usable prompts, need {args.steps}")
    # Assert the template is in the prompt, rather than noticing afterwards that the
    # lengths look wrong. The first run's bare `tok.encode(question)` produced completions
    # that read like a long tail and were the model continuing a document (0a, 2026-09-08).
    if not all(set(tok.encode("<|im_start|>")) <= set(p) for p in prompts):
        raise SystemExit("a prompt is missing the <|im_start|> the chat template opens "
                         "with, so the model is being handed a bare document and its "
                         "completion lengths are not the rollout's")

    base = untruncated(build_sampling(tok, False, args.gen, seed=0))
    if not base.stop_token_ids:
        raise SystemExit("stop_token_ids is empty, so no row can terminate naturally and "
                         "the control arm would be empty by construction")

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
    ctx = args.gen + max(map(len, prompts)) + 64
    engine = build_engine(cfg, model, backend, num_slots=args.group, max_batch=args.group,
                          num_blocks=-(-ctx // BLOCK_TOKENS) * args.group + args.group,
                          max_total_tokens=max(ctx, 8192),
                          decode_graph=True, prefix_store=NoPrefixStore())

    rows = []
    for step, prompt in enumerate(prompts):
        ids = [engine.submit(prompt, replace(base, seed=step * args.group + g))
               for g in range(args.group)]
        done = _drain(engine, ids, "degeneration probe")
        for g, i in enumerate(ids):
            c = list(done[i])
            rows.append({"step": step, "g": g, "len": len(c),
                         "capped": len(c) >= args.gen,
                         "recur": recurrence(c, args.window, args.ngram),
                         "gzip": gzip_ratio(tok.decode(c[-500:])) if len(c) >= 500 else None,
                         "tail": tok.decode(c[-300:])})
        print(f"step {step}: lens {sorted(r['len'] for r in rows[-args.group:])}", flush=True)

    def arm(capped):
        return [r for r in rows if r["capped"] is capped and r["recur"] is not None]

    cap, nat = arm(True), arm(False)
    print(f"\n{len(cap)} capped rows, {len(nat)} natural rows with >= "
          f"{2 * args.window + args.ngram} tokens")
    if len(cap) < 3 or len(nat) < 3:
        raise SystemExit(
            f"need 3+ rows per arm and have {len(cap)}/{len(nat)}: with fewer, a ratio "
            "between the two arms is one row's property, not the population's")

    # Positive control: duplicate a natural row's tail. A metric that cannot read a real
    # loop would print a low number for every input, including a looping one.
    src = max(nat, key=lambda r: r["len"])
    dup_src = tok.encode(src["tail"])[-args.window:]
    control = recurrence(dup_src * 3, args.window, args.ngram)
    print(f"positive control (a real tail repeated 3x): recurrence {control:.3f}")
    if control < 0.9:
        raise SystemExit(f"the metric reads {control:.3f} on text that IS a loop, so a low "
                         "number on the capped rows would prove nothing")

    def mean(xs):
        return sum(xs) / len(xs)

    rc, rn = mean([r["recur"] for r in cap]), mean([r["recur"] for r in nat])
    gc = mean([r["gzip"] for r in cap if r["gzip"]])
    gn = mean([r["gzip"] for r in nat if r["gzip"]])
    print(f"\n{'arm':>8} {'rows':>5} {'recurrence':>11} {'gzip ratio':>11}")
    print(f"{'capped':>8} {len(cap):>5} {rc:>11.3f} {gc:>11.2f}")
    print(f"{'natural':>8} {len(nat):>5} {rn:>11.3f} {gn:>11.2f}")
    print(f"{'ratio':>8} {'':>5} {rc / max(rn, 1e-9):>11.2f}x {gc / gn:>10.2f}x")
    verdict = ("degeneration: the capped rows repeat whole spans the natural rows do not"
               if rc >= args.ratio * rn else
               "not degeneration by this criterion: the capped rows recur no more than "
               "this run's own naturally-terminated rows")
    print(f"\n{rc:.3f} vs {rn:.3f}, threshold {args.ratio}x -> {verdict}")
    print(f"\nlongest capped row's last 300 tokens:\n"
          f"{max(cap, key=lambda r: r['len'])['tail']!r}")

    with open(args.out, "w") as f:
        json.dump({"window": args.window, "ngram": args.ngram, "gen": args.gen,
                   "control": control, "capped_recur": rc, "natural_recur": rn,
                   "capped_gzip": gc, "natural_gzip": gn, "rows": rows}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    # The two metrics, before they are pointed at a card. `template` is the case the
    # whole probe turns on: a repeating frame with a changing number must read 0, or a
    # worked solution would be reported as a loop.
    import random
    assert recurrence(list(range(50)) * 20, 200, 20) == 1.0
    frame = [t for i in range(200) for t in (900, 901, 902, 1000 + i, 903)]
    assert recurrence(frame, 200, 20) == 0.0
    random.seed(0)
    assert recurrence([random.randrange(50000) for _ in range(2000)], 200, 20) == 0.0
    assert recurrence([1, 2, 3], 200, 20) is None
    assert gzip_ratio("the answer is 12. " * 60) > 5 * gzip_ratio(
        "Natalia sold 48 clips in April and half as many in May, so 72 altogether." * 3)
    raise SystemExit(main())
