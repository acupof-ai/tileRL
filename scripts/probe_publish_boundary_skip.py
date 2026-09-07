"""A spec chain skips 16-boundaries unless its length divides 16, so publishes are not gen/16.

`errors/2026-09-07-a-prompts-own-publishes-evict-its-shared-prefix.md` prices the churn at
"64 decode publishes at gen 1024" -- gen/BLOCK_TOKENS. That counts boundaries that EXIST in
the token range, not boundaries a tick lands on. `engine.py:1309` guards the publish with
`i == last and materialized % BLOCK_TOKENS == 0`: only the chain's LAST accepted token is
tested, so a chain whose length does not divide BLOCK_TOKENS steps over boundaries silently.

Live V100 serve child (pid 2977356, 6h27m up, `--depth 1`), /health, no card taken:

    tokens_generated 5410   decode_forwards 3047   -> 1.776 tok/forward
    spec_accepted 2365 / spec_drafted 3047         -> p = 0.776
    boundaries in range     5410 // 16 = 338
    prefix_published        188                    -> 44% never published

The dependence is on DIVISIBILITY, not on tok/forward: at a fixed chain length the count is
338 when the length divides 16 (1, 2, 4, 8) and about 338/length when it does not (3 -> 112,
5 -> 67, 7 -> 48).

At `--depth 1` a tick emits 1 token, plus the drafted one when it is accepted, so the chain
length is Bernoulli: 2 with probability p, else 1. That is not a guess -- 1 + 0.776 = 1.776
reproduces the measured tok/forward exactly. Simulating it gives 163..214 landings (mean
190.6) against the observed 188, 1.4% off the mean.

Two wrong models were tried first and both are recorded, because each looked right:
  * `gen/16/tok_per_fwd` = 190.4, which matches 188 to 1.3% -- and is arithmetic
    coincidence, since no FIXED chain length yields 1.776.
  * uniform 1..W chains, which put 188 in the W=3 bracket -- but the server runs
    `--depth 1`, read from its argv, so W=3 was never possible.
Both agreed with the observation to about 1%. Agreement at one point is not a model.

  TILERL_TARGET=cpu python3 scripts/probe_publish_boundary_skip.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tilerl.kv_cache import BLOCK_TOKENS

#: Live reading, V100 serve child pid 2977356 at 2026-09-07, /health plus its argv.
LIVE = {"tokens_generated": 5410, "decode_forwards": 3047, "prefix_published": 188,
        "spec_drafted": 3047, "spec_accepted": 2365, "depth": 1}


def landings(gen: int, chains: list[int], *, last_only: bool = True) -> int:
    """Publishes over `gen` tokens, emitting `chains` accepted tokens per tick, cycling.

    `last_only` is the shipped guard (`i == last`): only a chain's final token is tested.
    False tests every accepted token, which is what gen/BLOCK_TOKENS assumes.
    """
    seq = out = 0
    i = 0
    while seq < gen:
        n = min(chains[i % len(chains)], gen - seq)
        i += 1
        for j in range(n):
            seq += 1
            if last_only and j != n - 1:
                continue
            if seq % BLOCK_TOKENS == 0:
                out += 1
    return out


def main() -> int:
    gen, pub = LIVE["tokens_generated"], LIVE["prefix_published"]
    tpf = gen / LIVE["decode_forwards"]
    p = LIVE["spec_accepted"] / LIVE["spec_drafted"]
    exist = gen // BLOCK_TOKENS
    print(f"live: {gen} tokens / {LIVE['decode_forwards']} forwards = {tpf:.3f} tok/fwd, "
          f"depth {LIVE['depth']}, accept {p:.3f}")
    print(f"      {exist} boundaries exist, {pub} published -> "
          f"{(1 - pub / exist) * 100:.0f}% skipped\n")

    print(f"{'chain':>6} {'i == last':>10} {'every token':>12} {'gcd(chain,16)':>14}")
    for c in range(1, 9):
        print(f"{c:>6} {landings(gen, [c]):>10} {landings(gen, [c], last_only=False):>12} "
              f"{math.gcd(c, BLOCK_TOKENS):>14}")

    # The mechanism is divisibility. Assert both halves, or the table above is a display.
    for c in (1, 2, 4, 8):
        assert landings(gen, [c]) == exist, (
            f"chain {c} divides {BLOCK_TOKENS} but skipped boundaries -- the guard is not "
            "what this probe models"
        )
    for c in (3, 5, 7):
        assert landings(gen, [c]) < exist * 0.6, (
            f"chain {c} does not divide {BLOCK_TOKENS} yet published nearly every boundary, "
            "so `i == last` is not skipping and the live 188 needs another cause"
        )

    # The server's own distribution: depth 1 emits 1 token plus the draft when accepted.
    # 1 + p must reproduce the measured tok/forward, or the chain model is not this server's.
    assert abs((1 + p) - tpf) < 0.01 * tpf, (
        f"1 + accept ({1 + p:.3f}) does not reproduce tok/forward ({tpf:.3f}), so a "
        f"Bernoulli chain at depth {LIVE['depth']} is the wrong shape for this run"
    )
    random.seed(7)
    obs = []
    for _ in range(400):
        chains = []
        while sum(chains) < gen + 10:
            chains.append(2 if random.random() < p else 1)
        obs.append(landings(gen, chains))
    lo, hi, mean = min(obs), max(obs), sum(obs) / len(obs)
    print(f"\nBernoulli(p={p:.3f}) chains: {lo}..{hi} landings, mean {mean:.1f}; "
          f"observed {pub} ({abs(mean - pub) / pub * 100:.1f}% off the mean)")
    assert lo <= pub <= hi, (
        f"the observed {pub} falls outside {lo}..{hi}, so the boundary skip does not explain "
        "the live publish count"
    )
    # Negative control: the model must NOT also accommodate the count it replaces.
    assert not lo <= exist <= hi, (
        f"gen/{BLOCK_TOKENS} = {exist} also falls inside {lo}..{hi}, so this simulation "
        "cannot discriminate the skip from the model it corrects"
    )

    print(f"\nThe entry states {1024 // BLOCK_TOKENS} decode publishes at gen 1024 -- the "
          f"boundaries that exist.\nThe count a tick LANDS on is lower: this server published "
          f"{pub} of {exist}, so {(1 - pub / exist) * 100:.0f}% never\nhappened. One "
          f"conversation's churn is about half what the entry prices, and the direction\n"
          f"matters -- the fix has less to throttle than it was credited with.")
    print("\nNo closed form is claimed: the count depends on the chain-length distribution, "
          "which\ndepends on depth and acceptance. Two earlier models matched 188 to ~1% and "
          "were both\nwrong; the accept rate reproducing tok/forward is what pins this one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
