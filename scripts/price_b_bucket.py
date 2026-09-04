"""Does bucketing B cost anything? Priced on the RUNG model, not a per-row tick share.

My #74 break-even table charged each padding row `TICK_MS / b * 3.3` -- 39-70 ms.
Both inputs were wrong:

- 3.3x is the ratio of the MARGINAL launched row (7.53 ms) to the MARGINAL useful row
  (2.29 ms) in the fit `tick_ms = 25.6 + 7.53*launched + 2.29*useful`
  (errors/2026-09-03-batching-is-non-monotone-padding-rows-cost-3x.md). It is not a
  multiplier on a tick share.
- A tick costs its RUNG, not its rows: rung 8 with 3 of 8 idle measured 82.15 ms
  against 83.40 ms fully packed -- 60% more useful rows for 1.5% more time
  (wins/2026-09-04-rung-cost-not-useful-rows.md).

So padding B costs nothing UNLESS it pushes M = B*W onto a higher rung. This prices it
that way: for each observed B, does rounding B up to a power of two change the rung,
and if so what does the rung step cost.
"""

LADDER = (1, 2, 4, 8, 32)
COMPILE_MS = 1108.0        # 15509 ms / 14 compiles, served first visit (#73)
# Marginal cost of a launched row, fitted at B=4/B=8 on rung 32 (#41). Used only to
# price a RUNG STEP, which is what a bucket can actually cause.
MS_PER_LAUNCHED = 7.53
TICKS_PER_REQUEST = 124    # /health: decode_forwards 124 over 5 prefills

# Served config is depth 1 -> W = 1+depth = 2. The B values the cache showed (1-7).
W = 2
OBSERVED_B = [1, 2, 3, 4, 5, 6, 7]
# Widths to sweep: engine.py:849 WARNS on an off-rung width rather than clamping, so
# depth 4 (W=5) is reachable and has to be priced too, not just the served W.
SWEEP_W = range(1, 9)


def rung(m: int) -> int:
    """The rung M lands on; past the top the dispatch chunks at 32 (engine.py:862)."""
    top = max(LADDER)
    if m > top:
        return top * -(-m // top)
    return next(r for r in LADDER if r >= m)


def bucket(b: int) -> int:
    return 1 << (b - 1).bit_length()


print(f"W = {W} (depth 1), rung = ceil(B*W) on {LADDER}\n")
print(f"{'B':>3} {'M=B*W':>6} {'rung':>5} {'B->':>4} {'M2':>4} {'rung2':>6} "
      f"{'rung step':>10} {'added ms/tick':>14} {'break-even ticks':>17}")

losses = 0
for b in OBSERVED_B:
    m, b2 = b * W, bucket(b)
    m2 = b2 * W
    r, r2 = rung(m), rung(m2)
    step = r2 - r
    if step == 0:
        print(f"{b:>3} {m:>6} {r:>5} {b2:>4} {m2:>4} {r2:>6} {0:>10} "
              f"{'free':>14} {'never':>17}")
        continue
    added = step * MS_PER_LAUNCHED
    be = COMPILE_MS / added
    losses += be < TICKS_PER_REQUEST
    print(f"{b:>3} {m:>6} {r:>5} {b2:>4} {m2:>4} {r2:>6} {step:>10} "
          f"{added:>13.1f}  {be:>16.1f}")

print(f"\nA request runs ~{TICKS_PER_REQUEST} ticks, so a break-even under that is a LOSS.")
print(f"{losses} of {len(OBSERVED_B)} observed B values lose.")

print("\nNot specific to W=2 -- every width the engine can reach:\n")
for w in SWEEP_W:
    bad = [b for b in OBSERVED_B if rung(b * w) != rung(bucket(b) * w)]
    star = "   <- served (depth 1)" if w == W else ""
    print(f"  W={w}: {'crosses a rung at B=' + str(bad) if bad else 'free for every B'}{star}")
print("\nOnly W=5 and W=6 cross a rung, and W=5 is depth 4 -- which engine.py:849 warns\n"
      "about but does NOT clamp, so it is reachable. Bucketing B is free at every width\n"
      "the ladder endorses and costs only where the ladder already says the width is\n"
      "wrong. The first table's 16-66 tick break-evens came from charging a per-row tick\n"
      "share that the rung model says is not paid at all.")

# Checked against the two entries this borrows from, so a wrong constant fails here
# rather than in a verdict: the rung ladder and the served width are both load-bearing.
assert rung(6) == 8 and rung(10) == 32, "LADDER moved; the free-padding result may not hold"
assert bucket(3) == 4 and bucket(5) == 8, "bucket() must round up to a power of two"
assert rung(3 * W) == rung(bucket(3) * W), "B=3 must stay on its rung at W=2"
# The sweep must actually discriminate: a version where nothing ever crosses would print
# "free" everywhere and prove nothing.
assert any(rung(b * 5) != rung(bucket(b) * 5) for b in OBSERVED_B), \
    "W=5 must cross a rung, or the sweep is not a test"
