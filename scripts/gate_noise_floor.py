"""What effect size each P1 exit gate can distinguish from noise at its own n.

The gates at `cli.py:831-837` compare two evaluations with a bare inequality on two
scalars. This prices what that costs, and the headline is that the roadmap's own
target does not clear the floor of the test the gate currently applies.

`_write_eval_rows` (`cli.py:358`) says why: "One JSON row per problem, so two arms
over the same set can be compared paired. P1 fell back to the unpaired interval
because only totals were kept." The rows are written now (`eval.py:104` carries
`correct`), both arms run the same slice (`cli.py:479`) at temperature 0, so the
paired test is available and the gate does not use it.

The paired SE cannot be stated in advance: it needs the discordant counts b and c
from a real run. What can be stated in advance is which test applies, and that the
unpaired formula overstates the floor. Both are below.

Run: uv run python scripts/gate_noise_floor.py
"""

import math
import random

Z80 = 0.8416  # one-sided 80% power
Z95 = 1.6449  # one-sided 5% significance


def unpaired_se(p: float, n: int) -> float:
    """SE of (after - before) as two independent binomials -- what P1 fell back to."""
    return math.sqrt(2 * p * (1 - p) / n)


def paired_se(flip_rate: float, n: int) -> float:
    """McNemar SE of the paired difference: sqrt(b + c) / n.

    Only the questions whose correctness CHANGED carry information; the ones both
    arms get right cancel. `flip_rate` is (b + c) / n, the discordant fraction --
    the quantity a run has to report and today's manifest does not.
    """
    return math.sqrt(flip_rate * n) / n


def min_detectable(se: float) -> float:
    """Effect size needed for 80% power at 5% significance, one-sided."""
    return (Z95 + Z80) * se


def false_pass_zero_threshold(p: float, n: int, trials: int = 200_000, seed: int = 0) -> float:
    """P(after > before) when nothing changed -- what `after > before` scores PASS.

    Simulated rather than assumed 0.5 because the discrete binomial puts real mass on
    the tie, and a tie fails a strict `>`. That tie mass is the only thing holding the
    gate under a coin flip, and it shrinks as n grows.
    """
    rng = random.Random(seed)
    return sum(sum(rng.random() < p for _ in range(n)) > sum(rng.random() < p for _ in range(n))
               for _ in range(trials)) / trials


def main() -> None:
    n, p_gsm, p_mmlu = 500, 0.40, 0.50  # roadmap's held-out set size and rough rates

    print("units first -- gsm8k_{tag} is a COUNT, mmlu_{tag} is a FRACTION:\n")
    print(f"  gsm8k_improves passes on +1 question = {100 / n:.1f}pt, its real threshold")
    print("  `gsm8k_after - gsm8k_before >= 0.05` on ints is `>= 1`: a NO-OP, not +5pt")
    print(f"  the roadmap's +5pt encodes as `>= 0.05 * total` = {int(0.05 * n)} questions")

    print("\ntoday's test -- two scalars, unpaired:\n")
    print(f"{'gate':<16} {'threshold':>16} {'SE':>7} {'min detectable':>15}  verdict")
    for name, rate, thr, label in (
        ("gsm8k_improves", p_gsm, 1 / n, "+1 question"),
        ("gsm8k roadmap", p_gsm, 0.05, ">= 25 questions"),
        ("mmlu_holds", p_mmlu, 0.03, "within -3 pt"),
    ):
        se = unpaired_se(rate, n)
        mde = min_detectable(se)
        v = "BELOW THE FLOOR" if abs(thr) < mde else "resolvable"
        print(f"{name:<16} {label:>16} {100 * se:>6.2f}pt {100 * mde:>14.2f}pt  {v}")

    fp = false_pass_zero_threshold(p_gsm, n)
    print(f"\n`after > before` passes {100 * fp:.1f}% of the time when nothing changed.")
    print("  A near-zero threshold is a coin flip at any n and any SE -- the tie mass is")
    print(f"  the only thing under 50%, and it is worth {100 * (0.5 - fp):.1f}pt here.")

    print("\nthe test the run's own rows already support -- McNemar, paired:\n")
    print(f"{'discordant (b+c)/n':>20} {'SE':>7} {'min detectable':>15}  +5pt")
    for flip in (0.30, 0.20, 0.10, 0.05, 0.02):
        se = paired_se(flip, n)
        mde = min_detectable(se)
        print(f"{flip:>19.0%} {100 * se:>6.2f}pt {100 * mde:>14.2f}pt  "
              f"{'resolvable' if mde < 0.05 else 'below the floor'}")
    print("\n  The floor scales with the FLIP rate, not the accuracy: a run that moves few")
    print("  questions resolves a small gain, and the unpaired formula cannot see that.")
    print("  (b + c) is not knowable in advance -- it is what a run must report.")
    print("  Caveat: a cached before-arm (cli.py:562) is still paired, but only as")
    print("  well as its cache key, so a paired SE from a cache-hit run is a claim")
    print("  about the key as much as about the model.")


if __name__ == "__main__":
    main()

    n = 500
    # A near-zero threshold is a coin flip. This is the number to quote about
    # `after > before`, and it holds regardless of which test is applied.
    fp = false_pass_zero_threshold(0.40, n)
    assert 0.45 < fp < 0.50, fp

    # The units trap, asserted because it is the one that survives a plausible fix:
    # gsm8k_{tag} is an int count, so a `>= 0.05` threshold is `>= 1` -- the same gate.
    assert (1 >= 0.05) and (int(0.05 * n) == 25), "the roadmap's +5pt is 25 questions"

    # The finding: the roadmap's +5 pt does NOT clear the floor of the test the gate
    # applies today. It clears the paired floor once the discordant rate is under ~13%.
    assert min_detectable(unpaired_se(0.40, n)) > 0.05, "unpaired floor no longer blocks +5pt"
    assert min_detectable(paired_se(0.10, n)) < 0.05, "paired at 10% flips should resolve +5pt"

    # Negative control on the instrument: the paired SE must fall with the flip rate.
    # If it tracked accuracy instead, these would be equal.
    assert paired_se(0.30, n) > paired_se(0.05, n)
    print("\nself-check: 4 asserts + 1 control passed")
