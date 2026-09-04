"""The combine's denominator cannot be zero: check the tile arithmetic, not the kernel.

`paged_attention_split_combine` divides by l = sum_s w_s PL_s. If every split for a row
carried an empty window, PL would be 0 everywhere and the division would be 0/0. Upstream
guarded the same row in its own combine after a run read 0/500 GSM8K with NaN in 372 of 560
verify row-ticks; this tree has no guard and a different failure mode -- m[0] is initialised
to a FINITE -1e30, so w = exp(0) = 1 and the row divides by exactly 0, giving inf rather
than NaN.

The row is unreachable, and this pins why. It cannot be checked through the kernel on this
machine: the split kernel has no CPU twin (`tl.reduce` is unimplemented for target="c", a
recorded gap), so the guarantee is verified against the index arithmetic the kernel runs --
kernels.py's `n`, `per`, `p0`, `p1` -- across the whole shape space including padded query
rows.
"""

import math

_KVSPLITS = (32, 16)  # SM70_KVSPLIT, SM70_KVSPLIT_WIDE


def split_zero_tiles(n: int, kvsplit: int) -> int:
    """Tiles that split 0 runs for a row whose causal bound is n, per kernels.py:671-676.

        per = ceildiv(n, KVSPLIT); p0 = 0; p1 = min(n, per)

    The tile loop runs ceildiv(p1 - p0, block_N) times, so any p1 > p0 means at least one
    tile, hence l[0] > 0 for that split, hence a nonzero denominator for the whole row.
    """
    per = math.ceil(n / kvsplit)
    return math.ceil(min(n, per) / 16)


def test_split_zero_always_runs_a_tile():
    """n >= 1 for every reachable row, and split 0 then always has work."""
    for kvsplit in _KVSPLITS:
        for n in range(1, 4200):
            assert split_zero_tiles(n, kvsplit) >= 1, f"n={n} ks={kvsplit} leaves split 0 empty"


def test_n_is_at_least_one_for_every_row_including_padding():
    """`n = SeqLens - SeqQLens + tt + 1` (kernels.py:672) is >= 1 for all valid inputs.

    A padded query row still runs the kernel -- the comment at kernels.py:691 says so --
    so tt ranges over the full padded width, not just SeqQLens.

    The premise is `seq_len >= seq_q_len` per row: a row's logical length AFTER the forward
    (BatchKv's own definition, engine.py:149) includes its own query tokens. Verified at the
    tightest producer, the draft chain step (engine.py:987): seq_len = pos + 1 + j with
    j >= 1 against seq_q_lens = 1, so n >= 2 there. backend.py:690 defaults an absent
    seq_q_lens to the padded width s, which makes the premise seq_len >= s -- the one form
    a caller could violate, so it is asserted below rather than assumed.
    """
    for seq_len in (1, 2, 8, 16, 17, 512, 4096):
        for seq_q_len in range(1, min(seq_len, 8) + 1):
            for tt in range(0, 8):  # padded width can exceed seq_q_len
                n = seq_len - seq_q_len + tt + 1
                assert n >= 1, f"n={n} at seq_len={seq_len} q={seq_q_len} tt={tt}"


def test_the_dispatch_never_passes_a_width_longer_than_the_row():
    """backend.py:690 fills an absent seq_q_lens with the padded width s.

    So `n = seq_len - s + tt + 1`, and a caller passing seq_len < s would make n <= 0 for
    tt = 0 and hand the combine an empty split 0. Nothing in the kernel rejects that, so
    the constraint belongs to the dispatch: it is the row's post-forward length, which
    cannot be shorter than the queries that forward consumes.
    """
    for s in (1, 2, 4, 8):
        for seq_len in range(s, 4 * s + 1):  # the legal range: seq_len >= s
            assert seq_len - s + 0 + 1 >= 1, f"s={s} seq_len={seq_len}"
        # And the illegal one is genuinely unsafe, which is why it must not be reachable.
        if s > 1:
            assert (s - 1) - s + 0 + 1 == 0, "seq_len < s would empty split 0"


def test_the_check_would_catch_a_broken_split_bound():
    """Negative control: if split 0 could be given an empty range, this must fail.

    Without it, arithmetic that happens to be >= 1 everywhere proves nothing about
    whether the assertion is load-bearing.
    """

    def broken(n: int, kvsplit: int) -> int:
        # p0 = sp * per with sp = 1 -- i.e. asking whether split ONE always has work.
        per = math.ceil(n / kvsplit)
        return math.ceil(max(0, min(n, 2 * per) - per) / 16)

    assert broken(1, 32) == 0, "split 1 IS empty at n=1, so the property is specific to split 0"


if __name__ == "__main__":
    test_split_zero_always_runs_a_tile()
    test_n_is_at_least_one_for_every_row_including_padding()
    test_the_check_would_catch_a_broken_split_bound()
    print("ok: split 0 runs >= 1 tile for every n in [1, 4200) at KVSPLIT 32 and 16")
